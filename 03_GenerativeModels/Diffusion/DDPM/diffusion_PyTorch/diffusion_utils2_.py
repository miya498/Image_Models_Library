import torch
import torch.nn.functional as F
import numpy as np


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    平均と対数分散でパラメータ化された正規分布間のKLダイバージェンスを計算
    """
    return 0.5 * (
        -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
        + torch.pow(mean1 - mean2, 2) * torch.exp(-logvar2)
    )


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    """
    ウォームアップ用のベータスケジュールを計算
    """
    betas = np.full(num_diffusion_timesteps, beta_end, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    """
    ベータスケジュールを取得
    """
    if beta_schedule == 'quad':
        # 二乗スケジュール
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        # 線形スケジュール
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        # 10%のウォームアップスケジュール
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        # 50%のウォームアップスケジュール
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        # 一定のベータ値
        betas = np.full(num_diffusion_timesteps, beta_end, dtype=np.float64)
    elif beta_schedule == 'jsd':
        # Jensen-Shannon Divergenceスケジュール
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(f"ベータスケジュール '{beta_schedule}' は未実装です。")
    return betas


class GaussianDiffusion:
    """
    Gaussian Diffusionモデルのユーティリティ

    Arguments:
    - モデルが予測するもの (x_{t-1}, x_0, またはepsilon)
    - 使用する損失関数 (KLまたはMSE)
    - p(x_{t-1}|x_t) の分散の種類 (学習済み, 固定)
    - デコーダの種類とその損失の重み
    """

    def __init__(self, betas, model_mean_type, model_var_type, loss_type):
        self.model_mean_type = model_mean_type  # "xprev", "xstart", "eps"
        self.model_var_type = model_var_type  # "learned", "fixedsmall", "fixedlarge"
        self.loss_type = loss_type  # "kl", "mse"

        # ベータ値の検証と初期化
        assert isinstance(betas, np.ndarray)
        self.betas = torch.tensor(betas, dtype=torch.float64)
        assert (self.betas > 0).all() and (self.betas <= 1).all()
        self.num_timesteps = len(betas)

        # アルファ値の計算
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(torch.tensor(alphas), dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # posterior q(x_{t-1} | x_t, x_0) の計算
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(torch.tensor(alphas)) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, a, t, x_shape):
        """
        指定されたタイムステップの係数をバッチサイズに応じて抽出
        """
        batch_size = t.shape[0]
        out = a[t].to(x_shape.device)  # a からタイムステップ t に基づいて値を取得
        return out.view(batch_size, *((1,) * (len(x_shape.shape) - 1)))

    def q_mean_variance(self, x_start, t):
        """
        拡散過程 q(x_t | x_0) の平均と分散を計算
        """
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        データにノイズを加えて拡散 (t == 0 の場合、1ステップ分の拡散)
        """
        if noise is None:
            noise = torch.randn_like(x_start)  # ノイズを生成
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        posterior q(x_{t-1} | x_t, x_0) の平均と分散を計算
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, x, t, clip_denoised=True):
        """
        モデルの予測平均と分散を計算
        """
        model_output = denoise_fn(x, t)  # モデルの出力

        if self.model_var_type == 'fixedsmall':
            variance = self._extract(self.posterior_variance, t, x.shape)
            log_variance = self._extract(self.posterior_log_variance_clipped, t, x.shape)
        elif self.model_var_type == 'fixedlarge':
            variance = self._extract(self.betas, t, x.shape)
            log_variance = torch.log(variance)
        else:
            raise NotImplementedError(f"モデル分散タイプ '{self.model_var_type}' は未実装です。")

        if self.model_mean_type == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
            if clip_denoised:
                pred_xstart = pred_xstart.clamp(-1.0, 1.0)
            model_mean = (
                self._extract(self.posterior_mean_coef1, t, x.shape) * pred_xstart +
                self._extract(self.posterior_mean_coef2, t, x.shape) * x
            )
        else:
            raise NotImplementedError(f"モデル平均タイプ '{self.model_mean_type}' は未実装です。")

        return model_mean, variance, log_variance, pred_xstart

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        x_t（ノイズが加えられたデータ）と予測ノイズ eps から元データ x_0 を推定
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        """
        x_t と予測された x_{t-1} から元データ x_0 を推定
        """
        return (
            self._extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            self._extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
        )

    def p_sample(self, denoise_fn, x, t, noise_fn=torch.randn, clip_denoised=True):
        """
        逆拡散プロセスで1ステップ分のサンプルを生成
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_fn(x.shape).to(x.device)
        # t == 0 の場合ノイズは不要
        nonzero_mask = (t != 0).float().view(-1, *[1] * (len(x.shape) - 1))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def p_sample_loop(self, denoise_fn, shape, noise_fn=torch.randn):
        """
        逆拡散プロセスを全タイムステップに渡って実行し、最終的なサンプルを生成
        """
        device = next(denoise_fn.parameters()).device
        img = noise_fn(shape).to(device)  # 初期ノイズを生成
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            img = self.p_sample(denoise_fn, img, t_batch)
        return img

    def p_sample_loop_progressive(self, denoise_fn, shape, noise_fn=torch.randn):
        """
        各タイムステップでの中間結果を追跡しながらサンプリング
        """
        device = next(denoise_fn.parameters()).device
        img = noise_fn(shape).to(device)  # 初期ノイズを生成
        imgs = [img]  # 中間結果を保存
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            img = self.p_sample(denoise_fn, img, t_batch)
            imgs.append(img)
        return imgs

    def training_losses(self, denoise_fn, x_start, t, noise=None):
        """
        トレーニング損失を計算
        """
        device = x_start.device
        if noise is None:
            noise = torch.randn_like(x_start).to(device)

        # データにノイズを加える
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 損失の計算
        if self.loss_type == 'mse':  # 平均二乗誤差
            if self.model_mean_type == 'eps':  # モデルがノイズを予測
                target = noise
            elif self.model_mean_type == 'xstart':  # モデルが x_0 を予測
                target = x_start
            else:
                raise NotImplementedError(f"未知のモデルタイプ: {self.model_mean_type}")
            pred = denoise_fn(x_t, t)
            loss = F.mse_loss(pred, target, reduction="mean")
        else:
            raise NotImplementedError(f"未知の損失タイプ: {self.loss_type}")

        return loss
    def _prior_bpd(self, x_start):
        """
        事前分布（Prior Distribution）に基づいたBPD（Bits Per Dimension）を計算
        """
        B, T = x_start.shape[0], self.num_timesteps
        qt_mean, _, qt_log_variance = self.q_mean_variance(
            x_start, t=torch.full((B,), T - 1, dtype=torch.long, device=x_start.device)
        )
        kl_prior = normal_kl(qt_mean, qt_log_variance, mean2=torch.zeros_like(qt_mean), logvar2=torch.zeros_like(qt_mean))
        return kl_prior.mean() / np.log(2.0)  # KLダイバージェンスをビット毎次元に変換

    def calc_bpd_loop(self, denoise_fn, x_start, clip_denoised=True):
        """
        各タイムステップでのBPD（Bits Per Dimension）を計算し、最終的な結果を集計
        """
        B, T = x_start.shape[0], self.num_timesteps
        device = x_start.device

        # タイムステップループ内の計算
        def loop_body(t, cur_vals, cur_mse):
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)
            # 現在のタイムステップでの q(x_t | x_0) をサンプリング
            x_t = self.q_sample(x_start, t_batch)

            # VLB（変分下限）項を計算
            new_vals, pred_xstart = self._vb_terms_bpd(
                denoise_fn, x_start, x_t, t_batch, clip_denoised=clip_denoised, return_pred_xstart=True
            )

            # MSEを計算
            new_mse = F.mse_loss(pred_xstart, x_start, reduction="none").mean(dim=(1, 2, 3))
            # 各タイムステップの値を更新
            cur_vals[:, t] = new_vals
            cur_mse[:, t] = new_mse

            return cur_vals, cur_mse

        # BPD計算の初期化
        terms_bpd = torch.zeros((B, T), device=device)
        mse = torch.zeros((B, T), device=device)

        # 各タイムステップでの計算
        for t in reversed(range(T)):
            terms_bpd, mse = loop_body(t, terms_bpd, mse)

        # 事前分布（Prior Distribution）に基づくBPD
        prior_bpd = self._prior_bpd(x_start)
        total_bpd = terms_bpd.sum(dim=1) + prior_bpd

        return total_bpd, terms_bpd, prior_bpd, mse

def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    離散化されたガウス分布の対数尤度を計算
    """
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_plus = torch.sigmoid(plus_in)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
    log_probs = torch.where(
        x < -0.999, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12)))
    )
    return log_probs


def _vb_terms_bpd(self, denoise_fn, x_start, x_t, t, clip_denoised=True, return_pred_xstart=False):
        """
        VLB（変分下限）に基づく各タイムステップのBPD項を計算
        """
        # 真の平均と分散
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start, x_t, t)

        # モデルの予測平均と分散
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, x_t, t, clip_denoised=clip_denoised
        )

        # KLダイバージェンスを計算
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=(1, 2, 3)) / np.log(2.0)  # ビット毎次元に変換

        # デコーダの負の対数尤度を計算
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=model_mean, log_scales=0.5 * model_log_variance
        )
        decoder_nll = decoder_nll.mean(dim=(1, 2, 3)) / np.log(2.0)

        # t == 0 の場合はデコーダNLLを使用、それ以外はKLダイバージェンスを使用
        output = torch.where(t == 0, decoder_nll, kl)

        return (output, pred_xstart) if return_pred_xstart else output