import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
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
        # JSDスケジュール (1/T, 1/(T-1), ..., 1)
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(f"Beta schedule '{beta_schedule}' not implemented")
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def noise_like(shape, noise_fn=torch.randn, repeat=False, device="cpu"):
    """
    指定された形状のノイズを生成
    """
    if repeat:
        noise = noise_fn((1, *shape[1:]), device=device).repeat(shape[0], 1, 1, 1)
    else:
        noise = noise_fn(shape, device=device)
    return noise

class GaussianDiffusion:
    """
    Gaussian DiffusionのPyTorchによる実装
    """

    def __init__(self, *, betas, loss_type, device="cpu"):
        self.device = device
        self.loss_type = loss_type

        # ベータ値をfloat64にキャストして検証
        assert isinstance(betas, np.ndarray)
        self.betas = torch.tensor(betas, dtype=torch.float64, device=self.device)
        assert (self.betas > 0).all() and (self.betas <= 1).all()

        # アルファ値とその累積積を計算
        alphas = 1.0 - self.betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # 必要なテンソルを初期化
        self.num_timesteps = len(self.betas)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float64, device=self.device)
        self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, dtype=torch.float64, device=self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # posterior q(x_{t-1} | x_t, x_0) の計算用
        posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = torch.tensor(posterior_variance, dtype=torch.float64, device=self.device)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        """
        タイムステップに応じた係数をバッチの形状に合わせて抽出
        """
        batch_size = t.shape[0]
        out = a[t].to(self.device)
        return out.view(batch_size, *((1,) * (len(x_shape) - 1)))

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
        データを拡散 (t = 0 のとき、1ステップ分の拡散)
        """
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        """
        x_t（ノイズが加えられたデータ）からノイズを予測し、元データ x_0 を推定
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        拡散後 q(x_{t-1} | x_t, x_0) の平均と分散を計算
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_losses(self, denoise_fn, x_start, t, noise=None):
        """
        トレーニング損失を計算
        """
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = denoise_fn(x_noisy, t)

        if self.loss_type == 'noisepred':
            loss = F.mse_loss(predicted_noise, noise, reduction="mean")
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")
        return loss

    def p_sample(self, denoise_fn, x, t, noise_fn=torch.randn, clip_denoised=True):
        """
        逆拡散プロセスによるサンプル生成
        """
        noise = noise_fn(x.shape).to(self.device)
        predicted_noise = denoise_fn(x, t)
        x_reconstructed = self.predict_start_from_noise(x, t, predicted_noise)

        if clip_denoised:
            x_reconstructed = torch.clamp(x_reconstructed, -1.0, 1.0)

        model_mean, _, model_log_variance = self.q_posterior(x_reconstructed, x, t)
        return model_mean + noise * torch.exp(0.5 * model_log_variance)

    def p_sample_loop(self, denoise_fn, shape, noise_fn=torch.randn):
        """
        逆拡散プロセスをタイムステップ全体でループ
        """
        img = noise_fn(shape).to(self.device)
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.tensor([t] * shape[0], device=self.device)
            img = self.p_sample(denoise_fn, img, t_batch)
        return img
    
    def p_sample_loop_trajectory(self, denoise_fn, shape, noise_fn=torch.randn):
        """
        中間状態を返しながらサンプル生成を実行
        各タイムステップでの生成された画像を可視化するために役立つ
        """
        img = noise_fn(shape).to(self.device)
        trajectory = [img.cpu().detach()]  # 初期状態を保存
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.tensor([t] * shape[0], device=self.device)
            img = self.p_sample(denoise_fn, img, t_batch)
            trajectory.append(img.cpu().detach())  # 中間状態を保存
        return trajectory

    def interpolate(self, denoise_fn, x1, x2, t, lam, noise_fn=torch.randn):
        """
        2つの画像 x1 と x2 を指定された比率 lam で補間し、逆拡散で再構築
        """
        assert x1.shape == x2.shape, "x1 と x2 の形状が一致している必要があります"
        t_batch = torch.tensor([t] * x1.shape[0], device=self.device)

        # 前方拡散を通じてノイズを追加
        xt1 = self.q_sample(x1, t_batch)
        xt2 = self.q_sample(x2, t_batch)

        # 潜在空間で線形補間
        xt_interp = (1 - lam) * xt1 + lam * xt2

        # 逆拡散プロセスを通じて再構築
        img = xt_interp
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.tensor([t] * x1.shape[0], device=self.device)
            img = self.p_sample(denoise_fn, img, t_batch)
        return img

    def sample(self, denoise_fn, batch_size, shape, noise_fn=torch.randn):
        """
        サンプル生成関数
        指定されたバッチサイズと形状でサンプルを生成
        """
        shape = (batch_size, *shape)
        return self.p_sample_loop(denoise_fn, shape, noise_fn=noise_fn)
