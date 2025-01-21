import contextlib
import io
import random
import time

import numpy as np
import torch
from PIL import Image

from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter

class SummaryWriter:
    """PyTorch用のサマリーロガー"""
    def __init__(self, dir, write_graph=True):
        self.writer = TorchSummaryWriter(log_dir=dir)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    def scalar(self, tag, value, step):
        """
        スカラー値をログに記録
        """
        self.writer.add_scalar(tag, value, global_step=step)

    def image(self, tag, image, step):
        """
        画像をログに記録
        """
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[:, :, None]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        self.writer.add_image(tag, image.transpose(2, 0, 1), global_step=step)

    def images(self, tag, images, step):
        """
        複数画像をタイル形式で記録
        """
        self.image(tag, tile_imgs(images), step=step)

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def tile_imgs(imgs, pad_pixels=1, pad_val=255, num_col=0):
        """
        複数画像をタイル形式に配置して1つの画像を作成
        """
        assert pad_pixels >= 0 and 0 <= pad_val <= 255

        imgs = np.asarray(imgs)
        assert imgs.dtype == np.uint8
        if imgs.ndim == 3:
            imgs = imgs[..., None]
        n, h, w, c = imgs.shape
        assert c in (1, 3), 'チャンネル数は1または3である必要があります'

        if num_col <= 0:
            # 正方形配置
            ceil_sqrt_n = int(np.ceil(np.sqrt(float(n))))
            num_row = ceil_sqrt_n
            num_col = ceil_sqrt_n
        else:
            # 固定列数で配置
            assert n % num_col == 0
            num_row = n // num_col

        imgs = np.pad(
            imgs,
            pad_width=((0, num_row * num_col - n), (pad_pixels, pad_pixels), (pad_pixels, pad_pixels), (0, 0)),
            mode='constant',
            constant_values=pad_val
        )
        h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
        imgs = imgs.reshape(num_row, num_col, h, w, c)
        imgs = imgs.transpose(0, 2, 1, 3, 4)
        imgs = imgs.reshape(num_row * h, num_col * w, c)

        if pad_pixels > 0:
            imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
        if c == 1:
            imgs = imgs[..., 0]
        return imgs

    def approx_standard_normal_cdf(x):
        """
        標準正規分布の累積分布関数（近似）
        """
        return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x.pow(3))))


    def discretized_gaussian_log_likelihood(x, means, log_scales):
        """
        離散化されたガウス分布の対数尤度を計算
        データは [-1, 1] にスケールされた整数 [0, 255] を仮定
        """
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
        cdf_min = approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
        log_one_minus_cdf_min = torch.log(torch.clamp(1. - cdf_min, min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999, log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min,
                        torch.log(torch.clamp(cdf_delta, min=1e-12))))
        assert log_probs.shape == x.shape
        return log_probs

    def rms(variables):
        """
        RMS（Root Mean Square）を計算
        """
        return torch.sqrt(
            sum([torch.sum(v ** 2) for v in variables]) /
            sum(v.numel() for v in variables)
        )


    def get_warmed_up_lr(max_lr, warmup, global_step):
        """
        ウォームアップスケジュールによる学習率の計算
        """
        if warmup == 0:
            return max_lr
        return max_lr * min(global_step / float(warmup), 1.0)

    def make_optimizer(
        model, optimizer_name, lr, grad_clip, rmsprop_decay=0.95, rmsprop_momentum=0.9, epsilon=1e-8
    ):
        """
        PyTorch用オプティマイザを作成し、勾配をクリップ
        """
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=epsilon)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=lr, alpha=rmsprop_decay, momentum=rmsprop_momentum, eps=epsilon
            )
        else:
            raise NotImplementedError(f"Optimizer '{optimizer_name}' is not implemented")

        def train_step(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            return optimizer

        return train_step

    class ExponentialMovingAverage:
        """
        PyTorchでのExponential Moving Averageの実装
        """
        def __init__(self, model, decay):
            self.model = model
            self.decay = decay
            self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}
            self.backup = {}

        def update(self):
            """
            EMAパラメータを更新
            """
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].data = self.decay * self.shadow[name].data + (1.0 - self.decay) * param.data

        def apply_shadow(self):
            """
            EMAを適用（現在のモデルパラメータを保存）
            """
            self.backup = {name: param.clone().detach() for name, param in self.model.named_parameters()}
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name].data)

        def restore(self):
            """
            保存されたパラメータを元に戻す
            """
            for name, param in self.model.named_parameters():
                if name in self.backup:
                    param.data.copy_(self.backup[name].data)
            self.backup = {}

    def get_gcp_region():
        """
        現在のGCPリージョンを取得
        """
        import requests
        metadata_server = "http://metadata/computeMetadata/v1/instance/"
        metadata_flavor = {'Metadata-Flavor': 'Google'}
        zone = requests.get(metadata_server + 'zone', headers=metadata_flavor).text
        zone = zone.split('/')[-1]
        region = '-'.join(zone.split('-')[:-1])
        return region

    def save_tiled_imgs(filename, imgs, pad_pixels=1, pad_val=255, num_col=0):
        """
        タイル形式の画像をファイルに保存
        """
        tiled_image = tile_imgs(imgs, pad_pixels=pad_pixels, pad_val=pad_val, num_col=num_col)
        Image.fromarray(tiled_image).save(filename)

    def make_optimizer(model, optimizer_name, lr, grad_clip, rmsprop_decay=0.95, rmsprop_momentum=0.9, epsilon=1e-8):
        """
        PyTorch用の最適化設定
        """
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=epsilon)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=lr, alpha=rmsprop_decay, momentum=rmsprop_momentum, eps=epsilon
            )
        else:
            raise NotImplementedError(f"未実装のオプティマイザ: {optimizer_name}")

        def train_step(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            return optimizer

        return train_step
