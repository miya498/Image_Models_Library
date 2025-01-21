import math
import string
import torch
import torch.nn as nn
import torch.nn.functional as F

# デフォルトのデータ型
DEFAULT_DTYPE = torch.float32


def default_init(scale):
    """
    PyTorchでのスケール付き初期化
    """
    if scale == 0:
        scale = 1e-10
    return nn.init.kaiming_uniform_

def debug_print(x, name):
    """
    デバッグプリント：名前、平均、標準偏差、最小値、最大値を表示
    """
    print(f"{name}: mean={x.mean().item()}, std={x.std().item()}, min={x.min().item()}, max={x.max().item()}")
    return x


def flatten(x):
    """
    入力テンソルを2次元にフラット化
    """
    return x.view(x.size(0), -1)


def sumflat(x):
    """
    入力テンソルの最後の次元以外をすべて合計
    """
    return x.sum(dim=tuple(range(1, len(x.shape))))


def meanflat(x):
    """
    入力テンソルの最後の次元以外を平均
    """
    return x.mean(dim=tuple(range(1, len(x.shape))))

def contract_inner(x, y):
    """
    内部積を計算（PyTorch版）
    """
    assert x.shape[-1] == y.shape[0], "最後の次元と最初の次元が一致している必要があります。"
    return torch.matmul(x, y)

class NIN(nn.Module):
    """
    Network-In-Network レイヤー
    """
    def __init__(self, in_dim, num_units, init_scale=1.):
        super(NIN, self).__init__()
        self.linear = nn.Linear(in_dim, num_units)
        nn.init.uniform_(self.linear.weight, -init_scale, init_scale)
        nn.init.constant_(self.linear.bias, 0.)

    def forward(self, x):
        return self.linear(x)

class Dense(nn.Module):
    """
    全結合層（Dense Layer）
    """
    def __init__(self, in_dim, num_units, init_scale=1.0, bias=True):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_dim, num_units, bias=bias)
        nn.init.uniform_(self.linear.weight, -init_scale, init_scale)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return self.linear(x)

class Conv2D(nn.Module):
    """
    2次元畳み込み層
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding="same", init_scale=1.0, bias=True):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5) if init_scale > 0 else 0)
        if bias:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.conv(x)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Sinusoidal埋め込みを構築
    timesteps: タイムステップ（1次元テンソル）
    embedding_dim: 埋め込みの次元数
    """
    assert timesteps.dim() == 1, "タイムステップは1次元テンソルである必要があります"

    half_dim = embedding_dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(-torch.arange(half_dim, dtype=torch.float32) * emb_scale)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # 次元が奇数の場合はゼロを追加
        emb = F.pad(emb, (0, 1))
    return emb