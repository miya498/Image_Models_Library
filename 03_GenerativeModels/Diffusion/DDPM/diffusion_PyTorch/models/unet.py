import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    """
    Swish活性化関数
    """
    return x * torch.sigmoid(x)


def normalize(x, temb, name=None):
    """
    Group Normalization
    """
    return nn.GroupNorm(32, x.shape[1])(x)


class Upsample(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        else:
            return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, dropout=0.0, conv_shortcut=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if conv_shortcut or in_channels != out_channels else None

    def forward(self, x, temb):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.conv_shortcut:
            x = self.conv_shortcut(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, C, -1)
        k = self.k(h).view(B, C, -1)
        v = self.v(h).view(B, C, -1)

        w = torch.bmm(q.permute(0, 2, 1), k) * (C**-0.5)
        w = F.softmax(w, dim=-1)
        h = torch.bmm(v, w.permute(0, 2, 1)).view(B, C, H, W)
        h = self.proj_out(h)
        return x + h


class UNetModel(nn.Module):
    def __init__(self, in_channels, out_channels, ch, ch_mult, num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True):
        super().__init__()
        self.temb_dense = nn.Sequential(
            nn.Linear(ch, ch * 4),
            nn.SiLU(),
            nn.Linear(ch * 4, ch * 4),
        )

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attn_resolutions = attn_resolutions
        self.resamp_with_conv = resamp_with_conv

        prev_channels = in_channels
        for i, mult in enumerate(ch_mult):
            out_channels = ch * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResnetBlock(prev_channels, out_channels, ch * 4, dropout))
                prev_channels = out_channels
                if 2**(len(ch_mult) - i - 1) in attn_resolutions:
                    self.downs.append(AttentionBlock(out_channels))
            if i != len(ch_mult) - 1:
                self.downs.append(Downsample(out_channels, resamp_with_conv))

        for i, mult in reversed(list(enumerate(ch_mult))):
            out_channels = ch * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResnetBlock(prev_channels, out_channels, ch * 4, dropout))
                prev_channels = out_channels
                if 2**(len(ch_mult) - i - 1) in attn_resolutions:
                    self.ups.append(AttentionBlock(out_channels))
            if i != 0:
                self.ups.append(Upsample(out_channels, resamp_with_conv))

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        temb = self.temb_dense(t)
        h = self.conv_in(x)
        hs = [h]

        # Downsampling
        for layer in self.downs:
            h = layer(h, temb) if isinstance(layer, ResnetBlock) else layer(h)
            hs.append(h)

        # Middle
        h = hs.pop()
        h = ResnetBlock(h.shape[1], h.shape[1], temb.shape[1])(h, temb)
        h = AttentionBlock(h.shape[1])(h)
        h = ResnetBlock(h.shape[1], h.shape[1], temb.shape[1])(h, temb)

        # Upsampling
        for layer in self.ups:
            h = torch.cat([h, hs.pop()], dim=1) if isinstance(layer, ResnetBlock) else h
            h = layer(h, temb) if isinstance(layer, ResnetBlock) else layer(h)

        return self.conv_out(h)