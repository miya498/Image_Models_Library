import torch.nn as nn
from nn import Conv2D, get_timestep_embedding


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, time_embed_dim=128):
        super(UNet, self).__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = Conv2D(in_ch, 64, init_scale=1.0)
        self.down2 = Conv2D(64, 128, init_scale=1.0)
        self.bot = Conv2D(128, 256, init_scale=1.0)
        self.up2 = Conv2D(256, 128, init_scale=1.0)
        self.up1 = Conv2D(128, 64, init_scale=1.0)
        self.out = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x, t):
        time_emb = get_timestep_embedding(t, self.time_embed_dim).to(x.device)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x = self.bot(x2)
        x = self.up2(x + x2)
        x = self.up1(x + x1)
        return self.out(x)