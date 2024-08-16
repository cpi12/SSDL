import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention

class AttentionBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residue = x
        x = self.groupnorm(x)

        n, c, l = x.shape
        x = x.view((n, c, l))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, l))

        x += residue
        return x

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)

class Decoder1D(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv1d(4, 4, kernel_size=1, padding=0),
            nn.Conv1d(4, 512, kernel_size=3, padding=1),
            ResidualBlock1D(512, 512),
            AttentionBlock1D(512),
            ResidualBlock1D(512, 512),
            ResidualBlock1D(512, 512),
            ResidualBlock1D(512, 512),
            ResidualBlock1D(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            ResidualBlock1D(512, 512),
            ResidualBlock1D(512, 512),
            ResidualBlock1D(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            ResidualBlock1D(512, 256),
            ResidualBlock1D(256, 256),
            ResidualBlock1D(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            ResidualBlock1D(256, 128),
            ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv1d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x
