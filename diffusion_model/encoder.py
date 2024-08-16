import torch
from torch import nn
from torch.nn import functional as F
from .decoder import AttentionBlock1D, ResidualBlock1D

class Encoder1D(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv1d(3, 128, kernel_size=3, padding=1),
            ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 128),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock1D(128, 256),
            ResidualBlock1D(256, 256),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock1D(256, 512),
            ResidualBlock1D(512, 512),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock1D(512, 512),
            ResidualBlock1D(512, 512),
            ResidualBlock1D(512, 512),
            AttentionBlock1D(512),
            ResidualBlock1D(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv1d(512, 8, kernel_size=3, padding=1),
            nn.Conv1d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x, noise):
        for module in self:
            if getattr(module, 'stride', None) == 2:
                x = F.pad(x, (0, 1))  # Padding at downsampling should be asymmetric
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise

        x *= 0.18215
        return x
