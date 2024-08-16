import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention, CrossAttention

def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, l = x.shape
        x = x.permute(0, 2, 1)  # (b, l, c)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, l, self.heads, -1).transpose(1, 2), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, l, -1)
        out = self.to_out(out)
        return out.permute(0, 2, 1)  # (b, c, l)

class TemporalAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = TemporalAttention(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, l = x.shape
        x = x + self.attention(x)
        return x
    
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout_prob=0.5):
        super().__init__()
        self.batchnorm_feature = nn.BatchNorm1d(in_channels)  # Replaced GroupNorm with BatchNorm
        self.conv_feature = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.linear_time = nn.Linear(time_emb_dim, out_channels)
        self.batchnorm_merged = nn.BatchNorm1d(out_channels)  # Replaced GroupNorm with BatchNorm
        self.conv_merged = nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(dropout_prob)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time_emb):
        residue = feature
        feature = self.batchnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        time_emb = F.silu(time_emb)
        time_emb = self.linear_time(time_emb)
        merged = feature + time_emb.unsqueeze(-1)
        merged = self.batchnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        merged = self.dropout(merged)
        return merged + self.residual_layer(residue)

class AttentionBlock1D(nn.Module):
    def __init__(self, n_head: int, n_embd: int, dim: int, d_context=128, dropout_prob=0.1):
        super().__init__()
        channels = n_head * n_embd
        self.linear_transform = nn.Linear(128, dim)
        self.batchnorm = nn.BatchNorm1d(channels)  # Replaced GroupNorm with BatchNorm
        self.conv_input = nn.Conv1d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv1d(channels, channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, context):
        residue_long = x
        x = self.batchnorm(x)
        x = self.conv_input(x)
        n, c, l = x.shape
        x = x.view((n, c, l))
        x = x.transpose(-1, -2)
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short
        residue_short = x
        x = self.layernorm_2(x)
        context = context.unsqueeze(1)
        context = context.expand(-1, x.size(1), -1)
        x = x.permute(1, 0, 2)
        context = context.permute(1, 0, 2)
        x = self.attention_2(x, context)
        x = x.permute(1, 0, 2)
        x += residue_short
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        x = x.transpose(-1, -2)
        x = x.view((n, c, l))
        x = self.dropout(x)
        return self.conv_output(x) + residue_long

class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time_emb):
        for layer in self:
            if isinstance(layer, AttentionBlock1D):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock1D):
                x = layer(x, time_emb)
            elif isinstance(layer, TemporalAttentionBlock):
                x = layer(x)
            else:
                x = layer(x)
        return x

class UNet1D(nn.Module):
    def __init__(self, dim, dropout_prob=0.1):
        super().__init__()
        time_emb_dim = 64

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.input_conv = nn.Conv1d(90, dim, kernel_size=1, padding=0)
        self.input_temporal_attention = TemporalAttentionBlock(dim)

        # Multi-Scale Feature Learning layers with varying kernel sizes
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.Conv1d(dim, dim, kernel_size=7, padding=3)
        ])

        self.encoders = nn.ModuleList([
            SwitchSequential(
                ResidualBlock1D(dim, dim, time_emb_dim, dropout_prob),
                AttentionBlock1D(4, int(dim/4), dim=dim, dropout_prob=dropout_prob),
                TemporalAttentionBlock(dim)
            ),
            SwitchSequential(
                Downsample1D(dim),
                ResidualBlock1D(dim, dim*2, time_emb_dim, dropout_prob),
                AttentionBlock1D(4, int((dim*2)/4), dim=dim*2, dropout_prob=dropout_prob),
                # Multi-scale feature extraction in encoder
                nn.Conv1d(dim * 2, dim * 2, kernel_size=3, padding=1),
                nn.Conv1d(dim * 2, dim * 2, kernel_size=5, padding=2)
            ),
            SwitchSequential(
                Downsample1D(dim*2),
                ResidualBlock1D(dim*2, dim*2, time_emb_dim, dropout_prob),
                AttentionBlock1D(4, int((dim*2)/4), dim=dim*2, dropout_prob=dropout_prob),
                # Multi-scale feature extraction in encoder
                nn.Conv1d(dim * 2, dim * 2, kernel_size=3, padding=1),
                nn.Conv1d(dim * 2, dim * 2, kernel_size=7, padding=3)
            ),
        ])

        self.bottleneck = SwitchSequential(
            ResidualBlock1D(dim*2, dim*2, time_emb_dim, dropout_prob),
            AttentionBlock1D(4, int((dim*2)/4), dim=dim*2, dropout_prob=dropout_prob),
            ResidualBlock1D(dim*2, dim*2, time_emb_dim, dropout_prob),
            TemporalAttentionBlock(dim*2)
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(
                ResidualBlock1D(dim*4, dim*2, time_emb_dim, dropout_prob),
                AttentionBlock1D(4, int((dim*2)/4), dim=dim*2, dropout_prob=dropout_prob),
                Upsample1D(dim*2)
            ),
            SwitchSequential(
                ResidualBlock1D(dim*4, dim, time_emb_dim, dropout_prob),
                AttentionBlock1D(4, int(dim/4), dim=dim, dropout_prob=dropout_prob),
                Upsample1D(dim)
            ),
            SwitchSequential(
                ResidualBlock1D(dim*2, dim, time_emb_dim, dropout_prob),
                AttentionBlock1D(4, int(dim/4), dim=dim, dropout_prob=dropout_prob)
            ),
        ])

    def forward(self, x, context, time):
        time_emb = self.time_mlp(time)

        x = self.input_conv(x)
        x = self.input_temporal_attention(x)

        # Multi-scale feature extraction
        multi_scale_features = [conv(x) for conv in self.multi_scale_convs]
        x = sum(multi_scale_features) / len(multi_scale_features)  # Aggregate multi-scale features

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time_emb)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time_emb)

        for layers in self.decoders:
            skip_connection = skip_connections.pop()
            if x.shape[2] != skip_connection.shape[2]:
                diff = skip_connection.shape[2] - x.shape[2]
                x = F.pad(x, (0, diff))
            x = torch.cat((x, skip_connection), dim=1)
            x = layers(x, context, time_emb)
        
        return x

class FinalLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(in_channels)  # Added BatchNorm here
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.batchnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

class Diffusion1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet1D(dim=320)
        self.final = FinalLayer1D(320, 90)
        self.decoder = nn.Identity()
        self.apply(initialize_weights)

    def forward(self, latent, context, time):
        latent = self.unet(latent, context, time)
        output = self.final(latent)
        output = self.decoder(output)
        return output