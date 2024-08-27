import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, latent_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(latent_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )

    def forward(self, latent):
        # Apply self-attention
        attention_output, _ = self.attention(latent, latent, latent)
        latent = self.layernorm(attention_output + latent)  # Residual connection
        
        # Feedforward
        latent = latent + self.feedforward(latent)  # Residual connection
        return latent

class CrossAttention(nn.Module):
    def __init__(self, latent_dim, context_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(latent_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )
        self.context_projection = nn.Linear(context_dim, latent_dim)

    def forward(self, latent, context):
        # Expand context to match the sequence length of latent
        context = self.context_projection(context)  # Project context to match latent dimension
        context = context.unsqueeze(1).expand(-1, latent.size(1), -1)  # Shape: [batch_size, 90, latent_dim]
        
        # Apply cross-attention
        attention_output, _ = self.attention(latent, context, context)
        latent = self.layernorm(attention_output + latent)  # Residual connection
        
        # Feedforward
        latent = latent + self.feedforward(latent)  # Residual connection
        return latent
class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.query_conv = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, length = x.size()
        query = self.query_conv(x).view(batch_size, -1, length).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, length)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, length)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, length)
        out = self.gamma * out + x
        return out
