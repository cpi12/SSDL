import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn

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

class CrossAttention(nn.Module):
    def __init__(self, latent_dim, context_dim, sequence_length, num_heads=4):
        super().__init__()
        # Adjust the latent projection to reduce the dimension from 4096 to 2048
        self.latent_projection = nn.Linear(latent_dim, latent_dim // 2)  # Reduce from 4096 to 2048
        self.context_projection = nn.Linear(context_dim, (latent_dim // 2) * sequence_length)
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim // 2, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(latent_dim // 2)
        self.feedforward = nn.Sequential(
            nn.Linear(latent_dim // 2, latent_dim * 2),  # Increase temporarily for feedforward
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim // 2)  # Reduce back to 2048
        )

    def forward(self, latent, context):
        # Apply projection to reduce the latent dimension
        latent = self.latent_projection(latent)

        # Project context to match the latent dimension and sequence length
        context = self.context_projection(context)
        context = context.view(latent.size(0), latent.size(1), -1)

        # Apply cross-attention
        attention_output, _ = self.attention(latent, context, context)
        latent = self.layernorm(attention_output + latent)  # Residual connection

        # Apply feedforward
        latent = latent + self.feedforward(latent)  # Residual connection
        return latent

class GraphConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, edge_index, time_emb_dim):
        super(GraphConvolutionBlock, self).__init__()
        self.edge_index = edge_index
        self.time_projection = nn.Linear(time_emb_dim, out_channels)
        self.gcn = geom_nn.GATConv(in_channels, out_channels, heads=3, concat=False)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, time_emb):
        b, c, l = x.shape

        # Project time embedding and add it to the feature space
        time_emb = self.time_projection(time_emb).unsqueeze(-1)

        # Flatten the input for the GCN operation
        x = x.reshape(-1, c)
        x = self.gcn(x, self.edge_index)

        # Batch normalization and activation
        x = self.batchnorm(x)
        x = self.relu(x)

        # Reshape back to the original batch dimension
        x = x.reshape(b, -1, l)
        return x

class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Shape: (batch_size, length, channels)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, channels, length)
        return x

class Downsample1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.transposed_conv(x)

class UNet1D(nn.Module):
    def __init__(self, edge_index, latent_dim=90, time_emb_dim=64, context_dim=512, hidden_dim=128, dropout_prob=0.1):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.input_conv = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1, padding=0)

        # Encoder 1
        self.lstm1 = LSTMBlock(hidden_dim, hidden_dim)
        self.gcn1 = GraphConvolutionBlock(hidden_dim * 2, hidden_dim * 2, edge_index, time_emb_dim)
        self.downsample1 = Downsample1D(hidden_dim * 2, hidden_dim * 4)
        self.dropout1 = nn.Dropout(dropout_prob)

        # Encoder 2
        self.lstm2 = LSTMBlock(hidden_dim * 4, hidden_dim * 4)
        self.gcn2 = GraphConvolutionBlock(hidden_dim * 8, hidden_dim * 4, edge_index, time_emb_dim)
        self.downsample2 = Downsample1D(hidden_dim * 4, hidden_dim * 8)
        self.dropout2 = nn.Dropout(dropout_prob)

        # Encoder 3
        self.lstm3 = LSTMBlock(hidden_dim * 8, hidden_dim * 8)
        self.gcn3 = GraphConvolutionBlock(hidden_dim * 16, hidden_dim * 16, edge_index, time_emb_dim)

        # Cross-Attention
        self.cross_attn = CrossAttention(latent_dim=hidden_dim * 16, context_dim=context_dim, sequence_length=12)

        # Decoder 1
        self.gcn4 = GraphConvolutionBlock(hidden_dim * 8, hidden_dim * 4, edge_index, time_emb_dim)
        self.lstm4 = LSTMBlock(hidden_dim * 4, hidden_dim * 4)
        self.upsample1 = Upsample1D(hidden_dim * 8, hidden_dim * 4)
        self.dropout3 = nn.Dropout(dropout_prob)

        # Decoder 2
        self.gcn5 = GraphConvolutionBlock(hidden_dim * 4, hidden_dim * 2, edge_index, time_emb_dim)
        self.lstm5 = LSTMBlock(hidden_dim * 2, hidden_dim * 2)
        self.upsample2 = Upsample1D(hidden_dim * 4, hidden_dim * 2)
        self.dropout4 = nn.Dropout(dropout_prob)

        # Decoder 3
        self.gcn6 = GraphConvolutionBlock(hidden_dim * 2, hidden_dim, edge_index, time_emb_dim)
        self.lstm6 = LSTMBlock(hidden_dim, latent_dim)
        self.final_conv = nn.Conv1d(180, latent_dim, kernel_size=1, padding=0)

    def forward(self, x, context, time):
        time_emb = self.time_mlp(time)
        x = self.input_conv(x)

        # Encoder Path
        x = self.lstm1(x)
        x = self.gcn1(x, time_emb)
        x = self.downsample1(x)
        x = self.dropout1(x)

        x = self.lstm2(x)
        x = self.gcn2(x, time_emb)
        x = self.downsample2(x)
        x = self.dropout2(x)

        x = self.lstm3(x)
        x = self.gcn3(x, time_emb)

        # Bottleneck (Cross-Attention)
        x = x.permute(0, 2, 1)
        x = self.cross_attn(x, context)
        x = x.permute(0, 2, 1)

        # Decoder Path
        x = self.gcn4(x, time_emb)
        x = self.lstm4(x)
        x = self.upsample1(x)
        x = self.dropout3(x)

        x = self.gcn5(x, time_emb)
        x = self.lstm5(x)
        x = self.upsample2(x)
        x = self.dropout4(x)

        x = self.gcn6(x, time_emb)
        x = self.lstm6(x)
        x = self.final_conv(x)

        return x

class Diffusion1D(nn.Module):
    def __init__(self, edge_index):
        super().__init__()
        self.unet = UNet1D(edge_index=edge_index)
        self.final = nn.Conv1d(90, 90, kernel_size=1, padding=0)
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, latent, context, time):
        latent = self.unet(latent, context, time)
        return self.final(latent)

# import torch
# from torch import nn
# import torch.nn.functional as F
# import torch_geometric.nn as geom_nn


# class SinusoidalPositionEmbedding(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         device = x.device
#         half_dim = self.dim // 2
#         emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb
    
# class Upsample1D(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.transposed_conv = nn.ConvTranspose1d(channels, channels, kernel_size=2, stride=2)

#     def forward(self, x):
#         return self.transposed_conv(x)

# class Downsample1D(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Conv1d(channels, channels, kernel_size=1, padding=0, stride=2)  # Stride=2 reduces the length by half

#     def forward(self, x):
#         return self.conv(x)

# class SwitchSequential(nn.Sequential):
#     def forward(self, x, context, time_emb):
#         for layer in self:
#             if isinstance(layer, ResidualBlock1D):
#                 x = layer(x, time_emb)
#             elif isinstance(layer, TemporalAttentionBlock):
#                 x = layer(x)
#             else:
#                 x = layer(x)
#         return x

# class GraphConvolutionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, edge_index):
#         super(GraphConvolutionBlock, self).__init__()
#         self.edge_index = edge_index
#         self.gcn = geom_nn.GATConv(in_channels, out_channels, heads=3, concat=False)
#         self.batchnorm = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU()
#         self.downsample = nn.Conv1d(out_channels, in_channels, kernel_size=1, padding=0)

#     def forward(self, x):
#         b, c, l = x.shape

#         # Flatten the input for the GCN operation
#         x = x.view(-1, c)
#         x = self.gcn(x, self.edge_index)

#         # Batch normalization and activation
#         x = self.batchnorm(x)
#         x = self.relu(x)

#         # Reshape back to the original batch dimension
#         x = x.view(b, -1, l)

#         # Downsample to match the desired output shape
#         x = self.downsample(x)

#         return x

# class TemporalAttentionBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=6, batch_first=True)
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(0.1)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.GELU(),
#             nn.Linear(dim * 4, dim)
#         )

#     def forward(self, x):
#         residual = x
#         x = x.permute(0, 2, 1)
#         x, _ = self.attention(x, x, x)
#         x = self.norm1(x + residual.permute(0, 2, 1))  # Residual connection
#         x = self.ffn(x)
#         x = self.norm2(x + residual.permute(0, 2, 1))  # Another residual connection
#         x = self.dropout(x)
#         x = x.permute(0, 2, 1)
#         return x

# class ResidualBlock1D(nn.Module):
#     def __init__(self, in_channels, out_channels, time_emb_dim, dropout_prob=0.5):
#         super().__init__()
#         self.groupnorm_feature = nn.GroupNorm(20, in_channels)
#         self.conv_feature = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
#         self.linear_time = nn.Linear(time_emb_dim, out_channels)
#         self.groupnorm_merged = nn.GroupNorm(20, out_channels)
#         self.conv_merged = nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0)
#         self.dropout = nn.Dropout(dropout_prob)

#         if in_channels == out_channels:
#             self.residual_layer = nn.Identity()
#         else:
#             self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

#     def forward(self, feature, time_emb):
#         residue = feature
#         feature = self.groupnorm_feature(feature)
#         feature = F.silu(feature)
#         feature = self.conv_feature(feature)
#         time_emb = F.silu(time_emb)
#         time_emb = self.linear_time(time_emb)
#         merged = feature + time_emb.unsqueeze(-1)
#         merged = self.groupnorm_merged(merged)
#         merged = F.silu(merged)
#         merged = self.conv_merged(merged)
#         merged = self.dropout(merged)
#         return merged + self.residual_layer(residue)

# class SimpleSelfAttention(nn.Module):
#     def __init__(self, latent_dim, num_heads=4):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
#         self.layernorm = nn.LayerNorm(latent_dim)
#         self.feedforward = nn.Sequential(
#             nn.Linear(latent_dim, latent_dim * 4),
#             nn.GELU(),
#             nn.Linear(latent_dim * 4, latent_dim)
#         )

#     def forward(self, latent):
#         # Apply self-attention
#         attention_output, _ = self.attention(latent, latent, latent)
#         latent = self.layernorm(attention_output + latent)  # Residual connection
        
#         # Feedforward
#         latent = latent + self.feedforward(latent)  # Residual connection
#         return latent

# class SimpleCrossAttention(nn.Module):
#     def __init__(self, latent_dim, context_dim, num_heads=4):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
#         self.layernorm = nn.LayerNorm(latent_dim)
#         self.feedforward = nn.Sequential(
#             nn.Linear(latent_dim, latent_dim * 4),
#             nn.GELU(),
#             nn.Linear(latent_dim * 4, latent_dim)
#         )
#         self.context_projection = nn.Linear(context_dim, latent_dim)

#     def forward(self, latent, context):
#         # Expand context to match the sequence length of latent
#         context = self.context_projection(context)
#         context = context.unsqueeze(1).expand(-1, latent.size(1), -1)
        
#         # Apply cross-attention
#         attention_output, _ = self.attention(latent, context, context)
#         latent = self.layernorm(attention_output + latent)  # Residual connection
        
#         # Feedforward
#         latent = latent + self.feedforward(latent)  # Residual connection
#         return latent

# class UNet1D(nn.Module):
#     def __init__(self, dim, edge_index, dropout_prob=0.1):
#         super().__init__()
#         time_emb_dim = 64

#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbedding(time_emb_dim),
#             nn.Linear(time_emb_dim, time_emb_dim),
#             nn.GELU(),
#             nn.Linear(time_emb_dim, time_emb_dim)
#         )

#         self.input_conv = nn.Conv1d(90, dim, kernel_size=1, padding=0)
#         self.input_temporal_attention = TemporalAttentionBlock(dim)

#         self.encoders = nn.ModuleList([
#             SwitchSequential(
#                 ResidualBlock1D(dim, dim, time_emb_dim, dropout_prob),
#                 SimpleSelfAttention(latent_dim=dim),
#                 GraphConvolutionBlock(dim, dim, edge_index),
#                 TemporalAttentionBlock(dim)
#             ),
#             SwitchSequential(
#                 Downsample1D(dim),  # <--- Downsampling happens here
#                 ResidualBlock1D(dim, dim * 2, time_emb_dim, dropout_prob),
#                 SimpleCrossAttention(latent_dim=dim * 2, context_dim=512),
#                 GraphConvolutionBlock(dim * 2, dim * 2, edge_index),
#             ),
#             SwitchSequential(
#                 Downsample1D(dim * 2),  # <--- Another downsampling happens here
#                 ResidualBlock1D(dim * 2, dim * 2, time_emb_dim, dropout_prob),
#                 SimpleSelfAttention(latent_dim=dim * 2),
#                 GraphConvolutionBlock(dim * 2, dim * 2, edge_index),
#             ),
#         ])
        
#         self.bottleneck = SwitchSequential(
#             ResidualBlock1D(dim * 2, dim * 2, time_emb_dim, dropout_prob),
#             SimpleCrossAttention(latent_dim=dim * 2, context_dim=512),
#             GraphConvolutionBlock(dim * 2, dim * 2, edge_index),
#             TemporalAttentionBlock(dim * 2)
#         )
        
#         self.decoders = nn.ModuleList([
#             SwitchSequential(
#                 ResidualBlock1D(dim * 2, dim * 2, time_emb_dim, dropout_prob),
#                 SimpleSelfAttention(latent_dim=dim * 2),
#                 GraphConvolutionBlock(dim * 2, dim * 2, edge_index),
#                 Upsample1D(dim * 2)
#             ),
#             SwitchSequential(
#                 ResidualBlock1D(dim * 2, dim * 2, time_emb_dim, dropout_prob),
#                 SimpleCrossAttention(latent_dim=dim * 2, context_dim=512),
#                 GraphConvolutionBlock(dim * 2, dim * 2, edge_index),
#                 Upsample1D(dim * 2)
#             ),
#             SwitchSequential(
#                 ResidualBlock1D(dim * 2, dim, time_emb_dim, dropout_prob),
#                 SimpleSelfAttention(latent_dim=dim),
#                 GraphConvolutionBlock(dim, dim, edge_index)
#             ),
#         ])

#     def forward(self, x, context, time):
#         time_emb = self.time_mlp(time)
        
#         x = self.input_conv(x)
#         x = self.input_temporal_attention(x)
        
#         for i, layers in enumerate(self.encoders):
#             x = layers(x, context, time_emb)

#         x = self.bottleneck(x, context, time_emb)

#         for i, layers in enumerate(self.decoders):
#             x = layers(x, context, time_emb)

#         return x

# class FinalLayer1D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.groupnorm = nn.GroupNorm(20, in_channels)
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

#     def forward(self, x):
#         x = self.groupnorm(x)
#         x = F.silu(x)
#         x = self.conv(x)
#         return x

# class Diffusion1D(nn.Module):
#     def __init__(self, edge_index):
#         super().__init__()
#         self.unet = UNet1D(dim=120, edge_index=edge_index)
#         self.final = FinalLayer1D(120, 90)
#         self.decoder = nn.Identity()
#         self.apply(self.initialize_weights)

#     def initialize_weights(self, m):
#         if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
#             nn.init.ones_(m.weight)
#             nn.init.zeros_(m.bias)

#     def forward(self, latent, context, time):
#         # Check for NaN values in latent and context
#         if torch.isnan(latent).any():
#             raise ValueError("NaN detected in latent tensor during forward pass.")
#         if torch.isnan(context).any():
#             raise ValueError("NaN detected in context tensor during forward pass.")

#         # Proceed with the forward pass
#         latent = self.unet(latent, context, time)
        
#         if torch.isnan(latent).any():
#             raise ValueError("NaN detected in UNet1D output during forward pass.")

#         output = self.final(latent)
        
#         if torch.isnan(output).any():
#             raise ValueError("NaN detected after final layer during forward pass.")

#         output = self.decoder(output)
        
#         if torch.isnan(output).any():
#             raise ValueError("NaN detected after decoder during forward pass.")
        
#         return output


# # import torch
# # from torch import nn
# # import torch.nn.functional as F
# # import torch_geometric.nn as geom_nn
# # from .attention import SelfAttention, CrossAttention


# # class SinusoidalPositionEmbedding(nn.Module):
# #     def __init__(self, dim):
# #         super().__init__()
# #         self.dim = dim

# #     def forward(self, x):
# #         device = x.device
# #         half_dim = self.dim // 2
# #         emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
# #         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
# #         emb = x[:, None] * emb[None, :]
# #         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
# #         return emb
    
# # class Upsample1D(nn.Module):
# #     def __init__(self, channels):
# #         super().__init__()
# #         self.transposed_conv = nn.ConvTranspose1d(channels, channels, kernel_size=2, stride=2)

# #     def forward(self, x):
# #         return self.transposed_conv(x)

# # class Downsample1D(nn.Module):
# #     def __init__(self, channels):
# #         super().__init__()
# #         self.conv = nn.Conv1d(channels, channels, kernel_size=1, padding=0, stride=2)  # Stride=2 reduces the length by half

# #     def forward(self, x):
# #         return self.conv(x)

# # class SwitchSequential(nn.Sequential):
# #     def forward(self, x, context, time_emb):
# #         for layer in self:
# #             if isinstance(layer, AttentionBlock1D):
# #                 x = layer(x, context)
# #             elif isinstance(layer, ResidualBlock1D):
# #                 x = layer(x, time_emb)
# #             elif isinstance(layer, TemporalAttentionBlock):
# #                 x = layer(x)
# #             else:
# #                 x = layer(x)
# #         return x

# # class GraphConvolutionBlock(nn.Module):
# #     def __init__(self, in_channels, out_channels, edge_index):
# #         super(GraphConvolutionBlock, self).__init__()
# #         self.edge_index = edge_index
# #         self.gcn = geom_nn.GATConv(in_channels, out_channels, heads=3, concat=False)  # Using GAT with 4 heads
# #         self.batchnorm = nn.BatchNorm1d(out_channels)
# #         self.relu = nn.ReLU()
# #         self.downsample = nn.Conv1d(out_channels, in_channels, kernel_size=1, padding=0)

# #     def forward(self, x):
# #         b, c, l = x.shape
# #         # print(f"Before GCN in GraphConvolutionBlock: {x.shape}")  # Debugging shape

# #         # Flatten the input for the GCN operation
# #         x = x.view(-1, c)
# #         x = self.gcn(x, self.edge_index)

# #         # Batch normalization and activation
# #         x = self.batchnorm(x)
# #         x = self.relu(x)

# #         # Reshape back to the original batch dimension
# #         x = x.view(b, -1, l)
# #         # print(f"After GCN in GraphConvolutionBlock: {x.shape}")  # Debugging shape

# #         # Downsample to match the desired output shape
# #         x = self.downsample(x)
# #         # print(f"After Downsampling in GraphConvolutionBlock: {x.shape}")  # Debugging shape

# #         return x

# # class TemporalAttentionBlock(nn.Module):
# #     def __init__(self, dim):
# #         super().__init__()
# #         self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=6, batch_first=True)
# #         self.norm1 = nn.LayerNorm(dim)
# #         self.norm2 = nn.LayerNorm(dim)
# #         self.dropout = nn.Dropout(0.1)
# #         self.ffn = nn.Sequential(
# #             nn.Linear(dim, dim * 4),
# #             nn.GELU(),
# #             nn.Linear(dim * 4, dim)
# #         )

# #     def forward(self, x):
# #         # print(f"Input to TemporalAttentionBlock: {x.shape}")  # Debugging shape
# #         residual = x
# #         x = x.permute(0, 2, 1)  # Shape: (batch_size, seq_length, channels)
# #         x, _ = self.attention(x, x, x)
# #         x = self.norm1(x + residual.permute(0, 2, 1))  # Residual connection
# #         x = self.ffn(x)
# #         x = self.norm2(x + residual.permute(0, 2, 1))  # Another residual connection
# #         x = self.dropout(x)
# #         x = x.permute(0, 2, 1)
# #         # print(f"Output from TemporalAttentionBlock: {x.shape}")  # Debugging shape
# #         return x

# # class ResidualBlock1D(nn.Module):
# #     def __init__(self, in_channels, out_channels, time_emb_dim, dropout_prob=0.5):
# #         super().__init__()
# #         self.groupnorm_feature = nn.GroupNorm(18, in_channels)
# #         self.conv_feature = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
# #         self.linear_time = nn.Linear(time_emb_dim, out_channels)
# #         self.groupnorm_merged = nn.GroupNorm(18, out_channels)
# #         self.conv_merged = nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0)
# #         self.dropout = nn.Dropout(dropout_prob)

# #         if in_channels == out_channels:
# #             self.residual_layer = nn.Identity()
# #         else:
# #             self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

# #     def forward(self, feature, time_emb):
# #         # print(f"Input to ResidualBlock1D: {feature.shape}")  # Debugging shape
# #         residue = feature
# #         feature = self.groupnorm_feature(feature)
# #         feature = F.silu(feature)
# #         feature = self.conv_feature(feature)
# #         time_emb = F.silu(time_emb)
# #         time_emb = self.linear_time(time_emb)
# #         merged = feature + time_emb.unsqueeze(-1)
# #         merged = self.groupnorm_merged(merged)
# #         merged = F.silu(merged)
# #         merged = self.conv_merged(merged)
# #         merged = self.dropout(merged)
# #         # print(f"Output from ResidualBlock1D: {merged.shape}")  # Debugging shape
# #         return merged + self.residual_layer(residue)

# # class AttentionBlock1D(nn.Module):
# #     def __init__(self, n_head: int, n_embd: int, dim: int, d_context=512, dropout_prob=0.5):
# #         super().__init__()
# #         channels = n_head * n_embd
# #         # print(n_head, n_embd)
# #         self.groupnorm = nn.GroupNorm(18, channels, eps=1e-6)
# #         self.conv_input = nn.Conv1d(channels, channels, kernel_size=1, padding=0)
# #         self.layernorm_1 = nn.LayerNorm(channels)
# #         self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
# #         self.layernorm_2 = nn.LayerNorm(channels)
# #         self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
# #         self.layernorm_3 = nn.LayerNorm(channels)
# #         self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
# #         self.linear_geglu_2 = nn.Linear(4 * channels, channels)
# #         self.conv_output = nn.Conv1d(channels, channels, kernel_size=1, padding=0)
# #         self.dropout = nn.Dropout(dropout_prob)

# #     def forward(self, x, context):
# #         residue_long = x
# #         x = self.groupnorm(x)
# #         x = self.conv_input(x)
# #         n, c, l = x.shape
# #         x = x.view((n, c, l))
# #         x = x.transpose(-1, -2)
# #         residue_short = x
# #         x = self.layernorm_1(x)
# #         x = self.attention_1(x)
# #         x += residue_short
# #         residue_short = x
# #         x = self.layernorm_2(x)
# #         context = context.unsqueeze(1)
# #         context = context.expand(-1, x.size(1), -1)
# #         x = self.attention_2(x, context)
# #         x = x.permute(1, 0, 2)
# #         residue_short = x
# #         x = self.layernorm_3(x)
# #         x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
# #         x = x * F.gelu(gate)
# #         x = self.linear_geglu_2(x)
# #         x = x.transpose(-1, -2)
# #         x = x.reshape((n, c, l))
# #         x = self.dropout(x)
# #         return self.conv_output(x) + residue_long

# # class UNet1D(nn.Module):
# #     def __init__(self, dim, edge_index, dropout_prob=0.1):
# #         super().__init__()
# #         time_emb_dim = 64

# #         self.time_mlp = nn.Sequential(
# #             SinusoidalPositionEmbedding(time_emb_dim),
# #             nn.Linear(time_emb_dim, time_emb_dim),
# #             nn.GELU(),
# #             nn.Linear(time_emb_dim, time_emb_dim)
# #         )

# #         self.input_conv = nn.Conv1d(90, dim, kernel_size=1, padding=0)
# #         self.input_temporal_attention = TemporalAttentionBlock(dim)

# #         self.encoders = nn.ModuleList([
# #             SwitchSequential(
# #                 ResidualBlock1D(dim, dim, time_emb_dim, dropout_prob),
# #                 AttentionBlock1D(3, int(dim / 3), dim=dim, dropout_prob=dropout_prob),
# #                 GraphConvolutionBlock(dim, dim, edge_index),
# #                 TemporalAttentionBlock(dim)
# #             ),
# #             SwitchSequential(
# #                 Downsample1D(dim),  # <--- Downsampling happens here
# #                 ResidualBlock1D(dim, dim * 2, time_emb_dim, dropout_prob),
# #                 AttentionBlock1D(3, int((dim * 2) / 3), dim=dim * 2, dropout_prob=dropout_prob),
# #                 GraphConvolutionBlock(dim * 2, dim * 2, edge_index),
# #             ),
# #             SwitchSequential(
# #                 Downsample1D(dim * 2),  # <--- Another downsampling happens here
# #                 ResidualBlock1D(dim * 2, dim * 2, time_emb_dim, dropout_prob),
# #                 AttentionBlock1D(3, int((dim * 2) / 3), dim=dim * 2, dropout_prob=dropout_prob),
# #                 GraphConvolutionBlock(dim * 2, dim * 2, edge_index),
# #             ),
# #         ])
        
# #         self.bottleneck = SwitchSequential(
# #             ResidualBlock1D(dim * 2, dim * 2, time_emb_dim, dropout_prob),
# #             AttentionBlock1D(3, int((dim * 2) / 3), dim=dim * 2, dropout_prob=dropout_prob),
# #             GraphConvolutionBlock(dim * 2, dim * 2, edge_index),
# #             TemporalAttentionBlock(dim * 2)
# #         )
        
# #         self.decoders = nn.ModuleList([
# #             SwitchSequential(
# #                 ResidualBlock1D(dim * 2, dim * 2, time_emb_dim, dropout_prob),
# #                 AttentionBlock1D(3, int((dim * 2) / 3), dim=dim * 2, dropout_prob=dropout_prob),
# #                 GraphConvolutionBlock(dim * 2, dim * 2, edge_index),
# #                 Upsample1D(dim * 2)
# #             ),
# #             SwitchSequential(
# #                 ResidualBlock1D(dim * 2, dim * 2, time_emb_dim, dropout_prob),
# #                 AttentionBlock1D(3, int(dim * 2 / 3), dim=dim, dropout_prob=dropout_prob),
# #                 GraphConvolutionBlock(dim * 2, dim * 2, edge_index),
# #                 Upsample1D(dim * 2)
# #             ),
# #             SwitchSequential(
# #                 ResidualBlock1D(dim * 2, dim, time_emb_dim, dropout_prob),
# #                 AttentionBlock1D(3, int(dim / 3), dim=dim, dropout_prob=dropout_prob),
# #                 GraphConvolutionBlock(dim, dim, edge_index)
# #             ),
# #         ])

# #     def forward(self, x, context, time):
# #         time_emb = self.time_mlp(time)
        
# #         x = self.input_conv(x)
# #         # print(f"After input_conv in UNet1D: {x.shape}")  # Debugging shape
# #         x = self.input_temporal_attention(x)
# #         # print(f"After input_temporal_attention in UNet1D: {x.shape}")  # Debugging shape
        
# #         for i, layers in enumerate(self.encoders):
# #             # print(f"Before encoder block {i} in UNet1D: {x.shape}")  # Debugging shape
# #             x = layers(x, context, time_emb)
# #             # print(f"After encoder block {i} in UNet1D: {x.shape}")  # Debugging shape

# #         # print(f"Before bottleneck in UNet1D: {x.shape}")  # Debugging shape
# #         x = self.bottleneck(x, context, time_emb)
# #         # print(f"After bottleneck in UNet1D: {x.shape}")  # Debugging shape

# #         for i, layers in enumerate(self.decoders):
# #             # print(f"Before decoder block {i} in UNet1D: {x.shape}")  # Debugging shape
# #             x = layers(x, context, time_emb)
# #             # print(f"After decoder block {i} in UNet1D: {x.shape}")  # Debugging shape

# #         return x

# # class FinalLayer1D(nn.Module):
# #     def __init__(self, in_channels, out_channels):
# #         super().__init__()
# #         self.groupnorm = nn.GroupNorm(30, in_channels)
# #         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

# #     def forward(self, x):
# #         x = self.groupnorm(x)
# #         x = F.silu(x)
# #         x = self.conv(x)
# #         return x

# # class Diffusion1D(nn.Module):
# #     def __init__(self, edge_index):
# #         super().__init__()
# #         self.unet = UNet1D(dim=90, edge_index=edge_index)
# #         self.final = FinalLayer1D(90, 90)
# #         self.decoder = nn.Identity()
# #         self.apply(self.initialize_weights)

# #     def initialize_weights(self, m):
# #         if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
# #             nn.init.xavier_uniform_(m.weight)
# #             if m.bias is not None:
# #                 nn.init.zeros_(m.bias)
# #         elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
# #             nn.init.ones_(m.weight)
# #             nn.init.zeros_(m.bias)

# #     def forward(self, latent, context, time):
# #         # Check for NaN values in latent and context
# #         if torch.isnan(latent).any():
# #             raise ValueError("NaN detected in latent tensor during forward pass.")
# #         if torch.isnan(context).any():
# #             raise ValueError("NaN detected in context tensor during forward pass.")

# #         # Proceed with the forward pass
# #         latent = self.unet(latent, context, time)
        
# #         if torch.isnan(latent).any():
# #             raise ValueError("NaN detected in UNet1D output during forward pass.")

# #         output = self.final(latent)
        
# #         if torch.isnan(output).any():
# #             raise ValueError("NaN detected after final layer during forward pass.")

# #         output = self.decoder(output)
        
# #         if torch.isnan(output).any():
# #             raise ValueError("NaN detected after decoder during forward pass.")
        
# #         return output
