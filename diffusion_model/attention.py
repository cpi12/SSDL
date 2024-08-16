import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_head: int, n_embd: int, in_proj_bias=True):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.q_proj = nn.Linear(n_embd, n_embd, bias=in_proj_bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=in_proj_bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=in_proj_bias)
        self.out_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.n_head, self.n_embd // self.n_head)
        k = k.view(batch_size, seq_len, self.n_head, self.n_embd // self.n_head)
        v = v.view(batch_size, seq_len, self.n_head, self.n_embd // self.n_head)

        attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k) / (self.n_embd // self.n_head) ** 0.5
        if mask is not None:
            attn_weights.masked_fill_(mask == 0, -1e9)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.n_embd)
        output = self.out_proj(attn_output)
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context: int, in_proj_bias=True):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.d_context = d_context

        self.q_proj = nn.Linear(n_embd, n_embd, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_context, n_embd, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_context, n_embd, bias=in_proj_bias)
        self.out_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, context):
        batch_size, seq_len, _ = x.size()
        context_len = context.size(1)

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.view(batch_size, seq_len, self.n_head, self.n_embd // self.n_head)
        k = k.view(batch_size, context_len, self.n_head, self.n_embd // self.n_head)
        v = v.view(batch_size, context_len, self.n_head, self.n_embd // self.n_head)

        attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k) / (self.n_embd // self.n_head) ** 0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.n_embd)
        output = self.out_proj(attn_output)
        return output
    
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
