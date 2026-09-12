import torch
import torch.nn as nn

from model.rotary import build_rope_cache, apply_rotary


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = build_rope_cache(
            seq_len=seq_len,
            head_dim=self.head_dim,
            device=x.device,
        )

        q, k = apply_rotary(q, k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        output = attention @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(
            batch,
            seq_len,
            self.hidden_size,
        )

        return self.out_proj(output)
