import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.eps)

        return self.weight * x


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        dropout=0.0,
    ):
        super().__init__()

        from model.attention import CausalSelfAttention

        self.norm1 = RMSNorm(hidden_size)
        self.attention = CausalSelfAttention(
            hidden_size,
            num_heads,
            dropout,
        )

        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(
            hidden_size,
            intermediate_size,
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
