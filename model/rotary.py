import torch


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def build_rope_cache(seq_len, head_dim, device, base=10000.0):
    theta = 1.0 / (
        base ** (
            torch.arange(0, head_dim, 2, device=device).float()
            / head_dim
        )
    )

    positions = torch.arange(seq_len, device=device).float()

    freqs = torch.outer(positions, theta)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)

    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
