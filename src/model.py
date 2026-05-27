import math

import torch
from torch import nn


def sin_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """Sinusoidal embedding for the scalar time variable t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.view(-1, 1) * freqs.view(1, -1) * 2 * math.pi
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityNet(nn.Module):
    """v_theta(x, t): a small MLP that outputs a 2D velocity vector."""

    def __init__(self, hidden: int = 256, t_emb_dim: int = 32):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        in_dim = 2 + t_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sin_embedding(t, self.t_emb_dim)
        return self.net(torch.cat([x, t_emb], dim=-1))
