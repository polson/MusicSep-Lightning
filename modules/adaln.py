import math

import torch
from einops import repeat
from torch import nn

from modules.seq import Seq


class AdaLN(nn.Module):
    """
    Adaptive Layer Norm with optional gating, following DiT design.
    Expects pre-computed time embedding passed via t_emb kwarg.
    """

    def __init__(self, dim, time_emb_dim=256, return_gate=False):
        super().__init__()
        self.return_gate = return_gate

        out_dim = dim * 3 if return_gate else dim * 2
        self.proj = nn.Linear(time_emb_dim, out_dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

        nn.init.constant_(self.proj.weight, 0.0)
        nn.init.constant_(self.proj.bias, 0.0)

    def __repr__(self):
        return f"AdaLN(dim={self.norm.normalized_shape[0]}, return_gate={self.return_gate})"

    def forward(self, x, t_emb=None, **kwargs):
        x_norm = self.norm(x)
        t_emb = self.proj(t_emb)

        if x.shape[0] != t_emb.shape[0]:
            splits = x.shape[0] // t_emb.shape[0]
            t_emb = repeat(t_emb, 'b ... -> (b s) ...', s=splits)

        if x.ndim == 4:
            t_emb = t_emb[:, None, None, :]
        elif x.ndim == 3:
            t_emb = t_emb[:, None, :]

        if self.return_gate:
            scale, shift, gate = t_emb.chunk(3, dim=-1)
            return x_norm * (1 + scale) + shift, gate
        else:
            scale, shift = t_emb.chunk(2, dim=-1)
            return x_norm * (1 + scale) + shift


class GatedResidual(nn.Module):
    def __init__(self, adaln, *sublayers):
        super().__init__()
        self.adaln = adaln
        self.sublayers = Seq(*sublayers)

    def forward(self, x, **kwargs):
        modulated, gate = self.adaln(x, **kwargs)
        out = self.sublayers(modulated)
        return x + gate * out


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t, **kwargs):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """
    Time embedding module: sinusoidal positional encoding + MLP projection.
    Compute once, pass to all AdaLN layers.
    """

    def __init__(self, dim=256):
        super().__init__()
        self.sinusoidal = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        if t.ndim > 1:
            t = t.view(t.shape[0])
        return self.mlp(self.sinusoidal(t))
