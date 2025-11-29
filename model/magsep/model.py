import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from model.base_model import BaseModel
from model.magsep.rwkv import BiRWKV
from modules.functional import STFTAndInverse, ReshapeBCFT, Module
from modules.seq import Seq
from modules.stft import STFT


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=x.device) / half
        )
        args = x[:, None] * freqs[None]
        return torch.cat([args.cos(), args.sin()], dim=-1)


class TimeEmbedding(nn.Module):
    """FiLM conditioning for flow matching with sinusoidal time embeddings."""

    def __init__(self, dim, time_embed_dim=1024):
        super().__init__()
        self.dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, x, t, **kwargs):
        t_flat = t.view(t.shape[0])
        modulation = self.time_mlp(t_flat)

        # x is (b, c, f, t) but c=1 and f=embed_dim
        # So reshape to broadcast over f dimension
        scale, shift = rearrange(modulation, 'b (two f) -> two b 1 f 1', two=2)

        return (1 + scale) * x + shift


class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim, time_embed_dim=1024):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Combined projection: mixture features + time -> gamma, beta
        self.to_gamma_beta = nn.Linear(dim * 2, dim * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

        self.mixture_proj = Seq(
            STFT(n_fft=2048, hop_length=512),
            Rearrange("b c f t -> b 1 (f c) t"),
            Rearrange("b c f t -> b c t f"),
            Module(lambda x: x[:, :, :, :dim])  # or a learned projection
        )

    def forward(self, x, mixture, t):
        mix_cond = self.mixture_proj(mixture)  # (b, 1, t, dim)
        time_cond = self.time_mlp(t.view(-1))  # (b, dim)
        time_cond = time_cond[:, None, None, :]  # (b, 1, 1, dim) broadcast over t

        # Concatenate and project
        cond = torch.cat([mix_cond, time_cond.expand_as(mix_cond)], dim=-1)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + gamma) + beta


class MagSplitModel(BaseModel):
    def __init__(self,
                 num_instruments=1,
                 n_fft=4096,
                 hop_length=1024,
                 layers=1,
                 splits=32,
                 ):
        super().__init__()
        dropout = 0.1

        self.num_instruments = num_instruments
        self.n_fft = n_fft
        self.hop_length = hop_length

        embed_dim_1 = 768
        embed_dim_2 = 256
        embed_dim = embed_dim_1 + embed_dim_2

        self.loss_factory = LossFactory.create(LossType.MSE)

        self.stft = nn.Identity()

        # Input channels: 2 (noisy) + 2 (mixture) = 4
        in_channels = 2

        self.conditioning = lambda embed_dim, mixture_dim: Seq(
            Rearrange("b c f t -> b c t f"),
            AdaLN(embed_dim, cond_dim=mixture_dim),
            Rearrange("b c t f -> b c f t"),
        )

        self.model = Seq(
            STFTAndInverse(
                in_channels=in_channels,  # Now handles concatenated input
                n_fft=n_fft,
                hop_length=hop_length,
                fn=lambda shape: Seq(
                    Seq(
                        Rearrange("b c f t -> b 1 (f c) t"),
                        Rearrange("b c f t -> b c t f"),
                        nn.Linear(shape.f * shape.c, embed_dim),
                        nn.RMSNorm(embed_dim),
                        Rearrange("b c t f -> b c f t"),

                        self.conditioning(embed_dim, shape.f * shape.c),
                        ReshapeBCFT(
                            "(b c) t f",
                            BiRWKV(embed_dim, num_layers=1),
                        ),

                        self.conditioning(embed_dim, shape.f * shape.c),
                        Rearrange("b c f t -> b c t f"),
                        nn.RMSNorm(embed_dim),
                        nn.Linear(embed_dim, embed_dim * 4),
                        nn.SiLU(),
                        nn.Linear(embed_dim * 4, (shape.f * shape.c)),
                        Rearrange("b c t f -> b c f t"),

                        Rearrange("b 1 (f c) t -> b c f t", c=shape.c),
                    ),
                ),
            ),
        )

    def process(self, x, mixture, t=None):
        b = x.shape[0]
        n = self.num_instruments
        x = self.model(x, t=t, mixture=mixture)
        print(f"Output: mean={x.mean():.6f}, std={x.std():.6f}")
        x = rearrange(x, 'b (n c) t -> b n c t', b=b, n=n)
        return x
