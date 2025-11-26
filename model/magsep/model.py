import math
from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from model.base_model import BaseModel
from model.magsep.rwkv import BiRWKV
from modules.functional import STFTAndInverse, ReshapeBCFT, Repeat, \
    Mask, ToMagnitudeAndInverse, DebugShape, Bandsplit, SplitNTensor, FFT2dAndInverse, Concat, FFT2d, ComplexMask, \
    Residual
from modules.seq import Seq


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for continuous time."""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.cos(), args.sin()], dim=-1)


class TimeConditionedLayer(nn.Module):
    """FiLM conditioning for flow matching with sinusoidal time embeddings."""

    def __init__(self, dim, time_embed_dim=1024):
        super().__init__()
        self.dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, x, t=None, **kwargs):
        if t is None:
            return x

        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif t.dim() == 2:
            t = t.squeeze(-1)

        if t.shape[0] != x.shape[0]:
            repeat_factor = x.shape[0] // t.shape[0]
            t = t.repeat_interleave(repeat_factor)

        params = self.time_mlp(t.float())
        scale, shift = params.chunk(2, dim=-1)

        num_middle_dims = x.dim() - 2
        view_shape = (x.shape[0],) + (1,) * num_middle_dims + (self.dim,)

        scale = scale.view(*view_shape)
        shift = shift.view(*view_shape)

        return x * (1 + scale) + shift


class MixtureConditionedLayer(nn.Module):
    """FiLM conditioning for flow matching with sinusoidal time embeddings."""

    def __init__(self, dim, time_embed_dim=512):
        super().__init__()
        self.dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, x, t=None, **kwargs):
        if t is None:
            return x

        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif t.dim() == 2:
            t = t.squeeze(-1)

        if t.shape[0] != x.shape[0]:
            repeat_factor = x.shape[0] // t.shape[0]
            t = t.repeat_interleave(repeat_factor)

        params = self.time_mlp(t.float())
        scale, shift = params.chunk(2, dim=-1)

        num_middle_dims = x.dim() - 2
        view_shape = (x.shape[0],) + (1,) * num_middle_dims + (self.dim,)

        scale = scale.view(*view_shape)
        shift = shift.view(*view_shape)

        return x * (1 + scale) + shift


class CombinedConditionedLayer(nn.Module):
    """Combined time + mixture FiLM conditioning."""

    def __init__(self, dim, mixture_dim=512, time_embed_dim=1024):
        super().__init__()
        self.dim = dim
        self.time_cond = TimeConditionedLayer(dim, time_embed_dim)
        self.mixture_cond = MixtureConditionedLayer(dim, mixture_dim)

    def forward(self, x, t, mixture, **kwargs):
        x = self.time_cond(x, t=t)
        x = self.mixture_cond(x, mixture=mixture)
        return x


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
        mixture_embed_dim = 512  # Dimension of mixture conditioning

        self.loss_factory = LossFactory.create(LossType.MSE)

        # STFT for mixture (shared params conceptually, but separate call)
        self.stft = nn.Identity()  # We'll compute STFT in process()

        self.model = Seq(
            STFTAndInverse(
                in_channels=2,  # Now only noisy input, not concatenated
                n_fft=n_fft,
                hop_length=hop_length,
                fn=lambda shape: Seq(
                    Seq(
                        Rearrange("b c f t -> b 1 (f c) t"),

                        Rearrange("b c f t -> b c t f"),
                        nn.LayerNorm(shape.f * shape.c),
                        Rearrange("b c t f -> b c f t"),

                        # EMBED
                        SplitNTensor(
                            shape=replace(shape, c=1, f=shape.f * shape.c),
                            split_points=[512],
                            fns=[
                                lambda shape, index: Seq(
                                    Rearrange("b c f t -> b c t f"),
                                    nn.Linear(shape.f, embed_dim_1),
                                    nn.RMSNorm(embed_dim_1),
                                    Rearrange("b c t f -> b c f t"),
                                ),
                                lambda shape, index: Seq(
                                    Rearrange("b c f t -> b c t f"),
                                    nn.Linear(shape.f, embed_dim_2),
                                    nn.RMSNorm(embed_dim_2),
                                    Rearrange("b c t f -> b c f t"),
                                ),
                            ],
                            dim=2,
                            concat_dim=2,
                        ),

                        # Combined time + mixture conditioning after embedding
                        ReshapeBCFT(
                            "b c t f",
                            CombinedConditionedLayer(embed_dim),
                        ),

                        SplitNTensor(
                            shape=replace(shape, c=1, f=embed_dim),
                            split_points=[embed_dim_1],
                            fns=[
                                lambda shape, index: Repeat(
                                    layers,
                                    ReshapeBCFT(
                                        "(b c) t f",
                                        CombinedConditionedLayer(shape.f),
                                        BiRWKV(shape.f, num_layers=1),
                                    ),
                                    FFT2dAndInverse(
                                        shape=shape,
                                        fn=lambda shape: Seq(
                                            ReshapeBCFT(
                                                "(b c) t f",
                                                CombinedConditionedLayer(shape.f),
                                                BiRWKV(shape.f, num_layers=1),
                                            ),
                                        )
                                    ),
                                ),
                                lambda shape, index: Seq(
                                    ReshapeBCFT(
                                        "(b c) t f",
                                        CombinedConditionedLayer(shape.f),
                                        BiRWKV(shape.f, num_layers=1),
                                    ),
                                ),
                            ],
                            dim=2,
                            concat_dim=2,
                        ),

                        Rearrange("b c f t -> b c t f"),
                        CombinedConditionedLayer(embed_dim),
                        nn.RMSNorm(embed_dim),
                        nn.Linear(embed_dim, embed_dim * 4),
                        nn.SiLU(),
                        nn.Linear(embed_dim * 4, shape.f * shape.c),
                        Rearrange("b c t f -> b c f t"),

                        Rearrange("b 1 (f c) t -> b c f t", c=shape.c),

                        self.visualize("mask"),
                    ),
                ),
            ),
        )

    def process(self, x, mixture, t=None):
        b = x.shape[0]
        n = self.num_instruments

        x = self.model(x, mixture=mixture, t=t)

        x = rearrange(x, 'b (n c) t -> b n c t', b=b, n=n)
        return x
