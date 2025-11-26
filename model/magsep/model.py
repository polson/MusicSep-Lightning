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
        # t: (batch,) values in [0, 1]
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

        # Initialize final layer for identity transform at t=0
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, x, t=None, **kwargs):
        if t is None:
            return x

        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif t.dim() == 2:
            t = t.squeeze(-1)

        # Handle batch size mismatch from reshapes like "(b c) t f"
        # x.shape[0] is the effective batch size after reshape
        # t.shape[0] is the original batch size
        if t.shape[0] != x.shape[0]:
            # Repeat t to match x's batch dimension
            repeat_factor = x.shape[0] // t.shape[0]
            t = t.repeat_interleave(repeat_factor)

        params = self.time_mlp(t.float())  # (b_eff, dim*2)
        scale, shift = params.chunk(2, dim=-1)  # each (b_eff, dim)

        # Build the broadcast shape: (b, 1, 1, ..., dim)
        # Feature dim is always last
        num_middle_dims = x.dim() - 2
        view_shape = (x.shape[0],) + (1,) * num_middle_dims + (self.dim,)

        scale = scale.view(*view_shape)
        shift = shift.view(*view_shape)

        return x * (1 + scale) + shift


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
        embed_dim_1 = 768
        embed_dim_2 = 256
        embed_dim = embed_dim_1 + embed_dim_2

        self.loss_factory = LossFactory.create(LossType.MSE)

        self.model = Seq(
            STFTAndInverse(
                in_channels=2,
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

                        # Inject time conditioning after embedding
                        ReshapeBCFT(
                            "b c t f",
                            TimeConditionedLayer(embed_dim),
                        ),

                        SplitNTensor(
                            shape=replace(shape, c=1, f=embed_dim),
                            split_points=[embed_dim_1],
                            fns=[
                                lambda shape, index: Repeat(
                                    layers,
                                    ReshapeBCFT(
                                        "(b c) t f",
                                        TimeConditionedLayer(shape.f),
                                        BiRWKV(shape.f, num_layers=1),
                                    ),

                                    # ReshapeBCFT(
                                    #     "(b c) f t",
                                    #     BiRWKV(690, num_layers=1),
                                    # ),
                                    FFT2dAndInverse(
                                        shape=shape,
                                        fn=lambda shape: Seq(
                                            ReshapeBCFT(
                                                "(b c) t f",
                                                TimeConditionedLayer(shape.f),
                                                BiRWKV(shape.f, num_layers=1),
                                            ),

                                            # ReshapeBCFT(
                                            #     "(b c) f t",
                                            #     BiRWKV(346, num_layers=1),
                                            # ),
                                        )
                                    ),
                                ),
                                lambda shape, index: Seq(
                                    ReshapeBCFT(
                                        "(b c) t f",
                                        TimeConditionedLayer(shape.f),
                                        BiRWKV(shape.f, num_layers=1),
                                    ),
                                ),
                            ],
                            dim=2,
                            concat_dim=2,
                        ),

                        Rearrange("b c f t -> b c t f"),
                        TimeConditionedLayer(embed_dim),
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
            # Rearrange("b (n c) t -> b n c t", n=num_instruments)
        )

    def process(self, x, t=None):
        # x: (b, n, c, samples) for blended input, (b, c, samples) for mixture
        # print(
        #     f"t first value: {t[0] if t is not None else 'None'}, is debug: {self.is_debug} is training: {self.training}"
        # )
        if x.dim() == 4:
            b, n, c, samples = x.shape
            x = rearrange(x, 'b n c t -> (b n) c t')
            t = t.repeat_interleave(n) if t is not None else None
        else:
            b = x.shape[0]
            n = self.num_instruments

        # Pass t as kwarg - Seq will propagate it through
        x = self.model(x, t=t)

        x = rearrange(x, '(b n) c t -> b n c t', b=b, n=n)
        return x
