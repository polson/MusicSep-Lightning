import math
from dataclasses import replace

import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from model.base_model import BaseModel
from model.magsep.rwkv import BiRWKVLayer
from modules.functional import ReshapeBCFT, WithShape, CFTShape, Residual, Repeat, Bandsplit
from modules.self_attention import SelfAttention
from modules.seq import Seq
from modules.unet import UNet


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLN(nn.Module):
    """
    Adaptive Layer Norm with optional gating, following DiT design.
    If return_gate=True: returns (modulated_x, gate) for use with GatedResidual
    If return_gate=False: returns just modulated_x for standalone use
    """

    def __init__(self, dim, time_emb_dim=256, return_gate=False):
        super().__init__()

        self.return_gate = return_gate

        out_dim = dim * 3 if return_gate else dim * 2
        self.time_mlp = Seq(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_dim)
        )

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

        # Small-value initialization so block starts near-identity
        nn.init.constant_(self.time_mlp[-1].weight, 0.0)
        nn.init.constant_(self.time_mlp[-1].bias, 0.0)  # bias can stay at 0

    def __repr__(self):
        return f"AdaLN(dim={self.norm.normalized_shape[0]}, return_gate={self.return_gate})"

    def forward(self, x, t, **kwargs):
        x_norm = self.norm(x)

        # Ensure t is (batch_size,)
        if t.ndim > 1:
            t = t.view(t.shape[0])

        # 24, 1, 512
        t_emb = self.time_mlp(t)

        # First match batch sizes for t and x
        if x.shape[0] != t_emb.shape[0]:
            splits = x.shape[0] // t_emb.shape[0]
            t_emb = repeat(t_emb, 'b ... -> (b s) ...', s=splits)

        # Now match dimensions for t so it can broadcast
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
    """
    Residual block that expects the first module to return (x, gate).
    Applies: x + gate * sublayer(x)
    """

    def __init__(self, adaln, *sublayers):
        super().__init__()
        self.adaln = adaln
        self.sublayers = Seq(*sublayers)

    def forward(self, x, **kwargs):
        # AdaLN returns (modulated_x, gate)
        modulated, gate = self.adaln(x, **kwargs)

        # Pass through attention/FFN
        out = self.sublayers(modulated)

        # Gated residual connection
        return x + gate * out


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

        self.loss_factory = LossFactory.create(LossType.MSE)

        self.dit = lambda dim, reshape: Seq(
            ReshapeBCFT(
                reshape,
                GatedResidual(
                    AdaLN(dim, return_gate=True),
                    SelfAttention(
                        embed_dim=dim,
                        num_heads=8,
                        dropout=dropout,
                        use_rope=True
                    ),
                ),
                GatedResidual(
                    AdaLN(dim, return_gate=True),
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout),
                ),
            ),
        )

        self.rwkv = lambda dim, reshape: Seq(
            ReshapeBCFT(
                reshape,
                GatedResidual(
                    AdaLN(dim, return_gate=True),
                    BiRWKVLayer(
                        dim,
                    ),
                ),
                GatedResidual(
                    AdaLN(dim, return_gate=True),
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout),
                ),
            ),
        )

        self.rwkv_no_ada = lambda dim, reshape: Seq(
            ReshapeBCFT(
                reshape,
                Residual(
                    nn.LayerNorm(dim),
                    BiRWKVLayer(
                        dim,
                    ),
                ),
                Residual(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout),
                ),
            ),
        )

        self.unet = lambda shape: UNet(
            input_shape=shape,
            channels=[shape.c, 128, 256],
            stride=(8, 1),
            output_channels=shape.c,
            post_downsample_fn=lambda shape: Seq(
                # Rearrange("b c f t -> b c t f"),
                # AdaLN(shape.f),
                # Rearrange("b c t f -> b c f t"),
            ),
            bottleneck_fn=lambda shape: Seq(
                # Repeat(
                #     1,
                #     self.rwkv(shape.f, "(b c) t f"),
                # ),
            ),
            post_upsample_fn=lambda shape: Seq(
                # Rearrange("b c f t -> b c t f"),
                # AdaLN(shape.f),
                # Rearrange("b c t f -> b c f t", c=shape.c),
            ),
            post_skip_fn=lambda shape: Seq(
                Rearrange("b c f t -> b c t f"),
                AdaLN(shape.f),
                Rearrange("b c t f -> b c f t", c=shape.c),
            )
        )

        embed_dim = 128

        self.model = Seq(
            WithShape(
                shape=CFTShape(c=8, f=n_fft // 2, t=87),
                fn=lambda shape: Seq(

                    # Rearrange("b c f t -> b c t f"),
                    # AdaLN(shape.f),
                    # Rearrange("b c t f -> b c f t", c=shape.c),

                    # self.unet(shape),

                    Rearrange("b c f t -> b 1 (f c) t"),
                    Bandsplit(
                        shape=replace(shape, c=1, f=shape.f * (shape.c)),
                        num_splits=64,
                        fn=lambda shape: Seq(
                            Rearrange("b c f t -> b c t f"),
                            nn.Linear(shape.f, embed_dim),
                            AdaLN(embed_dim),
                            Rearrange("b c t f -> b c f t"),

                            Repeat(
                                1,
                                self.rwkv(embed_dim, "(b c) t f"),
                                # self.rwkv(embed_dim, "(b t) c f"),
                            ),

                            Rearrange("b c f t -> b c t f"),
                            AdaLN(embed_dim),
                            nn.Linear(embed_dim, shape.f),
                            nn.GLU(dim=-1),
                            Rearrange("b c t f -> b c f t", c=shape.c),
                        )
                    ),
                    Rearrange("b 1 (f c) t -> b c f t", c=4),

                ),
            ),
        )

    def process(self, x, mixture, t=None):
        x = torch.cat([x, mixture], dim=1)
        # 'mixture' is still passed here, but AdaLN will ignore it via **kwargs
        x = self.model(x, t=t, mixture=mixture)
        # x = x[:, :2, :, :]
        return x
