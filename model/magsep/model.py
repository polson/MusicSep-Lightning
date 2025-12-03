import math
from dataclasses import replace

import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from model.base_model import BaseModel
from model.magsep.rwkv import BiRWKVLayer
from modules.functional import ReshapeBCFT, WithShape, CFTShape, Residual, Repeat, Bandsplit, ComplexMask
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

        # Shared time embedding - computed once per forward pass
        self.time_embedding = TimeEmbedding(dim=n_fft // 2)

        self.dit = lambda dim, reshape: Seq(
            ReshapeBCFT(
                reshape,
                GatedResidual(
                    AdaLN(dim, time_emb_dim=time_emb_dim, return_gate=True),
                    SelfAttention(
                        embed_dim=dim,
                        num_heads=8,
                        dropout=dropout,
                        use_rope=True
                    ),
                ),
                GatedResidual(
                    AdaLN(dim, time_emb_dim=time_emb_dim, return_gate=True),
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
                    AdaLN(dim, time_emb_dim=time_emb_dim, return_gate=True),
                    BiRWKVLayer(dim),
                ),
                GatedResidual(
                    AdaLN(dim, time_emb_dim=time_emb_dim, return_gate=True),
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
                    BiRWKVLayer(dim),
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
            output_channels=4,
            post_downsample_fn=lambda shape: Seq(
                # Repeat(
                #     1,
                #     self.rwkv_no_ada(shape.c, "(b t) f c"),
                # ),
            ),
            bottleneck_fn=lambda shape: Seq(
                Repeat(
                    1,  # slightly deeper bottleneck
                    # self.rwkv_no_ada(shape.c, "(b t) f c"),
                ),
            ),
            post_upsample_fn=lambda shape: Seq(
                # Repeat(
                #     1,
                #     self.rwkv_no_ada(shape.c, "(b t) f c"),
                # ),
            ),
            # This is the critical one for stability:
            post_skip_fn=lambda shape: Seq(
                # Rearrange("b c f t -> b c t f"),
                # nn.LayerNorm(shape.f),
                # Rearrange("b c t f -> b c f t", c=shape.c),
            )
        )

        embed_dim = 256
        self.model = ComplexMask(
            WithShape(
                shape=CFTShape(c=9, f=n_fft // 2, t=87),
                fn=lambda shape: Seq(
                    self.unet(shape=shape),
                    nn.Tanh(),

                    # Rearrange("b c f t -> b 1 (f c) t"),
                    # Bandsplit(
                    #     shape=replace(shape, c=1, f=shape.f * (shape.c)),
                    #     num_splits=64,
                    #     fn=lambda shape: Seq(
                    #         Rearrange("b c f t -> b c t f"),
                    #         nn.Linear(shape.f, embed_dim),
                    #         nn.LayerNorm(embed_dim),
                    #         # AdaLN(embed_dim),
                    #         Rearrange("b c t f -> b c f t"),
                    #
                    #         Repeat(
                    #             1,
                    #             self.rwkv_no_ada(embed_dim, "b (c t) f"),
                    #             # self.rwkv(embed_dim, "(b t) c f"),
                    #         ),
                    #
                    #         Rearrange("b c f t -> b c t f"),
                    #         # AdaLN(embed_dim),
                    #         nn.LayerNorm(embed_dim),
                    #         nn.Linear(embed_dim, shape.f * 2),
                    #         nn.GLU(dim=-1),
                    #         Rearrange("b c t f -> b c f t", c=shape.c),
                    #
                    #     )
                    # ),
                    # Rearrange("b 1 (f c) t -> b c f t", c=9),
                ),
            ),
        )

        self.model2 = Seq(
            WithShape(
                shape=CFTShape(c=9, f=n_fft // 2, t=87),
                fn=lambda shape: Seq(
                    self.unet(shape=shape),

                    # Rearrange("b c f t -> b 1 (f c) t"),
                    # Bandsplit(
                    #     shape=replace(shape, c=1, f=shape.f * (shape.c)),
                    #     num_splits=64,
                    #     fn=lambda shape: Seq(
                    #         Rearrange("b c f t -> b c t f"),
                    #         nn.Linear(shape.f, embed_dim),
                    #         nn.LayerNorm(embed_dim),
                    #         # AdaLN(embed_dim),
                    #         Rearrange("b c t f -> b c f t"),
                    #
                    #         Repeat(
                    #             1,
                    #             self.rwkv_no_ada(embed_dim, "b (c t) f"),
                    #             # self.rwkv(embed_dim, "(b t) c f"),
                    #         ),
                    #
                    #         Rearrange("b c f t -> b c t f"),
                    #         # AdaLN(embed_dim),
                    #         nn.LayerNorm(embed_dim),
                    #         nn.Linear(embed_dim, shape.f * 2),
                    #         nn.GLU(dim=-1),
                    #         Rearrange("b c t f -> b c f t", c=shape.c),
                    #
                    #     )
                    # ),
                    # Rearrange("b 1 (f c) t -> b c f t", c=9),
                ),
            ),
        )

        self.model_waveform = Seq(
            # input b,c,t
            WithShape(
                shape=CFTShape(c=4, f=0, t=44100),
                fn=lambda shape: Seq(
                    self.rwkv_no_ada(shape.t, "b c t")
                )
            )
        )

        self.complex_mask = ComplexMask()

    def process(self, x, mixture, t=None):
        original = x

        t_emb = self.time_embedding(t)
        t_emb_expanded = t_emb[:, None, :, None].expand(-1, -1, -1, x.shape[3])

        # Concatenate: [B, 4+4+1, c, t] = [B, 9, c, t]
        x = torch.cat([x, mixture, t_emb_expanded], dim=1)
        x = self.model(x, t_emb=t_emb, mixture=mixture)

        # x2 = torch.cat([x, mixture, t_emb_expanded], dim=1)
        # x2 = self.model2(x2, t_emb=t_emb, mixture=mixture)

        # x = x + x2

        x = x - original

        x = x[:, :4, :, :] + 1e-8
        return x
