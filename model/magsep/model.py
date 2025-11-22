from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from rwkv.model import RWKV

from model.base_model import BaseModel
from model.magsep.rwkv import RWKV_LSTM
from modules.functional import STFTAndInverse, Residual, ReshapeBCFT, Repeat, \
    Mask, ToMagnitudeAndInverse, Concat, DebugShape, Bandsplit, SideEffect, CopyDim, SplitNTensor, WithShape
from modules.self_attention import SelfAttention
from modules.seq import Seq
from modules.unet import UNet


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, bias=False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * dim * 4 / 3)
        self.gate_proj = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        gate_value = self.gate_proj(x)
        gate, value = gate_value.chunk(2, dim=-1)
        hidden = F.silu(gate) * value
        return self.down_proj(hidden)


# @torch.compile()
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

        self.transformer = lambda dim: Seq(
            Residual(
                nn.RMSNorm(dim),
                SelfAttention(
                    embed_dim=dim,
                    num_heads=8,
                    dropout=dropout,
                    use_rope=True
                ),
            ),
            # Residual(
            #     nn.RMSNorm(dim),
            #     SwiGLU(dim),
            #     nn.Dropout(dropout),
            # ),
        )

        self.unet = lambda shape: UNet(
            input_shape=shape,
            channels=[shape.c, 64, 128, 256],
            stride=(2, 1),
            output_channels=shape.c,
            dropout_rate=dropout,
            pre_downsample_fn=lambda shape: Seq(
                # ReshapeBCFT(
                #     "(b t) f c",
                #     self.transformer(shape.c),
                # ),
                # ReshapeBCFT(
                #     "(b f) t c",
                #     self.transformer(shape.c),
                # ),
            ),
            post_downsample_fn=lambda shape: Seq(
                # ReshapeBCFT(
                #     "(b t) f c",
                #     self.transformer(shape.c),
                # ),
                # ReshapeBCFT(
                #     "(b f) t c",
                #     self.transformer(shape.c),
                # ),
            ),
            bottleneck_fn=lambda shape: Seq(
                # ReshapeBCFT(
                #     "(b t) f c",
                #     self.transformer(shape.c),
                # ),
                # ReshapeBCFT(
                #     "(b f) t c",
                #     self.transformer(shape.c),
                # ),
            ),
            post_upsample_fn=lambda shape: Seq(
            )
        )

        self.lstm = lambda dim: Seq(
            Residual(
                # nn.RMSNorm(dim),
                RWKV_LSTM(
                    input_size=dim,
                    hidden_size=128,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,
                ),
                nn.Linear(128, dim),
            ),
            # Residual(
            #     nn.RMSNorm(dim),
            #     nn.Dropout(dropout),
            #     SwiGLU(dim),
            #     nn.Dropout(dropout),
            # ),
        )

        self.thing = lambda shape: Seq(
            ReshapeBCFT(
                "(b c) t f",
                self.lstm(shape.f),
            ),
            ReshapeBCFT(
                "(b t) c f",
                self.lstm(shape.f),
            ),
            ReshapeBCFT(
                "(b f) t c",
                self.lstm(shape.c),
            ),
            ReshapeBCFT(
                "(b t) f c",
                self.lstm(shape.c),
            ),
        )

        self.model = Seq(
            STFTAndInverse(
                in_channels=2,
                n_fft=n_fft,
                hop_length=hop_length,
                fn=lambda shape: ToMagnitudeAndInverse(
                    shape=shape,
                    fn=lambda shape: Mask(
                        Seq(
                            # CopyDim(times=num_instruments, dim=1),
                            Rearrange("b (n c) f t -> b n (f c) t", n=num_instruments),
                            # WithShape(
                            #     shape=replace(shape, c=num_instruments, f=shape.f * shape.c),
                            #     fn=lambda shape: Seq(
                            #         SplitNTensor(
                            #             shape=shape,
                            #             dim=2,
                            #             fns=[
                            #                 lambda shape, index: self.thing(shape),
                            #                 lambda shape, index: self.thing(shape),
                            #                 lambda shape, index: self.thing(shape),
                            #                 lambda shape, index: self.thing(shape),
                            #             ],
                            #             split_points=[512, 768, 1024],
                            #             concat_dim=2,
                            #         ),
                            #     )
                            # ),

                            Bandsplit(
                                shape=replace(shape, c=num_instruments, f=shape.f * shape.c),
                                num_splits=splits,
                                fn=lambda shape: Seq(
                                    self.thing(shape),
                                )
                            ),
                            Rearrange("b n (c f) t -> b (n c) f t", f=shape.f, n=num_instruments),
                        ),
                        self.visualize("mask")
                    ),
                ),
            ),
            Rearrange("b (n c) t -> b n c t", n=num_instruments),
        )

    def process(self, x):
        return self.model(x)
