from dataclasses import replace

import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from model.base_model import BaseModel
from modules.functional import STFTAndInverse, Residual, ReshapeBCFT, Bandsplit, Repeat, \
    Mask, ToMagnitudeAndInverse
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
            Residual(
                nn.RMSNorm(dim),
                SwiGLU(dim),
                nn.Dropout(dropout),
            ),
        )

        self.unet = lambda shape: UNet(
            input_shape=shape,
            channels=[shape.c, 128, 256],
            stride=(8, 1),
            output_channels=shape.c,
            dropout_rate=dropout,
            post_downsample_fn=lambda shape: Seq(
                ReshapeBCFT(
                    "(b t) f c",
                    self.transformer(shape.c),
                ),
                ReshapeBCFT(
                    "(b f) t c",
                    self.transformer(shape.c),
                ),
            ),
            bottleneck_fn=lambda shape: Repeat(
                layers,
                ReshapeBCFT(
                    "(b t) f c",
                    self.transformer(shape.c),
                ),
                ReshapeBCFT(
                    "(b f) t c",
                    self.transformer(shape.c),
                ),
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
                        Rearrange("b c f t -> b 1 (f c) t"),
                        Bandsplit(
                            shape=replace(shape, c=1, f=shape.f * shape.c),
                            num_splits=splits,
                            fn=lambda shape: Seq(
                                self.unet(shape),
                            )
                        ),
                        Rearrange("b 1 (f c) t -> b c f t", c=shape.c),
                    ),
                ),
            ),
            Rearrange("b (n c) t -> b n c t", n=num_instruments)
        )

    def process(self, x):
        return self.model(x)
