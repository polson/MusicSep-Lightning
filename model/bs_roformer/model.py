from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from model.base_model import BaseModel
from modules.functional import STFTAndInverse, Residual, ReshapeBCFT, Repeat, \
    Mask, SplitNTensor, RepeatWithArgs, ToMagnitudeAndInverse, ComplexMask
from modules.self_attention import SelfAttention
from modules.seq import Seq


class TimeConditionedLayer(nn.Module):
    """Injects FiLM conditioning from t kwarg passed through Seq."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )

    def forward(self, x, t=None, **kwargs):
        if t is None:
            return x
        # x: (b, num_splits, embed_dim, t_frames)
        params = self.net(t.unsqueeze(-1))  # (b, dim*2)
        scale, shift = params.chunk(2, dim=-1)
        scale = rearrange(scale, 'b f -> b 1 f 1') + 1
        shift = rearrange(shift, 'b f -> b 1 f 1')
        return x * scale + shift


class BSRoformer(BaseModel):

    def __init__(self,
                 num_instruments=1,
                 n_fft=2048,
                 hop_length=512,
                 layers=1,
                 mask_layers=1,
                 dropout=0.1,
                 embed_dim=64,
                 freqs_per_bands=[2] * 24 + [4] * 12 + [12] * 8 + [24] * 8 + [48] * 8 + [128] * 2
                 ):
        super().__init__()

        self.num_instruments = num_instruments

        # Time conditioning layer (no parent reference needed)
        self.time_cond = TimeConditionedLayer(embed_dim)

        # Convert the original bs roformer freq band configuration
        scale_factor = (n_fft * 2) / sum(freqs_per_bands)
        freqs_per_bands_scaled = [int(x * scale_factor) for x in freqs_per_bands]
        freqs_per_bands_cumsum = np.cumsum(freqs_per_bands_scaled).tolist()
        num_splits = len(freqs_per_bands_scaled)

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
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout),
            ),
        )

        self.embed = lambda shape, index: Seq(
            Rearrange("b c f t -> b c t f"),
            nn.RMSNorm(shape.f),
            nn.Linear(shape.f, embed_dim),
            Rearrange("b c t f -> b c f t"),
        )

        self.masking = lambda shape, split_index: Seq(
            Rearrange("b c f t -> b c t f"),
            RepeatWithArgs(
                num_repeats=mask_layers,
                block=lambda repeat_index: Seq(
                    nn.Linear(embed_dim, embed_dim * 4),
                    Seq(
                        nn.GELU(),
                        nn.Linear(embed_dim * 4, embed_dim),
                    ) if repeat_index < mask_layers - 1 else Seq(
                        nn.Tanh(),
                        nn.Linear(embed_dim * 4, freqs_per_bands_scaled[split_index] * 2),
                        nn.GLU(dim=-1),
                    ),
                ),
            ),
            Rearrange("b c t f -> b c f t"),
        )

        self.model = Seq(
            STFTAndInverse(
                in_channels=2,
                n_fft=n_fft,
                hop_length=hop_length,
                fn=lambda shape:
                ComplexMask(
                    Rearrange("b c f t -> b 1 (f c) t"),

                    # Split frequency dim and apply embedding
                    SplitNTensor(
                        replace(shape, c=1, f=shape.f * shape.c),
                        fns=[self.embed] * num_splits,
                        split_points=freqs_per_bands_cumsum[:-1],
                        dim=2,
                        concat_dim=1
                    ),

                    # Inject time conditioning here (t comes from kwargs)
                    self.time_cond,

                    # Transformer layers
                    Repeat(
                        layers,
                        ReshapeBCFT(
                            "(b c) t f",
                            self.transformer(dim=embed_dim)
                        ),
                        ReshapeBCFT(
                            "(b t) c f",
                            self.transformer(dim=embed_dim)
                        ),
                    ),

                    ReshapeBCFT("b c t f", nn.RMSNorm(embed_dim)),

                    # Masking layers
                    SplitNTensor(
                        replace(shape, c=num_splits, f=embed_dim),
                        fns=[self.masking] * num_splits,
                        split_points=list(range(1, num_splits)),
                        dim=1,
                        concat_dim=2
                    ),
                    Rearrange("b 1 (f c) t -> b c f t", c=shape.c),
                    self.visualize("Mask")
                )
            )
        )

    def process(self, x, t=None):
        # x: (b, n, c, samples) for blended input, (b, c, samples) for mixture
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
