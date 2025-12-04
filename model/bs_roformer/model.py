from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from model.base_model import BaseModel, SeparationMode
from modules.functional import STFTAndInverse, Residual, ReshapeBCFT, Repeat, \
    SplitNTensor, RepeatWithArgs, ComplexMask
from modules.self_attention import SelfAttention
from modules.seq import Seq


class BSRoformer(BaseModel):

    def __init__(self,
                 n_fft=2048,
                 hop_length=512,
                 layers=1,
                 mask_layers=1,
                 dropout=0.1,
                 embed_dim=64,
                 freqs_per_bands=[2] * 24 + [4] * 12 + [12] * 8 + [24] * 8 + [48] * 8 + [128] * 2
                 ):
        super().__init__()

        self.num_instruments = 1

        # Convert the original bs roformer freq band configuration
        scale_factor = (n_fft * 2) / sum(freqs_per_bands)
        freqs_per_bands_scaled = [int(x * scale_factor) for x in freqs_per_bands]
        freqs_per_bands_cumsum = np.cumsum(freqs_per_bands_scaled).tolist()
        num_splits = len(freqs_per_bands_scaled)

        self.loss_factory = LossFactory.create(LossType.MULTI_STFT)

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

    def get_mode(self):
        return SeparationMode.ONE_SHOT

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform

    def decode(self, encoded: torch.Tensor, original_length: int = None) -> torch.Tensor:
        return encoded

    # TODO: fail before processing if we are trying to separate more than one instrument
    def process(self, x, mixture, t):
        b = x.shape[0]
        x = self.model(x)
        return x
