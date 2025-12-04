import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from model.base_model import BaseModel, SeparationMode
from model.magsep.rwkv import BiRWKVLayer
from modules.functional import ReshapeBCFT, WithShape, CFTShape, Residual, Repeat, ComplexMask, ToSTFT, InverseSTFT, \
    Mask, DebugShape
from modules.seq import Seq
from modules.stft import STFT
from modules.unet import UNet


class MagSplitModel(BaseModel):
    def __init__(self,
                 config,
                 ):
        super().__init__()
        dropout = 0.1

        self.num_instruments = len(config.training.target_sources)
        self.n_fft = config.model.n_fft
        self.hop_length = config.model.hop_length
        self.loss_factory = LossFactory.create(LossType.STFT_RMSE)

        self.rwkv = lambda dim, reshape: Seq(
            ReshapeBCFT(
                reshape,
                Residual(
                    nn.RMSNorm(dim),
                    BiRWKVLayer(dim),
                ),
                Residual(
                    nn.RMSNorm(dim),
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
            stride=(2, 2),
            output_channels=8,
            post_downsample_fn=lambda shape: Seq(
            ),
            bottleneck_fn=lambda shape: Seq(
                # Repeat(
                #     1,
                #     self.rwkv(shape.f, "(b c) t f")
                # ),
            ),
            post_upsample_fn=lambda shape: Seq(
            ),
            post_skip_fn=lambda shape: Seq(
                # Repeat(
                #     1,
                #     self.rwkv(shape.f, "(b c) t f")
                # ),
            )
        )

        self.model = ComplexMask(
            WithShape(
                shape=CFTShape(c=4, f=self.n_fft // 2, t=87),
                fn=lambda shape: Seq(
                    DebugShape("Before UNet"),
                    self.unet(shape=shape),
                    DebugShape("After UNet"),
                    nn.Tanh(),
                ),
            ),
        )

        self.stft = STFT(n_fft=self.n_fft, hop_length=self.hop_length)

    def get_mode(self):
        return SeparationMode.ONE_SHOT

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.stft(waveform)

    def decode(self, encoded: torch.Tensor, original_length: int = None) -> torch.Tensor:
        print(f"Encoded shape in decode: {encoded.shape}")
        return self.stft.inverse(encoded, original_length)

    def process(self, x, mixture, t):
        return self.model(x)
