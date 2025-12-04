import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from model.base_model import BaseModel, SeparationMode
from model.magsep.rwkv import BiRWKVLayer
from modules.functional import ReshapeBCFT, WithShape, CFTShape, Residual, Repeat, ComplexMask, ToSTFT, InverseSTFT, \
    Mask, DebugShape, Bandsplit
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
        self.loss_factory = LossFactory.create(LossType.MULTI_STFT)

        self.rwkv = lambda dim, reshape: Seq(
            ReshapeBCFT(
                reshape,
                Residual(
                    nn.LayerNorm(dim),
                    BiRWKVLayer(dim),
                ),
                Residual(
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
            channels=[shape.c, 32, 64, 128, 256],
            stride=(2, 1),
            output_channels=shape.c * self.num_instruments,
            post_downsample_fn=lambda shape: Seq(
            ),
            bottleneck_fn=lambda shape: Seq(
            ),
            post_upsample_fn=lambda shape: Seq(
            ),
            post_skip_fn=lambda shape: Seq(
            )
        )

        self.model = Seq(
            WithShape(
                shape=CFTShape(c=4, f=self.n_fft // 2, t=87),
                fn=lambda shape: Seq(
                    Bandsplit(
                        shape=shape,
                        num_splits=1,
                        fn=lambda shape: Seq(
                            self.unet(shape=shape),
                        )
                    ),
                    self.visualize("mask")
                ),
            ),
        )

        self.stft = STFT(n_fft=self.n_fft, hop_length=self.hop_length)
        self.original_length = None

    def get_mode(self):
        return SeparationMode.ONE_SHOT

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        self.original_length = waveform.shape[-1]
        return self.stft(waveform)

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.stft.inverse(encoded, self.original_length)

    def process(self, x, mixture, t):
        return self.model(x)

    def loss(self, pred, targets, mixture):
        pred = self.decode(pred)
        targets = self.decode(targets)
        return self.loss_factory.calculate(pred, targets)
