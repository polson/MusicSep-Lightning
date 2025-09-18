from abc import ABC, abstractmethod

import torch.nn as nn
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from modules.functional import ToSTFT, Condition, ToMagnitude, SideEffect
from modules.seq import Seq
from modules.visualize import VisualizationHook


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.is_debug = False

        self.visualize = lambda name, transform=nn.Identity(): Condition(
            condition=lambda x: self.is_debug,
            true_fn=lambda: SideEffect(
                Seq(
                    transform,
                    VisualizationHook(name),
                )
            ),
        )

    @abstractmethod
    def process(self, x):
        pass

    def loss(self, x, targets, mixture):
        loss_factory = LossFactory.create(LossType.STFT_RMSE)
        return loss_factory.calculate(x, targets)

    def forward(self, x, targets=None):
        mixture = x
        x = self.process(x)
        loss = self.loss(x, targets, mixture) if targets is not None else None
        return x, loss

    def debug_forward(self, x, targets=None):
        self.is_debug = True
        mixture = x
        x = self.forward(x)
        self.visualize("mixture", Seq(
            ToSTFT(),
            ToMagnitude(),
        ))(mixture)
        self.visualize("progress", Seq(
            Rearrange("b n c t -> (b n) c t"),
            ToSTFT(),
            ToMagnitude(),
        ))(x)
        self.visualize("target", Seq(
            Rearrange("b n c t -> (b n) c t"),
            ToSTFT(),
            ToMagnitude(),
        ))(targets)
        self.is_debug = False
