from abc import ABC, abstractmethod

import torch
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
        self.is_validating = False

        self.visualize = lambda name, transform=Seq(nn.Identity()): Condition(
            condition=lambda x: True,
            true_fn=lambda: SideEffect(
                Seq(
                    transform,
                    VisualizationHook(name),
                )
            ),
        )

        self.loss_factory = LossFactory.create(LossType.MULTI_STFT)

    @abstractmethod
    def process(self, x, mixture=None, t=None):
        pass

    def loss(self, x, targets, mixture):
        return self.loss_factory.calculate(x, targets)

    def forward(self, x, mixture=None, targets=None, t=None):
        pred = self.process(x, mixture=mixture, t=t)
        loss = None
        if targets is not None:
            loss = self.loss_factory.calculate(pred, targets)

        return pred, loss

    def valid_forward(self, x, targets=None, t=None):
        print("valid forward t mean: ", t.mean().item() if t is not None else 'None')
        self.is_validating = True
        result = self.forward(x, targets, t=t)
        self.is_validating = False
        return result

    def debug_forward(self, mixture, separated, targets):
        """
        Visualize debug outputs.

        Args:
            mixture: Original mixture waveform (b, c, t)
            separated: Model's separated output from iterative_inference (b, n, c, t)
            targets: Ground truth targets (b, n, c, t)
        """
        self.is_debug = True
        self.visualize("progress_mag", Seq(
            Rearrange("b n c t -> (b n) c t"),
            ToSTFT(),
            ToMagnitude(),
        ))(separated)
        self.visualize("target_mag", Seq(
            Rearrange("b n c t -> (b n) c t"),
            ToSTFT(),
            ToMagnitude(),
        ))(targets)

        self.visualize("progress_stft", Seq(
            Rearrange("b n c t -> (b n) c t"),
            ToSTFT(),
        ))(separated)
        self.visualize("mixture_mag", Seq(
            ToSTFT(),
            ToMagnitude(),
        ))(mixture)
        self.visualize("mixture_stft", Seq(
            ToSTFT(),
        ))(mixture)
        self.visualize("target_stft", Seq(
            Rearrange("b n c t -> (b n) c t"),
            ToSTFT(),
        ))(targets)

        self.is_debug = False
