import enum
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from modules.functional import ToSTFT, Condition, ToMagnitude, SideEffect
from modules.seq import Seq
from modules.visualize import VisualizationHook


class SeparationMode(enum.Enum):
    ONE_SHOT = "one_shot"
    FLOW_MATCHING = "flow_matching"


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.is_debug = False
        self.is_validating = False

        self.visualize = lambda name, transform=Seq(nn.Identity()): Condition(
            condition=lambda x: self.is_debug,
            true_fn=lambda: SideEffect(
                Seq(
                    transform,
                    VisualizationHook(name),
                )
            ),
        )

        self.stft = ToSTFT()

    @abstractmethod
    def get_mode(self):
        pass

    @abstractmethod
    def process(self, x, mixture, t):
        pass

    @abstractmethod
    def loss(self, pred, targets, mixture):
        pass

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform  # Identity by default

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded  # Identity by default

    def forward(self, x, mixture=None, targets=None, t=None):
        pred = self.process(x, mixture=mixture, t=t)
        loss = None
        if targets is not None:
            loss = self.loss(pred, targets, mixture)

        return pred, loss

    def valid_forward(self, x, targets=None, t=None):
        print("valid forward t mean: ", t.mean().item() if t is not None else 'None')
        self.is_validating = True
        result = self.forward(x, targets, t=t)
        self.is_validating = False
        return result

    def visualize_debug(self, mixture, separated, targets):
        """
        Visualize debug outputs.

        Args:
            mixture: Original mixture waveform (b, c, t)
            separated: Model's separated output from iterative_inference (b, n, c, t)
            targets: Ground truth targets (b, n, c, t)
        """

        if mixture.ndim == 3:
            mixture = self.stft(mixture)

        if separated.ndim == 3:
            separated = self.stft(separated)

        if targets.ndim == 3:
            targets = self.stft(targets)

        self.visualize("progress_mag", Seq(
            ToMagnitude(),
        ))(separated)
        self.visualize("target_mag", Seq(
            ToMagnitude(),
        ))(targets)

        self.visualize("progress_stft", Seq(
        ))(separated)
        self.visualize("mixture_mag", Seq(
            ToMagnitude(),
        ))(mixture)
