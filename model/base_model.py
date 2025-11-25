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

        self.visualize = lambda name, transform=nn.Identity(): Condition(
            condition=lambda x: self.is_debug,
            true_fn=lambda: SideEffect(
                Seq(
                    transform,
                    VisualizationHook(name),
                )
            ),
        )

        self.loss_factory = LossFactory.create(LossType.MULTI_STFT)

    @abstractmethod
    def process(self, x, t=None):
        """
        Process input with optional time conditioning.

        In 'Predict Source' formulation:
        - Input x is x_t = (1-t)*mixture + t*target (a blended state)
        - Model predicts the clean target (separated sources)
        - This prediction is used to calculate velocity during ODE solving

        Args:
            x: Input tensor
               - (b, n, c, samples) for blended input during training/iterative inference
               - (b, c, samples) for mixture during single-pass inference
            t: Optional time conditioning (b,) in [0, 1]
               - t=0 means input is pure mixture
               - t=1 means input is pure target

        Returns:
            Processed tensor (b, n, c, samples) - predicted clean separated sources
        """
        pass

    def loss(self, x, targets, mixture):
        return self.loss_factory.calculate(x, targets)

    def forward(self, x, targets=None, t=None):
        """
        Forward pass with 'Predict Source' formulation.

        The model sees x_t (blended input) and predicts the clean target sources.
        Loss is computed by comparing predicted sources to ground truth targets.

        Args:
            x: Input tensor
               - (b, n, c, samples) blended input x_t = (1-t)*mixture + t*target
               - (b, c, samples) mixture during single-pass inference
            targets: Optional target sources (b, n, c, samples) for loss computation
            t: Optional time conditioning (b,) indicating blend ratio

        Returns:
            Tuple of (predicted_sources, loss)
            - predicted_sources: (b, n, c, samples) - model's prediction of clean sources
            - loss: scalar loss if targets provided, None otherwise
        """
        # Model predicts clean sources from blended input
        predicted_sources = self.process(x, t=t)

        # Calculate loss between prediction and clean target
        loss = None
        if targets is not None:
            # Note: we don't need mixture here anymore for 'Predict Source' formulation
            loss = self.loss_factory.calculate(predicted_sources, targets)

        return predicted_sources, loss

    def valid_forward(self, x, targets=None, t=None):
        print("valid forward t mean: ", t.mean().item() if t is not None else 'None')
        self.is_validating = True
        result = self.forward(x, targets, t=t)
        self.is_validating = False
        return result

    def debug_forward(self, x, targets=None):
        self.is_debug = True
        mixture = x
        t = torch.ones(x.shape[0], device=x.device)
        x, _ = self.forward(x, t=t)

        self.visualize("mixture_mag", Seq(
            ToSTFT(),
            ToMagnitude(),
        ))(mixture)
        self.visualize("progress_mag", Seq(
            Rearrange("b n c t -> (b n) c t"),
            ToSTFT(),
            ToMagnitude(),
        ))(x)
        self.visualize("target_mag", Seq(
            Rearrange("b n c t -> (b n) c t"),
            ToSTFT(),
            ToMagnitude(),
        ))(targets)
        self.visualize("mixture_stft", Seq(
            ToSTFT(),
        ))(mixture)
        self.visualize("progress_stft", Seq(
            Rearrange("b n c t -> (b n) c t"),
            ToSTFT(),
        ))(x)
        self.visualize("target_stft", Seq(
            Rearrange("b n c t -> (b n) c t"),
            ToSTFT(),
        ))(targets)
        self.is_debug = False
