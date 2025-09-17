import torch
from torch import optim
from enum import Enum, auto


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    PRODIGY = "prodigy"


class OptimizerFactory:

    def __init__(self, model):
        self.model = model
        self.optimizer = None

    def _get_optimizer_enum(self, optimizer_type):
        if isinstance(optimizer_type, str):
            try:
                return next(t for t in OptimizerType if t.value == optimizer_type)
            except StopIteration:
                return OptimizerType.ADAM
        return optimizer_type

    def get_optimizer(self, optimizer_type=OptimizerType.ADAM, lr=1e-3, **kwargs):
        optimizer_type = self._get_optimizer_enum(optimizer_type)

        match optimizer_type:
            case OptimizerType.ADAM:
                return self._configure_adam(lr=lr, **kwargs)
            case OptimizerType.ADAMW:
                return self._configure_adamw(lr=lr, **kwargs)
            case OptimizerType.PRODIGY:
                return self._configure_prodigy(lr=lr, **kwargs)
            case _:
                return self._configure_adam(lr=lr, **kwargs)

    def _configure_adam(self, lr=1e-3, **kwargs):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_avg_sdr",
                "interval": "step",
                "frequency": 1000
            }
        }

    def _configure_adamw(self, lr=1e-3, **kwargs):
        weight_decay = kwargs.get('weight_decay', 1e-2)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        return {
            "optimizer": optimizer,
        }

    def _configure_prodigy(self, lr=None, **kwargs):
        try:
            from prodigyopt import Prodigy
        except ImportError:
            raise ImportError(
                "Prodigy optimizer not found. Install it with: pip install prodigyopt"
            )

        optimizer = Prodigy(
            self.model.parameters(),
            weight_decay=0.01,
        )

        return {
            "optimizer": optimizer
        }
