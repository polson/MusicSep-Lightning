import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from enum import Enum, auto


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    PRODIGY = "prodigy"


class OptimizerFactory:

    def __init__(self, model, total_steps=None):
        self.model = model
        self.total_steps = total_steps
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

    def _configure_adamw(self, lr=2e-4, **kwargs):
        weight_decay = kwargs.get('weight_decay', 0.01)
        betas = kwargs.get('betas', (0.9, 0.99))
        eps = kwargs.get('eps', 1e-8)
        warmup_ratio = kwargs.get('warmup_ratio', 0.1)
        min_lr = kwargs.get('min_lr', 1e-6)
        grad_clip = kwargs.get('grad_clip', 1.0)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps
        )

        result = {
            "optimizer": optimizer,
            "gradient_clip_val": grad_clip,
        }

        # Add cosine schedule with warmup if total_steps is provided
        if self.total_steps is not None:
            warmup_steps = int(self.total_steps * warmup_ratio)

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps
            )

            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.total_steps - warmup_steps,
                eta_min=min_lr
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )

            result["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }

        return result

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
