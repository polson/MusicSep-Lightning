import statistics
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import torchaudio
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from util.metrics import sdr


class AudioSeparationProgressBar(RichProgressBar):

    def __init__(
            self,
            target_sources: List[str],
            hide_columns: List[str] = None,
    ):
        if hide_columns is None:
            hide_columns = ["loss"]

        theme = RichProgressBarTheme(
            description="yellow",
            progress_bar="cyan1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="cyan",
            time="grey82",
            processing_speed="green_yellow",
            metrics="cyan",
        )
        super().__init__(theme=theme, refresh_rate=1, leave=True)
        self.target_sources: List[str] = target_sources
        self.hide_columns: List[str] = hide_columns
        self.loss_values: deque[float] = deque(maxlen=100)
        self.train_sdr_values: deque[float] = deque(maxlen=100)
        self.val_loss_values: deque[float] = deque(maxlen=100)
        self.val_sdr_values: deque[float] = deque(maxlen=100)

    def configure_columns(self, trainer: Trainer) -> List[Any]:
        columns = super().configure_columns(trainer)
        return columns

    def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int
    ):
        loss, avg_loss = self._process_loss(outputs["loss"], trainer.accumulate_grad_batches)
        train_sdr = sdr(outputs["targets"], outputs["separated"])
        self.train_sdr_values.append(train_sdr)

        pl_module.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        pl_module.log("avg_loss", avg_loss, prog_bar=True, on_step=True, on_epoch=False)
        pl_module.log("train_avg_sdr", statistics.mean(self.train_sdr_values), prog_bar=True, on_step=True,
                      on_epoch=False)

        val_targets = outputs["val_targets"]
        val_separated = outputs["val_separated"]
        val_loss = outputs["val_loss"]
        if val_targets is not None and val_separated is not None and val_loss is not None:
            val_sdr = sdr(val_targets, val_separated)
            self.val_sdr_values.append(val_sdr)
            self.val_loss_values.append(val_loss.item())
            pl_module.log("val_train_loss", statistics.mean(self.val_loss_values), prog_bar=True, on_step=True,
                          on_epoch=False)
            pl_module.log("val_train_sdr", statistics.mean(self.val_sdr_values), prog_bar=True, on_step=True,
                          on_epoch=False)

        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        predictions = outputs["predictions"]
        target_paths = outputs["target_paths"]
        total_score = 0
        for i, prediction in enumerate(predictions):
            target_path = target_paths[i][0]
            target_name = Path(target_path).stem
            prediction = prediction.unsqueeze(0)
            target_tensor = torchaudio.load(target_path)[0].unsqueeze(0).to(prediction)
            score = sdr(target_tensor, prediction, use_mean=False)
            total_score += score
            pl_module.log(f"val_{target_name}", score, prog_bar=True, batch_size=1)
        pl_module.log(f"val_avg_sdr", total_score / len(predictions), prog_bar=True, batch_size=1)

    def _process_loss(self, loss, grad_accum):
        loss_value: float = loss.item()
        scaled_loss: float = loss_value * grad_accum
        self.loss_values.append(scaled_loss)
        avg_loss: float = sum(self.loss_values) / len(self.loss_values)
        return scaled_loss, avg_loss

    def _process_sdr(self, targets, separated, pl_module: LightningModule):
        targets_tensor: torch.Tensor = targets.float()
        separated_tensor: torch.Tensor = separated.float()
        out_sdr = sdr(targets_tensor, separated_tensor)
        return out_sdr

    def get_metrics(self, trainer: Trainer, model: LightningModule) -> Dict[str, Union[int, str, float]]:
        items: Dict[str, Union[int, str, float]] = super().get_metrics(trainer, model)
        for column in self.hide_columns:
            if column in items:
                items.pop(column, None)
        return items
