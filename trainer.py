from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
from einops import rearrange

from dataset.inference_dataset import InferenceDataset
from dataset.train_dataset import TrainDataset
from dataset.validation_dataset import TestSongDataset
from model.bs_roformer.model import BSRoformer
from model.magsep.model import MagSplitModel
from optimizer import OptimizerFactory
from separator import Separator


@dataclass
class ValidationOutput:
    loss: float = 0.0
    separated: torch.Tensor = None
    targets: torch.Tensor = None
    mixture: torch.Tensor = None


class AudioSourceSeparation(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self.get_model(config)
        self.inference_steps = config.inference.steps
        self.separator = Separator(
            target_sources=config.training.target_sources,
            batch_size=config.inference.batch_size,
            overlap_percent=0.5,
            chunk_size_seconds=config.inference.duration,
            steps=self.inference_steps,
        )
        self.validation_iter = iter(
            TrainDataset(
                root_dir=str(Path(self.config.paths.dataset) / "test"),
                duration_seconds=config.training.duration,
                targets=config.training.target_sources,
                aligned_mixture=False
            )
        )
        self.debug_mixture = None
        self.debug_targets = None
        self.num_instruments = len(config.training.target_sources)

    def get_model(self, config):
        model_type = config.model.type

        if model_type == "MagSep":
            return MagSplitModel(
                num_instruments=len(config.training.target_sources),
                n_fft=config.model.n_fft,
                hop_length=config.model.hop_length,
                layers=config.model.layers,
                splits=config.model.splits,
            )
        elif model_type == "BSRoformer":
            return BSRoformer(
                num_instruments=len(config.training.target_sources),
                n_fft=config.model.n_fft,
                hop_length=config.model.hop_length,
                layers=config.model.layers,
                mask_layers=config.model.mask_layers,
                dropout=config.model.dropout,
                embed_dim=config.model.embed_dim,
                freqs_per_bands=config.model.freqs_per_bands,
            )
        else:
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                "Supported types are 'MagSplitModel' and 'BSRoformer'."
            )

    def on_train_start(self):
        val_mixture, val_targets = next(self.validation_iter)
        val_mixture = rearrange(val_mixture, "c t -> 1 c t").to(self.device)
        val_targets = rearrange(val_targets, "n c t -> 1 n c t", n=self.num_instruments).to(self.device)
        self.debug_mixture = val_mixture
        self.debug_targets = val_targets

    def training_step(self, batch, batch_idx):
        mixture, targets = batch
        b = mixture.shape[0]
        t = torch.rand(b, 1, 1).to(mixture.device)
        noise = torch.randn_like(targets)  # noise should match targets shape

        # Don't flatten targets before interpolation
        noisy = (1 - t.unsqueeze(1)) * noise + t.unsqueeze(1) * targets  # b,n,c,t
        target_velocity = targets - noise

        # Flatten for model input
        noisy_flat = rearrange(noisy, "b n c t -> (b n) c t")
        target_velocity_flat = rearrange(target_velocity, "b n c t -> (b n) c t")

        # Repeat mixture for each instrument
        mixture_expanded = mixture.unsqueeze(1).expand(-1, self.num_instruments, -1, -1)
        mixture_flat = rearrange(mixture_expanded, "b n c t -> (b n) c t")

        # Pass mixture as conditioning
        predicted_velocity, loss = self.model(
            noisy_flat,
            mixture=mixture_flat,
            targets=target_velocity_flat,
            t=t
        )

        validation_output = self.validation_loss_step()
        self._debug_step()

        self.log("train/t_mean", t.mean(), prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)

        mixture_expanded = rearrange(mixture, "b c t -> b 1 c t")

        return {
            "loss": loss,
            "separated": predicted_velocity,  # expects b,n,c,t
            "targets": target_velocity,  # expects b,n,c,t
            "mixture": mixture_expanded,
            "t": t,
            "val_loss": validation_output.loss if validation_output else 0.0,
            "val_separated": validation_output.separated,
            "val_targets": validation_output.targets,
            "val_mixture": validation_output.mixture,
        }

    def _debug_step(self):
        debug_every = self.config.logging.debug_every
        if debug_every != 0 and self.global_step % debug_every == 0 and self.global_step != 0:
            self.model.eval()
            with torch.no_grad():
                self.model.debug_forward(self.debug_mixture, self.debug_targets)
            self.model.train()

    def validation_loss_step(self):
        validate_every = self.config.logging.validate_every
        if validate_every != 0 and self.global_step % validate_every == 0:
            val_mixture, val_targets = next(self.validation_iter)
            val_mixture = rearrange(val_mixture, "c t -> 1 c t").to(self.device)
            val_targets = rearrange(val_targets, "n c t -> 1 n c t", n=self.num_instruments).to(self.device)
            self.model.eval()

            # Use iterative inference for validation
            val_separated = self.iterative_inference(val_mixture)
            val_loss = torch.nn.functional.mse_loss(val_separated, val_targets)

            self.model.train()
            return ValidationOutput(
                loss=val_loss,
                separated=val_separated,
                targets=val_targets,
                mixture=val_mixture,
            )
        return ValidationOutput()

    def train_dataloader(self):
        train_dataset = TrainDataset(
            root_dir=str(Path(self.config.paths.dataset) / "train"),
            duration_seconds=self.config.training.duration,
            targets=self.config.training.target_sources
        )
        num_workers = 2
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )

    def val_dataloader(self):
        dataset_path = self.config.paths.dataset
        test_dir = Path(dataset_path) / "test"
        val_dataset = TestSongDataset(test_dir, self.config.training.target_sources)
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=0
        )

    def predict_dataloader(self):
        inference_dir = Path(self.config.paths.inference_input)
        dataset = InferenceDataset(inference_dir, self.config.training.target_sources)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0
        )

    def validation_step(self, batch, batch_idx):
        mixture_path = batch["mixture_path"][0]
        target_paths = batch["target_paths"]
        predictions = self.separator.process_file(
            model=self,
            mixture_path=mixture_path,
            output_dir=self.config.paths.validation
        )
        return {
            "mixture_path": mixture_path,
            "predictions": predictions,
            "target_paths": target_paths,
        }

    def predict_step(self, batch, batch_idx):
        mixture_path = batch["mixture_path"][0]
        # this calls forward internally, todo refactor
        predictions = self.separator.process_file(
            model=self,
            mixture_path=mixture_path,
            output_dir=self.config.paths.validation
        )
        return {
            "mixture_path": mixture_path,
            "predictions": predictions,
        }

    def forward(self, x):
        """Forward pass uses iterative inference."""
        return self.iterative_inference(x)

    @torch.no_grad()
    def iterative_inference(self, mixture: torch.Tensor, steps: int = None) -> torch.Tensor:
        if steps is None:
            steps = self.inference_steps

        b, c, t = mixture.shape
        x = torch.randn(b, self.num_instruments, c, t, device=mixture.device)

        # Expand mixture for each instrument
        mixture_expanded = mixture.unsqueeze(1).expand(-1, self.num_instruments, -1, -1)
        mixture_flat = rearrange(mixture_expanded, "b n c t -> (b n) c t")

        dt = 1.0 / steps

        for step in range(steps):
            t_val = torch.full((b, 1, 1), step * dt, device=mixture.device)
            x_flat = rearrange(x, "b n c t -> (b n) c t")

            predicted_velocity, _ = self.model(x_flat, mixture=mixture_flat, t=t_val)
            x = x + predicted_velocity * dt

        return x

    def configure_optimizers(self):
        optimizer = OptimizerFactory(self.model).get_optimizer(
            optimizer_type=self.config.training.optimizer,
            lr=float(self.config.training.learning_rate)
        )
        return optimizer
