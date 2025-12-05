import time
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from dataset.ChirpDataset import ChirpDataset
from dataset.inference_dataset import InferenceDataset
from dataset.train_dataset import TrainDataset
from dataset.validation_dataset import TestSongDataset
from model.autoencoder.autoencoder import AutoencoderModel
from model.base_model import SeparationMode
from model.bs_roformer.model import BSRoformer
from model.magsep.model import MagSplitModel
from model.vae.vae import VAEModel
from modules.functional import ToMagnitude, FromMagnitude, DebugShape
from modules.seq import Seq
from modules.stft import STFT
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

        # Mode: "one_shot" or "flow_matching"
        self.mode = self.model.get_mode()

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
                aligned_mixture=True,
                augment=False
            )
        )
        self.debug_mixture = None
        self.debug_targets = None
        self.num_instruments = len(config.training.target_sources)
        self.debug_batch = None

        # Noise factor: 0.0 = start from mixture, 1.0 = start from pure noise
        self.noise_factor = 0.0  # getattr(config.training, 'noise_factor', 0.0)

    def get_model(self, config):
        model_type = config.model.type

        if model_type == "MagSep":
            return MagSplitModel(config=config)
        elif model_type == "BSRoformer":
            return BSRoformer(
                n_fft=config.model.n_fft,
                hop_length=config.model.hop_length,
                layers=config.model.layers,
                mask_layers=config.model.mask_layers,
                dropout=config.model.dropout,
                embed_dim=config.model.embed_dim,
                freqs_per_bands=config.model.freqs_per_bands,
            )
        elif model_type == "VAE":
            return VAEModel(config=config)
        elif model_type == "Autoencoder":
            return AutoencoderModel(config=config)
        else:
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                "Supported types are 'MagSplitModel' and 'BSRoformer'."
            )

    def sample_t(self, batch_size: int, device: torch.device, logit_normal: bool = False) -> torch.Tensor:
        if logit_normal:
            t = torch.sigmoid(torch.randn(batch_size, device=device))
        else:
            t = torch.rand(batch_size, device=device)

        t = t.clamp(1e-4, 1 - 1e-4)
        return t.view(-1, 1, 1, 1)

    def to_flow_matching(self, batch):
        """
        Prepare batch for Rectified Flow training.
        Pure linear interpolation: x_t = (1-t)*x_0 + t*x_1

        With noise_factor:
        - x_0 is interpolated between mixture and gaussian noise
        - x_0 = (1 - noise_factor) * mixture + noise_factor * noise
        """
        mixture, targets = batch

        mixture_encoded = self.model.encode(mixture)

        targets = rearrange(targets, "b n c t -> (b n) c t")
        targets_encoded = self.model.encode(targets)

        # Expand mixture to match the shape of flattened targets
        mixture_encoded_expanded = repeat(mixture_encoded, "b c f t -> (b n) c f t", n=self.num_instruments)

        b = targets_encoded.shape[0]
        t = self.sample_t(b, targets_encoded.device)

        # x_0: mixture plus scaled gaussian noise (matched to mixture statistics)
        noise = torch.randn_like(mixture_encoded_expanded)
        mixture_mean = mixture_encoded_expanded.mean()
        mixture_std = mixture_encoded_expanded.std()
        noise = noise * mixture_std + mixture_mean
        x_0 = mixture_encoded_expanded + self.noise_factor * noise

        # x_1: target sources (ending point)
        x_1 = targets_encoded

        # Rectified Flow interpolant: x_t = (1-t)*x_0 + t*x_1
        x_t = (1 - t) * x_0 + t * x_1

        # Target velocity: constant along the straight path
        target_velocity = x_1 - x_0

        return x_t, mixture_encoded, target_velocity, t

    # Returns mixture = b c t and targets = b (n c) t
    def to_one_shot(self, batch):
        """
        Prepare batch for one-shot source separation.
        Model directly predicts target sources from mixture.
        """
        mixture, targets = batch

        mixture_encoded = self.model.encode(mixture)
        targets_encoded = self.model.encode(targets)

        return mixture_encoded, targets_encoded

    def training_step(self, batch, batch_idx):
        if self.mode == "flow_matching":
            return self._training_step_flow_matching(batch, batch_idx)
        else:
            return self._training_step_one_shot(batch, batch_idx)

    def _training_step_flow_matching(self, batch, batch_idx):
        x_t, mixture_encoded, target_velocity, t = self.to_flow_matching(batch)

        predicted_velocity, loss = self.model(
            x_t,
            mixture=mixture_encoded,
            targets=target_velocity,
            t=t
        )

        validation_output = self.validation_loss_step()
        self._debug_step()

        self.log("train/t_mean", t.mean(), prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/noise_factor", self.noise_factor, prog_bar=False)

        mixture_waveform = self.model.decode(mixture_encoded)
        mixture_waveform = rearrange(mixture_waveform, "(b n) c t -> b n c t", n=self.num_instruments)

        target_velocity_waveform = self.model.decode(target_velocity)
        target_velocity_waveform = rearrange(target_velocity_waveform, "(b n) c t -> b n c t", n=self.num_instruments)

        predicted_velocity_waveform = self.model.decode(predicted_velocity)
        predicted_velocity_waveform = rearrange(predicted_velocity_waveform, "(b n) c t -> b n c t",
                                                n=self.num_instruments)

        return {
            "loss": loss,
            "separated": predicted_velocity_waveform,
            "targets": target_velocity_waveform,
            "mixture": mixture_waveform,
            "t": t,
            "val_loss": validation_output.loss if validation_output else 0.0,
            "val_separated": validation_output.separated,
            "val_targets": validation_output.targets,
            "val_mixture": validation_output.mixture,
        }

    def _training_step_one_shot(self, batch, batch_idx):
        # b,c,t and b (n c) t
        mixture_encoded, targets_encoded = self.to_one_shot(batch)

        # For one-shot, model predicts targets directly (no time conditioning)
        predicted, loss = self.model(
            x=mixture_encoded,
            targets=targets_encoded
        )

        validation_output = self.validation_loss_step()
        self._debug_step()

        self.log("train/loss", loss, prog_bar=True)

        mixture_waveform = self.model.decode(mixture_encoded)
        mixture_waveform = rearrange(mixture_waveform, "(b n) c t -> b n c t", n=self.num_instruments)

        targets_waveform = self.model.decode(targets_encoded)
        targets_waveform = rearrange(targets_waveform, "(b n) c t -> b n c t", n=self.num_instruments)

        predicted_waveform = self.model.decode(predicted)
        predicted_waveform = rearrange(predicted_waveform, "(b n) c t -> b n c t", n=self.num_instruments)

        return {
            "loss": loss,
            "separated": predicted_waveform,
            "targets": targets_waveform,
            "mixture": mixture_waveform,
            "t": None,
            "val_loss": validation_output.loss if validation_output else 0.0,
            "val_separated": validation_output.separated,
            "val_targets": validation_output.targets,
            "val_mixture": validation_output.mixture,
        }

    @torch.no_grad()
    def inference(self, mixture: torch.Tensor, steps: int = None, noise_factor: float = None) -> torch.Tensor:
        """
        Inference method that adapts to the training mode.

        For flow_matching: Deterministic Euler integration from t=0 to t=1.
        For one_shot: Single forward pass prediction.
        """
        if self.mode == SeparationMode.ONE_SHOT:
            return self._one_shot_inference(mixture)
        else:
            return self._flow_matching_inference(mixture, steps, noise_factor)

    @torch.no_grad()
    def _one_shot_inference(self, mixture: torch.Tensor) -> torch.Tensor:
        """Single forward pass inference for one-shot mode."""
        b, c, t_audio = mixture.shape

        mixture_encoded = self.model.encode(mixture.float())
        predicted, loss = self.model(mixture_encoded, mixture=mixture_encoded, t=None)
        x = self.model.decode(predicted)
        return x

    @torch.no_grad()
    def _flow_matching_inference(self, mixture: torch.Tensor, steps: int = None,
                                 noise_factor: float = None) -> torch.Tensor:
        """Euler integration inference for flow matching mode."""
        if steps is None:
            steps = self.inference_steps
        if noise_factor is None:
            noise_factor = self.noise_factor

        mixture_encoded = self.model.encode(mixture.float())

        # Expand mixture for all instruments
        mixture_encoded_expanded = repeat(mixture_encoded, "b c f t -> (b n) c f t", n=self.num_instruments)

        # Start from x_0: mixture plus scaled noise (matched to mixture statistics)
        noise = torch.randn_like(mixture_encoded_expanded)
        mixture_mean = mixture_encoded_expanded.mean()
        mixture_std = mixture_encoded_expanded.std()
        noise = noise * mixture_std + mixture_mean
        x = mixture_encoded_expanded + noise_factor * noise

        dt = 1.0 / steps

        for step in range(steps):
            t_val = torch.full((x.shape[0], 1, 1, 1), step * dt, device=mixture.device)

            predicted_velocity, _ = self.model(x, mixture=mixture_encoded, t=t_val)

            # Pure Euler step
            x = x + predicted_velocity * dt

        x = self.model.decode(x)
        x = rearrange(x, "(b n) c t -> b n c t", n=self.num_instruments)

        return x

    def on_train_start(self):
        pass

    def _debug_step(self):
        debug_every = self.config.logging.debug_every
        if (self.debug_mixture is None) or (self.debug_targets is None):
            self.debug_mixture, self.debug_targets = next(self.validation_iter)
            self.debug_mixture = rearrange(self.debug_mixture, "c t -> 1 c t").to(self.device)
            self.debug_targets = rearrange(self.debug_targets, "c t -> 1 c t").to(self.device)

        if debug_every != 0 and self.global_step % debug_every == 0 and self.global_step != 0:
            self.model.eval()
            with torch.no_grad():
                self.model.is_debug = True
                debug_separated = self.inference(self.debug_mixture, steps=self.config.logging.debug_steps)
                self.model.visualize_debug(self.debug_mixture, debug_separated, self.debug_targets)
                self.model.is_debug = False
            self.model.train()

    def validation_loss_step(self):
        validate_every = self.config.logging.validate_every
        if validate_every != 0 and self.global_step != 0 and self.global_step % validate_every == 0:
            # Grab batch_size items from the validation iterator
            val_mixtures = []
            val_targets_list = []
            for _ in range(self.config.inference.batch_size):
                val_mixture, val_targets = next(self.validation_iter)
                val_mixtures.append(val_mixture)
                val_targets_list.append(val_targets)

            # Stack along batch dimension: each item is (c, t) -> stacked is (b, c, t)
            val_mixture = torch.stack(val_mixtures, dim=0).to(self.device)
            val_targets = torch.stack(val_targets_list, dim=0).to(self.device)

            self.model.eval()

            # For one-shot, steps=1 is natural; for flow matching, use configured steps
            inference_steps = 1
            val_separated = self.inference(val_mixture, steps=inference_steps)

            # TODO: use the loss from model
            val_loss = F.mse_loss(val_separated, val_targets)

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
            targets=self.config.training.target_sources,
            augment=self.config.training.augment
        )
        num_workers = 6
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
        predictions = self.separator.process_file(
            model=self,
            mixture_path=mixture_path,
            output_dir=self.config.paths.validation
        )
        return {
            "mixture_path": mixture_path,
            "predictions": predictions,
        }

    def configure_optimizers(self):
        optimizer = OptimizerFactory(self.model).get_optimizer(
            optimizer_type=self.config.training.optimizer,
            lr=float(self.config.training.learning_rate)
        )
        return optimizer
