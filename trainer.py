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
from model.bs_roformer.model import BSRoformer
from model.magsep.model import MagSplitModel
from modules.functional import InverseSTFT, ToSTFT, ToMagnitude, FromMagnitude, DebugShape
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
                aligned_mixture=config.training.aligned
            )
        )
        self.debug_mixture = None
        self.debug_targets = None
        self.num_instruments = len(config.training.target_sources)
        self.debug_batch = None

        self.stft = ToSTFT(
            n_fft=config.model.n_fft,
            hop_length=config.model.hop_length
        )

        self.inverse_stft = InverseSTFT(
            n_fft=config.model.n_fft,
            hop_length=config.model.hop_length
        )

        self.noise_scale = 0.1
        self.sigma = getattr(config.model, 'sigma', None) or 0.005

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

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Logit-normal timestep sampling."""
        normal_samples = torch.randn(batch_size, device=device)
        t = torch.sigmoid(normal_samples)
        t = t.clamp(1e-4, 1 - 1e-4)
        return t.view(-1, 1, 1, 1)

    def to_flow_matching(self, batch):
        """
        Prepare batch for OT-CFM training with Noise matching Mixture Statistics.
        """
        mixture, targets = batch

        mixture_stft = self.stft(mixture)

        targets = rearrange(targets, "b n c t -> (b n) c t")
        targets_stft = self.stft(targets)

        # Expand mixture to match the shape of flattened targets
        # Shape: (Batch * Num_Instruments, Channels, Freq, Time)
        mixture_stft_expanded = repeat(mixture_stft, "b c f t -> (b n) c f t", n=self.num_instruments)

        b = targets_stft.shape[0]
        t = self.sample_t(b, targets_stft.device)

        # --- KEY CHANGE START ---
        # 1. Generate standard Gaussian noise
        raw_noise = torch.randn_like(targets_stft)

        # 2. Calculate Mean and Std of the mixture per sample
        # We calculate over Freq (-2) and Time (-1) to get stats for each spectrogram
        # keepdim=True ensures shape is (b*n, c, 1, 1) for broadcasting
        mix_mean = mixture_stft_expanded.mean(dim=(-2, -1), keepdim=True)
        mix_std = mixture_stft_expanded.std(dim=(-2, -1), keepdim=True)

        # 3. Scale and Shift the noise to match mixture statistics
        # Formula: Noise_Final = (Noise_Raw * Mixture_Std) + Mixture_Mean
        # We also apply self.noise_scale to control how strong this "matched" noise is relative to the signal
        matched_noise = (raw_noise * mix_std * self.noise_scale) + mix_mean

        # x_0: noisy mixture
        x_0 = mixture_stft_expanded + matched_noise
        # --- KEY CHANGE END ---

        # x_1: target sources
        x_1 = targets_stft

        # Gaussian path noise
        z = torch.randn_like(targets_stft)

        # Interpolant: x_t = (1-t)*x_0 + t*x_1 + sigma*z
        x_t = (1 - t) * x_0 + t * x_1 + self.sigma * z

        # Target velocity for OT-CFM
        target_velocity = x_1 - x_0

        return x_t, mixture_stft, target_velocity, t

    def training_step(self, batch, batch_idx):
        x_t, mixture_stft, target_velocity, t = self.to_flow_matching(batch)

        predicted_velocity, loss = self.model(
            x_t,
            mixture=mixture_stft,
            targets=target_velocity,
            t=t
        )

        validation_output = self.validation_loss_step()
        self._debug_step()

        self.log("train/t_mean", t.mean(), prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)

        mixture_waveform = self.inverse_stft(mixture_stft)
        mixture_waveform = rearrange(mixture_waveform, "(b n) c t -> b n c t", n=self.num_instruments)

        target_velocity_waveform = self.inverse_stft(target_velocity)
        target_velocity_waveform = rearrange(target_velocity_waveform, "(b n) c t -> b n c t", n=self.num_instruments)

        predicted_velocity_waveform = self.inverse_stft(predicted_velocity)
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

    @torch.no_grad()
    def iterative_inference(self, mixture: torch.Tensor, steps: int = None) -> torch.Tensor:
        """Euler integration from t=0 to t=1 with Matched Noise."""
        if steps is None:
            steps = self.inference_steps

        b, c, t_audio = mixture.shape
        original_length = t_audio

        mixture_stft = self.stft(mixture.float())

        # Start from x_0: noisy mixture
        mixture_stft_expanded = repeat(mixture_stft, "b c f t -> (b n) c f t", n=self.num_instruments)

        # --- KEY CHANGE START ---
        # 1. Calculate stats exactly like in training
        mix_mean = mixture_stft_expanded.mean(dim=(-2, -1), keepdim=True)
        mix_std = mixture_stft_expanded.std(dim=(-2, -1), keepdim=True)

        # 2. Generate matched noise
        raw_noise = torch.randn_like(mixture_stft_expanded)
        matched_noise = (raw_noise * mix_std * self.noise_scale) + mix_mean

        # 3. Initialize x_0
        # Note: I removed the "(1 - self.noise_scale)" term from your snippet because
        # your training code uses simple addition: x_0 = mixture + noise
        x = mixture_stft_expanded + matched_noise
        # --- KEY CHANGE END ---

        visualize = self.model.visualize("thing")
        dt = 1.0 / steps

        # SDE injection scale (keep this small if you want deterministic results)
        sde_noise_scale = 0.05

        for step in range(steps):
            # Broadcast t to the batch shape
            t_val = torch.full((x.shape[0], 1, 1, 1), step * dt, device=mixture.device)

            predicted_velocity, _ = self.model(x, mixture=mixture_stft, t=t_val)

            # Optional: SDE Langevin noise injection (stochastic sampler)
            # This helps "shake" the model out of local errors, but isn't strictly necessary for OT-CFM.
            # If the results are fuzzy, try setting sde_noise_scale to 0.0.
            noise = torch.randn_like(x) * sde_noise_scale * (1 - step * dt) * (dt ** 0.5)

            # Euler Step
            x = x + predicted_velocity * dt + noise

            # self.model.is_debug = True
            # visualize(x)
            # self.model.is_debug = False

        x = self.inverse_stft(x, original_length)
        x = rearrange(x, "(b n) c t -> b n c t", n=self.num_instruments)

        return x

    def on_train_start(self):
        pass

    def _debug_step(self):
        debug_every = self.config.logging.debug_every
        if (self.debug_mixture is None) or (self.debug_targets is None):
            self.debug_mixture, self.debug_targets = next(self.validation_iter)
            self.debug_mixture = rearrange(self.debug_mixture, "c t -> 1 c t").to(self.device)
            self.debug_targets = rearrange(self.debug_targets, "n c t -> 1 n c t", n=self.num_instruments).to(
                self.device)

        if debug_every != 0 and self.global_step % debug_every == 0 and self.global_step != 0:
            self.model.eval()
            with torch.no_grad():
                debug_separated = self.iterative_inference(self.debug_mixture, steps=self.config.debug_steps)
                self.model.debug_forward(self.debug_mixture, debug_separated, self.debug_targets)
            self.model.train()

    def validation_loss_step(self):
        validate_every = self.config.logging.validate_every
        if validate_every != 0 and self.global_step % validate_every == 0:
            val_mixture, val_targets = next(self.validation_iter)
            val_mixture = rearrange(val_mixture, "c t -> 1 c t").to(self.device)
            val_targets = rearrange(val_targets, "n c t -> 1 n c t", n=self.num_instruments).to(self.device)
            self.model.eval()

            val_separated = self.iterative_inference(val_mixture, steps=1)
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
            targets=self.config.training.target_sources
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
