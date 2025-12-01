import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm


class SpectrogramNormalizer(nn.Module):
    """Normalizes spectrograms to match N(0,1) noise distribution for Flow Matching."""

    def __init__(self, mean: torch.Tensor = None, std: torch.Tensor = None, freq_wise: bool = False):
        super().__init__()
        self.freq_wise = freq_wise

        # Initialize with defaults if not provided
        if mean is None:
            mean = torch.tensor(0.0)
        if std is None:
            std = torch.tensor(1.0)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize spectrogram: (x - mean) / std"""
        return (x - self.mean) / (self.std + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize spectrogram: x * std + mean"""
        return x * self.std + self.mean

    @classmethod
    def compute_from_dataset(
            cls,
            dataset,
            stft_fn,
            num_samples: int = 1000,
            freq_wise: bool = False,
            device: str = 'cuda'
    ) -> 'SpectrogramNormalizer':
        """Compute normalization statistics from dataset.

        Args:
            dataset: Training dataset yielding (mixture, targets) tuples
            stft_fn: STFT transform function
            num_samples: Number of samples to use for statistics
            freq_wise: If True, compute per-frequency-bin statistics
            device: Device for computation
        """
        print(f"Computing spectrogram statistics from {num_samples} samples...")

        # Online mean/variance computation (Welford's algorithm)
        count = 0
        mean = None
        m2 = None  # Sum of squared differences

        stft_fn = stft_fn.to(device)

        for i, (mixture, targets) in enumerate(tqdm(dataset, total=num_samples)):
            if i >= num_samples:
                break

            # targets: (n_instruments, channels, time)
            targets = targets.to(device)
            n, c, t = targets.shape

            # STFT expects (batch, channels, time) with channels=2 for stereo
            targets_stft = stft_fn(targets.float())  # (n, c, F, T)

            if freq_wise:
                # Compute stats per frequency bin: reduce over batch, channels, and time
                # Shape for stats: (1, 1, F, 1)
                sample_mean = targets_stft.mean(dim=(0, 1, 3), keepdim=True)  # (1, 1, F, 1)
                sample_var = targets_stft.var(dim=(0, 1, 3), keepdim=True)
                n_elements = targets_stft.shape[0] * targets_stft.shape[1] * targets_stft.shape[3]
            else:
                # Global stats (single scalar)
                sample_mean = targets_stft.mean()
                sample_var = targets_stft.var()
                n_elements = targets_stft.numel()

            # Welford's online algorithm
            if mean is None:
                mean = sample_mean
                m2 = sample_var * n_elements
                count = n_elements
            else:
                delta = sample_mean - mean
                count += n_elements
                mean = mean + delta * (n_elements / count)
                # For variance, we're approximating with batch variances
                m2 = m2 + sample_var * n_elements

        std = torch.sqrt(m2 / count)

        # Ensure minimum std to avoid division issues
        std = torch.clamp(std, min=1e-6)

        print(f"Computed statistics:")
        print(f"  Mean: {mean.mean().item():.4f} (shape: {mean.shape})")
        print(f"  Std:  {std.mean().item():.4f} (shape: {std.shape})")

        normalizer = cls(mean=mean.cpu(), std=std.cpu(), freq_wise=freq_wise)
        return normalizer

    def save(self, path: Path):
        """Save normalizer state to file."""
        torch.save({
            'mean': self.mean,
            'std': self.std,
            'freq_wise': self.freq_wise
        }, path)
        print(f"Saved normalizer to {path}")

    @classmethod
    def load(cls, path: Path) -> 'SpectrogramNormalizer':
        """Load normalizer from file."""
        state = torch.load(path, weights_only=True)
        normalizer = cls(
            mean=state['mean'],
            std=state['std'],
            freq_wise=state['freq_wise']
        )
        print(f"Loaded normalizer from {path}")
        return normalizer
