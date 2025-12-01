import torch
import numpy as np
from torch.utils.data import IterableDataset
from typing import List, Tuple


class ChirpDataset(IterableDataset):
    def __init__(
            self,
            root_dir: str,
            duration_seconds: float,
            targets: List[str],
            aligned_mixture: bool = False,
            max_retries: int = 50
    ):
        super().__init__()
        self.sample_rate = 44100
        self.chunk_length = int(duration_seconds * self.sample_rate)
        self.targets = targets

        # Generate the chirp once
        self.chirp = self._generate_chirp()

    def _generate_chirp(self) -> torch.Tensor:
        """Generate a stereo chirp signal."""
        t = np.linspace(0, self.chunk_length / self.sample_rate, self.chunk_length, dtype=np.float32)

        # Chirp from 20Hz to 8000Hz
        f0, f1 = 20.0, 8000.0
        chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * self.chunk_length / self.sample_rate)))
        chirp = chirp.astype(np.float32) * 0.5  # Scale to avoid clipping

        # Make stereo (2, chunk_length)
        stereo_chirp = np.stack([chirp, chirp], axis=0)
        return torch.from_numpy(stereo_chirp).float().contiguous()

    def __iter__(self):
        while True:
            mixture_audio, targets_tensor = self._get_mixture_and_targets_tensor()
            yield mixture_audio, targets_tensor

    def _get_mixture_and_targets_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return chirp as both mixture and target."""
        mixture_audio = self.chirp.clone()

        # Stack targets (one chirp per target)
        target_audios = [self.chirp.clone() for _ in self.targets]
        targets_tensor = torch.stack(target_audios)

        return mixture_audio, targets_tensor
