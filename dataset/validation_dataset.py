import os
import torch
from pathlib import Path
from glob import glob


class TestSongDataset(torch.utils.data.Dataset):
    def __init__(self, test_dir, target_sources):
        self.test_dir = Path(test_dir)
        self.target_sources = target_sources
        wav_files = glob(os.path.join(test_dir, "**/mixture.wav"), recursive=True)
        flac_files = glob(os.path.join(test_dir, "**/mixture.flac"), recursive=True)
        self.mixture_files = wav_files + flac_files
        self.mixture_files.sort()
        print(f"Found {len(self.mixture_files)} test songs")

    def __len__(self):
        return len(self.mixture_files)

    def __getitem__(self, idx):
        mixture_path = self.mixture_files[idx]
        extension = Path(mixture_path).suffix.lower()
        target_paths = [
            str(Path(mixture_path).parent / f"{source}{extension}") for source in self.target_sources
        ]

        return {
            "mixture_path": mixture_path,
            "target_paths": target_paths
        }
