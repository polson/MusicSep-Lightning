import random
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import IterableDataset

from dataset.augmentor import AudioAugmentor, MixupParams, LoudnessParams, ShuffleParams, FlipSignParams, \
    TimeShiftParams


@dataclass
class WavMetadata:
    path: Path
    num_samples: int
    num_channels: int
    mmap: np.memmap  # Pre-mapped file data


@dataclass
class AudioChunk:
    file: WavMetadata
    start_sample: int


@dataclass
class Mixture:
    audios: Dict[str, torch.Tensor]


class TrainDataset(IterableDataset):
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
        self.root_dir = Path(root_dir)
        self.chunk_length = int(duration_seconds * self.sample_rate)
        self.targets = targets
        self.aligned_mixture = aligned_mixture
        self.max_retries = max_retries

        self.wav_groups = self._group_wavs_by_file_stem()
        self.song_groups = self._group_wavs_by_song()
        self.wav_weights = self._compute_sampling_weights()

        self.augmentor = AudioAugmentor(
            mixup=MixupParams(prob=0.0),
            loudness=LoudnessParams(prob=1.0, min=0.75, max=1.25),
            shuffle=ShuffleParams(prob=1.0),
            flip_sign=FlipSignParams(prob=1.0),
            time_shift=TimeShiftParams(prob=0.0),
        )

        self.workers_started = False

    def _create_memmap(self, path: Path) -> Tuple[np.memmap, int, int]:
        """Create a memory-mapped array for a WAV file, skipping the header."""
        with wave.open(str(path), 'rb') as wf:
            num_channels = wf.getnchannels()
            num_samples = wf.getnframes()
            sample_width = wf.getsampwidth()
            if sample_width != 2:
                raise ValueError(f"Expected 16-bit PCM, got {sample_width * 8}-bit")

        # WAV header is typically 44 bytes for standard PCM
        # But we need to find the actual data chunk offset
        header_offset = self._find_data_chunk_offset(path)

        # Memory-map the raw PCM data as int16
        mmap = np.memmap(
            path,
            dtype=np.int16,
            mode='r',
            offset=header_offset,
            shape=(num_samples, num_channels) if num_channels > 1 else (num_samples,)
        )
        return mmap, num_samples, num_channels

    def _find_data_chunk_offset(self, path: Path) -> int:
        """Find the byte offset where the 'data' chunk starts in a WAV file."""
        with open(path, 'rb') as f:
            f.seek(12)  # Skip RIFF header
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError(f"Could not find data chunk in {path}")
                chunk_size = int.from_bytes(f.read(4), 'little')
                if chunk_id == b'data':
                    return f.tell()
                f.seek(chunk_size, 1)  # Skip this chunk

    def _compute_sampling_weights(self) -> Dict[str, List[float]]:
        weights = {}
        for wav_type, wav_list in self.wav_groups.items():
            type_weights = []
            for wav in wav_list:
                usable = max(0, wav.num_samples - self.chunk_length + 1)
                type_weights.append(float(usable))

            total = sum(type_weights)
            if total == 0:
                type_weights = [1.0] * len(wav_list)

            weights[wav_type] = type_weights
        return weights

    def _weighted_random_choice(self, wav_type: str) -> WavMetadata:
        wav_list = self.wav_groups[wav_type]
        weights = self.wav_weights[wav_type]
        return random.choices(wav_list, weights=weights, k=1)[0]

    def __iter__(self):
        while True:
            try:
                mixture = self._create_mixture()
                if mixture is None:
                    continue

                if self.augmentor.mixup.prob > 0.0:
                    mixture2 = self._create_mixture()
                    if mixture2 is not None:
                        mixture = self.augmentor.augment(mixture, mixture2)
                    else:
                        mixture = self.augmentor.augment(mixture)
                else:
                    mixture = self.augmentor.augment(mixture)

                mixture_audio, targets_tensor = self._get_mixture_and_targets_tensor(mixture)
                yield mixture_audio, targets_tensor
            except Exception:
                continue

    def _group_wavs_by_file_stem(self) -> Dict[str, List[WavMetadata]]:
        wav_groups = defaultdict(list)
        for path in self.root_dir.rglob("*"):
            if path.suffix.lower() != '.wav':
                continue
            if path.name == "mixture.wav":
                continue
            base_name = path.stem
            mmap, num_samples, num_channels = self._create_memmap(path)
            wav_groups[base_name].append(
                WavMetadata(
                    path=path,
                    num_samples=num_samples,
                    num_channels=num_channels,
                    mmap=mmap
                )
            )
        return wav_groups

    def _group_wavs_by_song(self) -> Dict[Path, Dict[str, WavMetadata]]:
        """Build song groups from the already-loaded wav_groups to reuse memmaps."""
        song_groups = defaultdict(dict)
        for wav_type, wav_list in self.wav_groups.items():
            for wav_meta in wav_list:
                song_dir = wav_meta.path.parent
                song_groups[song_dir][wav_type] = wav_meta
        return {k: v for k, v in song_groups.items() if v}

    def _create_mixture(self) -> Optional[Mixture]:
        if self.aligned_mixture:
            return self._create_aligned_mixture()
        else:
            return self._try_create_random_mixture()

    def _create_aligned_mixture(self) -> Mixture | None:
        if not self.song_groups:
            return None

        for _ in range(self.max_retries):
            song_path = random.choice(list(self.song_groups.keys()))
            song_tracks = self.song_groups[song_path]

            if not song_tracks:
                continue

            min_num_samples = min(track.num_samples for track in song_tracks.values())
            if min_num_samples < self.chunk_length:
                continue

            for _ in range(5):
                start_sample = random.randint(0, min_num_samples - self.chunk_length)
                temp_audios = {}
                failed = False

                try:
                    for wav_type, wav_meta in song_tracks.items():
                        audio = self._read_chunk_from_memmap(wav_meta, start_sample)
                        if self._is_chunk_silent(audio):
                            failed = True
                            break
                        temp_audios[wav_type] = self._audio_to_tensor(audio)
                except Exception:
                    failed = True

                if not failed:
                    return Mixture(audios=temp_audios)

        return None

    def _try_create_random_mixture(self) -> Mixture | None:
        audios = {}

        for wav_type in self.wav_groups.keys():
            valid_chunk_found = False

            for _ in range(self.max_retries):
                wav = self._weighted_random_choice(wav_type)
                start_sample = random.randint(0, max(0, wav.num_samples - self.chunk_length))

                try:
                    audio = self._read_chunk_from_memmap(wav, start_sample)

                    if not self._is_chunk_silent(audio):
                        audios[wav_type] = self._audio_to_tensor(audio)
                        valid_chunk_found = True
                        break
                except Exception:
                    continue

            if not valid_chunk_found:
                return None

        return Mixture(audios=audios)

    def _read_chunk_from_memmap(self, wav_meta: WavMetadata, start_sample: int) -> np.ndarray:
        """Read a chunk directly from the memory-mapped array."""
        end_sample = start_sample + self.chunk_length

        if wav_meta.num_channels > 1:
            # Shape is (num_samples, num_channels)
            chunk = wav_meta.mmap[start_sample:end_sample, :]
            # Convert to float32 and transpose to (channels, samples)
            data = chunk.astype(np.float32) / 32768.0
            data = data.T
        else:
            chunk = wav_meta.mmap[start_sample:end_sample]
            data = chunk.astype(np.float32) / 32768.0

        # Pad if needed
        if data.shape[-1] < self.chunk_length:
            if wav_meta.num_channels > 1:
                pad_width = ((0, 0), (0, self.chunk_length - data.shape[1]))
            else:
                pad_width = (0, self.chunk_length - len(data))
            data = np.pad(data, pad_width, mode='constant')

        return data

    def _is_chunk_silent(self, data: np.ndarray) -> bool:
        return np.max(np.abs(data)) < 0.1

    def _audio_to_tensor(self, audio: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(audio.copy()).float().contiguous()

    def _get_mixture_and_targets_tensor(self, mixture: Mixture) -> Tuple[torch.Tensor, torch.Tensor]:
        base_tensor = torch.zeros(2, self.chunk_length, dtype=torch.float32)

        present_audios = list(mixture.audios.values())
        mixture_audio = torch.stack(present_audios).sum(dim=0) if present_audios else base_tensor.clone()

        target_audios = []
        for target_name in self.targets:
            if target_name == "mixture":
                target_audios.append(mixture_audio)
            elif target_name in mixture.audios:
                target_audios.append(mixture.audios[target_name])
            else:
                target_audios.append(base_tensor.clone())

        targets_tensor = torch.stack(target_audios).flatten(0, 1)  # (n, c, t) -> (n*c, t)
        return mixture_audio, targets_tensor
