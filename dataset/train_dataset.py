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

# Assuming these imports exist in your project structure
from dataset.augmentor import AudioAugmentor, MixupParams, LoudnessParams, ShuffleParams, FlipSignParams, \
    TimeShiftParams


@dataclass
class WavMetadata:
    path: Path
    num_samples: int


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

        # Precompute sampling weights for each wav group
        self.wav_weights = self._compute_sampling_weights()

        self.augmentor = AudioAugmentor(
            mixup=MixupParams(prob=0.0),
            loudness=LoudnessParams(prob=1.0, min=0.75, max=1.25),
            shuffle=ShuffleParams(prob=1.0),
            flip_sign=FlipSignParams(prob=1.0),
            time_shift=TimeShiftParams(prob=0.0),
        )

        self.workers_started = False

    def _compute_sampling_weights(self) -> Dict[str, List[float]]:
        """
        Compute sampling weights for each wav file based on usable length.
        Weight = max(0, num_samples - chunk_length + 1), which represents
        the number of valid start positions for a chunk.
        """
        weights = {}
        for wav_type, wav_list in self.wav_groups.items():
            type_weights = []
            for wav in wav_list:
                # Usable length: how many valid start positions exist
                usable = max(0, wav.num_samples - self.chunk_length + 1)
                type_weights.append(float(usable))

            # Handle edge case where all weights are zero
            total = sum(type_weights)
            if total == 0:
                # Fall back to uniform sampling
                type_weights = [1.0] * len(wav_list)

            weights[wav_type] = type_weights

        return weights

    def _weighted_random_choice(self, wav_type: str) -> WavMetadata:
        """Select a wav file with probability proportional to its usable length."""
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
            except Exception as e:
                continue

    def _group_wavs_by_file_stem(self) -> Dict[str, List[WavMetadata]]:
        wav_groups = defaultdict(list)
        for path in self.root_dir.rglob("*"):
            if path.suffix.lower() != '.wav':
                continue
            if path.name == "mixture.wav":
                continue
            base_name = path.stem
            wav_groups[base_name].append(
                WavMetadata(
                    path=path,
                    num_samples=sf.info(path).frames
                )
            )
        return wav_groups

    def _group_wavs_by_song(self) -> Dict[Path, Dict[str, WavMetadata]]:
        song_groups = defaultdict(dict)
        for path in self.root_dir.rglob("*"):
            if path.suffix.lower() != '.wav':
                continue
            if path.name == "mixture.wav":
                continue

            instrument_type = path.stem
            song_dir = path.parent
            song_groups[song_dir][instrument_type] = WavMetadata(
                path=path,
                num_samples=sf.info(path).frames
            )
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

            valid_timestamp_found = False
            audios = {}

            for _ in range(5):
                start_sample = random.randint(0, min_num_samples - self.chunk_length)

                temp_audios = {}
                failed_silence_check = False

                try:
                    for wav_type, wav_metadata in song_tracks.items():
                        audio = self._read_wav_chunk(
                            file_path=str(wav_metadata.path),
                            chunk_duration=self.chunk_length,
                            start_position=start_sample
                        )
                        if self._is_chunk_silent(audio):
                            failed_silence_check = True
                            break

                        audio = self._audio_to_tensor(audio)
                        temp_audios[wav_type] = audio
                except Exception:
                    failed_silence_check = True

                if not failed_silence_check:
                    audios = temp_audios
                    valid_timestamp_found = True
                    break

            if valid_timestamp_found and audios:
                return Mixture(audios=audios)

        return None

    def _try_create_random_mixture(self) -> Mixture | None:
        audios = {}

        for wav_type in self.wav_groups.keys():
            valid_chunk_found = False

            for _ in range(self.max_retries):
                # Use weighted selection instead of uniform random
                wav = self._weighted_random_choice(wav_type)
                start_sample = random.randint(0, max(0, wav.num_samples - self.chunk_length))

                try:
                    audio = self._read_wav_chunk(
                        file_path=str(wav.path),
                        chunk_duration=self.chunk_length,
                        start_position=start_sample
                    )

                    if not self._is_chunk_silent(audio):
                        audio_tensor = self._audio_to_tensor(audio)
                        audios[wav_type] = audio_tensor
                        valid_chunk_found = True
                        break
                except Exception as e:
                    continue

            if not valid_chunk_found:
                return None

        return Mixture(audios=audios)

    def _is_chunk_silent(self, data):
        max_peak = np.max(np.abs(data))
        return max_peak < 0.1

    def _audio_to_tensor(self, audio):
        return torch.from_numpy(audio.copy()).float().contiguous()

    def _get_mixture_and_targets_tensor(self, mixture: Mixture) -> Tuple[torch.Tensor, torch.Tensor]:
        base_tensor = torch.zeros(2, self.chunk_length, dtype=torch.float32)

        present_audios = list(mixture.audios.values())
        mixture_audio = torch.stack(present_audios).sum(dim=0) if present_audios else base_tensor.clone()

        target_audios = []
        for target_name in self.targets:
            if target_name == "mixture":
                target_audios.append(mixture_audio)
            else:
                if target_name in mixture.audios:
                    target_audios.append(mixture.audios[target_name])
                else:
                    target_audios.append(base_tensor.clone())

        targets_tensor = torch.stack(target_audios)
        return mixture_audio, targets_tensor

    def _read_wav_chunk(self, file_path, chunk_duration=44100, start_position=0):
        with wave.open(file_path, 'rb') as wav_file:
            total_frames = wav_file.getnframes()
            if wav_file.getsampwidth() != 2:
                raise ValueError(f"Expected 16-bit PCM, got {wav_file.getsampwidth()}")

            start_frame = min(start_position, total_frames - 1) if start_position < total_frames else 0
            chunk_frames = min(chunk_duration, total_frames - start_frame)

            wav_file.setpos(start_frame)
            raw_data = wav_file.readframes(chunk_frames)

            data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

            if len(data) < chunk_duration:
                padding = np.zeros(chunk_duration - len(data), dtype=np.float32)
                data = np.concatenate((data, padding))

            if wav_file.getnchannels() > 1:
                data = data.reshape(-1, wav_file.getnchannels()).T
            else:
                pass

        return data
