from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class Mixture:
    audios: Dict[str, torch.Tensor]


@dataclass
class LoudnessParams:
    prob: float
    min: float = 0.5
    max: float = 1.5


@dataclass
class MixupParams:
    prob: float


@dataclass
class ShuffleParams:
    prob: float


@dataclass
class FlipSignParams:
    prob: float


@dataclass
class TimeShiftParams:
    prob: float
    max_shift_samples: int = 1000


import random


class AudioAugmentor:

    def __init__(
            self,
            loudness: LoudnessParams = None,
            mixup: MixupParams = None,
            shuffle: ShuffleParams = None,
            flip_sign: FlipSignParams = None,
            time_shift: TimeShiftParams = None,
    ):
        self.loudness = loudness
        self.mixup = mixup
        self.shuffle = shuffle
        self.flip_sign = flip_sign
        self.time_shift = time_shift

    def augment(self, mixture: Mixture, mixture2: Mixture = None) -> Mixture:
        if self.loudness and random.random() < self.loudness.prob:
            mixture = self._apply_loudness(mixture)
            mixture2 = self._apply_loudness(mixture2) if mixture2 else None

        if self.shuffle and random.random() < self.shuffle.prob:
            mixture = self._shuffle_channels(mixture)
            mixture2 = self._shuffle_channels(mixture2) if mixture2 else None

        if self.flip_sign and random.random() < self.flip_sign.prob:
            mixture = self._flip_sign(mixture)
            mixture2 = self._flip_sign(mixture2) if mixture2 else None

        if self.time_shift and random.random() < self.time_shift.prob:
            mixture = self._time_shift(mixture)
            mixture2 = self._time_shift(mixture2) if mixture2 else None

        if self.mixup and random.random() < self.mixup.prob:
            mixture = self.combine_mixtures(mixture, mixture2) if mixture2 else mixture

        return mixture

    def _apply_loudness(self, mixture: Mixture) -> Mixture:
        def adjust_audio_loudness(audio: torch.Tensor) -> torch.Tensor:
            loudness = random.uniform(self.loudness.min, self.loudness.max)
            adjusted_audio = audio * loudness
            return self._apply_limiter(adjusted_audio)

        audios = {
            type: adjust_audio_loudness(audio)
            for type, audio in mixture.audios.items()
        }
        return Mixture(audios)

    def _apply_limiter(self, audio: torch.Tensor, limit_threshold: float = 0.95) -> torch.Tensor:
        max_peak = torch.max(torch.abs(audio))
        if max_peak > limit_threshold:
            gain_reduction_factor = limit_threshold / max_peak
            audio *= gain_reduction_factor
        return audio

    def _shuffle_channels(self, mixture: Mixture) -> Mixture:
        def do_shuffle(audio: torch.Tensor) -> torch.Tensor:
            shuffled_indices = torch.randperm(audio.shape[0])
            audio = audio[shuffled_indices]
            return audio

        audios = {
            type: do_shuffle(audio)
            for type, audio in mixture.audios.items()
        }
        return Mixture(audios)

    def _flip_sign(self, mixture: Mixture) -> Mixture:
        def do_flip_sign(audio: torch.Tensor) -> torch.Tensor:
            signs = torch.randint(2, (audio.shape[0], 1), device=audio.device, dtype=torch.float32)
            signs = 2 * signs - 1
            return audio * signs

        audios = {
            type: do_flip_sign(audio)
            for type, audio in mixture.audios.items()
        }
        return Mixture(audios)

    def _time_shift(self, mixture: Mixture) -> Mixture:
        def do_time_shift(audio: torch.Tensor) -> torch.Tensor:
            shift = random.randint(-self.time_shift.max_shift_samples,
                                   self.time_shift.max_shift_samples)
            return torch.roll(audio, shifts=shift, dims=-1)

        audios = {
            type: do_time_shift(audio)
            for type, audio in mixture.audios.items()
        }
        return Mixture(audios)

    def combine_mixtures(self, mixture1: Mixture, mixture2: Mixture) -> Mixture:
        combined_audios = {}
        for key in mixture1.audios:
            combined_audios[key] = (mixture1.audios[key] * 0.5) + (mixture2.audios[key] * 0.5)
        return Mixture(audios=combined_audios)
