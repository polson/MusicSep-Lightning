import math

import torch
import torch.nn as nn


class RFFTModule(nn.Module):
    def __init__(self, norm="ortho"):
        super(RFFTModule, self).__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Input tensor must have 3 dimensions (batch, channels, time), but got {x.ndim}")

        x_processed = x.float()
        if x.dtype == torch.bfloat16:
            x_processed = x.to(torch.float32)

        complex_rfft = torch.fft.rfft(x_processed, dim=-1, norm=self.norm)

        real_part = complex_rfft.real
        imag_part = complex_rfft.imag

        output = torch.cat((real_part, imag_part), dim=1)
        return output

    def inverse(self, y: torch.Tensor, original_time_dim: int) -> torch.Tensor:
        if y.ndim != 3:
            raise ValueError(f"Input tensor y must have 3 dimensions (batch, 2*channels, n_freq), but got {y.ndim}")

        if y.shape[1] % 2 != 0:
            raise ValueError(
                f"The channel dimension (dim=1) of input y must be even (2*channels), but got {y.shape[1]}")

        y = y.float()
        num_original_channels = y.shape[1] // 2

        real_part = y[:, :num_original_channels, :]
        imag_part = y[:, num_original_channels:, :]

        complex_input_to_irfft = torch.complex(real_part, imag_part)

        reconstructed_signal = torch.fft.irfft(complex_input_to_irfft, n=original_time_dim, dim=-1, norm=self.norm)

        return reconstructed_signal


class RFFT2DModule(nn.Module):
    def __init__(self, norm="ortho"):
        super(RFFT2DModule, self).__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (batch, channels, freq, time), but got {x.ndim}")

        x_processed = x.float()

        if x_processed.is_complex():
            print(
                "Warning: Input to RFFT2DModule is complex. rfft2 expects real input. Behavior might be unexpected or error.")

        complex_rfft2 = torch.fft.rfft2(x_processed, dim=(-2, -1), norm=self.norm)

        real_part = complex_rfft2.real
        imag_part = complex_rfft2.imag

        output = torch.cat((real_part, imag_part), dim=1)
        return output

    def inverse(self, y: torch.Tensor, original_freq_dim: int, original_time_dim: int) -> torch.Tensor:
        if y.ndim != 4:
            raise ValueError(
                f"Input tensor y must have 4 dimensions (batch, 2*channels, n_freq, n_time), but got {y.ndim}")

        if y.shape[1] % 2 != 0:
            raise ValueError(
                f"The channel dimension (dim=1) of input y must be even (2*channels), but got {y.shape[1]}")

        y = y.float()
        num_original_channels = y.shape[1] // 2

        real_part = y[:, :num_original_channels, :, :]
        imag_part = y[:, num_original_channels:, :, :]

        complex_input_to_irfft2 = torch.complex(real_part, imag_part)

        reconstructed_signal = torch.fft.irfft2(
            complex_input_to_irfft2,
            s=(original_freq_dim, original_time_dim),
            dim=(-2, -1),
            norm=self.norm
        )

        return reconstructed_signal
