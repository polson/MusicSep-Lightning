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

        if x_processed.is_complex():
            print(
                "Warning: Input to RFFTModule is complex. rfft expects real input. Behavior might be unexpected or error.")

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

        # Apply 2D RFFT on the last two dimensions (freq, time)
        complex_rfft2 = torch.fft.rfft2(x_processed, dim=(-2, -1), norm=self.norm)

        real_part = complex_rfft2.real
        imag_part = complex_rfft2.imag

        # Concatenate along the channel dimension
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

        # Apply inverse 2D RFFT on the last two dimensions
        reconstructed_signal = torch.fft.irfft2(
            complex_input_to_irfft2,
            s=(original_freq_dim, original_time_dim),
            dim=(-2, -1),
            norm=self.norm
        )

        return reconstructed_signal


class DCT1D(nn.Module):
    def __init__(self, norm="ortho"):
        super().__init__()
        self.norm = norm

    def _dct_1d(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute 1D DCT using FFT
        """
        N = x.shape[dim]

        # Move the dimension to process to the last position
        x = x.transpose(dim, -1)

        # Prepare input for FFT
        # For DCT-II, we need to create a mirrored signal
        x_padded = torch.zeros(*x.shape[:-1], 2 * N, dtype=x.dtype, device=x.device)
        x_padded[..., :N] = x
        x_padded[..., N:] = torch.flip(x, dims=[-1])

        # Apply FFT
        X = torch.fft.fft(x_padded, dim=-1)

        # Extract DCT coefficients
        dct_coeff = X[..., :N].real

        # Apply normalization
        if self.norm == "ortho":
            dct_coeff[..., 0] *= math.sqrt(1 / (4 * N))
            dct_coeff[..., 1:] *= math.sqrt(1 / (2 * N))
        elif self.norm is None:
            pass  # No normalization

        # Move dimension back to original position
        dct_coeff = dct_coeff.transpose(dim, -1)

        return dct_coeff

    def _idct_1d(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute 1D inverse DCT using FFT
        """
        N = x.shape[dim]

        # Move the dimension to process to the last position
        x = x.transpose(dim, -1)

        # Apply inverse normalization
        x_scaled = x.clone()
        if self.norm == "ortho":
            x_scaled[..., 0] *= math.sqrt(4 * N)
            x_scaled[..., 1:] *= math.sqrt(2 * N)

        # Prepare for inverse FFT
        X = torch.zeros(*x.shape[:-1], 2 * N, dtype=torch.complex64, device=x.device)
        X[..., :N] = x_scaled.to(torch.complex64)
        X[..., N + 1:] = torch.flip(x_scaled[..., 1:], dims=[-1]).to(torch.complex64)

        # Apply inverse FFT
        x_reconstructed = torch.fft.ifft(X, dim=-1)

        # Extract real part and take first N samples
        result = x_reconstructed[..., :N].real

        # Move dimension back to original position
        result = result.transpose(dim, -1)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Input tensor must have 3 dimensions (batch, channels, time), but got {x.ndim}")

        x_processed = x.float()
        if x.dtype == torch.bfloat16:
            x_processed = x.to(torch.float32)

        if x_processed.is_complex():
            print("Warning: Input to ToDCT1D is complex. DCT expects real input. Taking real part.")
            x_processed = x_processed.real

        # Apply 1D DCT on the last dimension (time)
        dct_output = self._dct_1d(x_processed, dim=-1)

        return dct_output

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 3:
            raise ValueError(
                f"Input tensor y must have 3 dimensions (batch, channels, n_time), but got {y.ndim}")

        y = y.float()

        # Apply inverse 1D DCT on the last dimension
        reconstructed_signal = self._idct_1d(y, dim=-1)

        return reconstructed_signal


class DCT2DModule(nn.Module):
    def __init__(self, norm="ortho"):
        super(DCT2DModule, self).__init__()
        self.norm = norm

    def _dct_1d(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute 1D DCT using FFT
        """
        N = x.shape[dim]

        # Move the dimension to process to the last position
        x = x.transpose(dim, -1)

        # Prepare input for FFT
        # For DCT-II, we need to create a mirrored signal
        x_padded = torch.zeros(*x.shape[:-1], 2 * N, dtype=x.dtype, device=x.device)
        x_padded[..., :N] = x
        x_padded[..., N:] = torch.flip(x, dims=[-1])

        # Apply FFT
        X = torch.fft.fft(x_padded, dim=-1)

        # Extract DCT coefficients
        dct_coeff = X[..., :N].real

        # Apply normalization
        if self.norm == "ortho":
            dct_coeff[..., 0] *= math.sqrt(1 / (4 * N))
            dct_coeff[..., 1:] *= math.sqrt(1 / (2 * N))
        elif self.norm is None:
            pass  # No normalization

        # Move dimension back to original position
        dct_coeff = dct_coeff.transpose(dim, -1)

        return dct_coeff

    def _idct_1d(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute 1D inverse DCT using FFT
        """
        N = x.shape[dim]

        # Move the dimension to process to the last position
        x = x.transpose(dim, -1)

        # Apply inverse normalization
        x_scaled = x.clone()
        if self.norm == "ortho":
            x_scaled[..., 0] *= math.sqrt(4 * N)
            x_scaled[..., 1:] *= math.sqrt(2 * N)

        # Prepare for inverse FFT
        X = torch.zeros(*x.shape[:-1], 2 * N, dtype=torch.complex64, device=x.device)
        X[..., :N] = x_scaled.to(torch.complex64)
        X[..., N + 1:] = torch.flip(x_scaled[..., 1:], dims=[-1]).to(torch.complex64)

        # Apply inverse FFT
        x_reconstructed = torch.fft.ifft(X, dim=-1)

        # Extract real part and take first N samples
        result = x_reconstructed[..., :N].real

        # Move dimension back to original position
        result = result.transpose(dim, -1)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (batch, channels, freq, time), but got {x.ndim}")

        x_processed = x.float()
        if x.dtype == torch.bfloat16:
            x_processed = x.to(torch.float32)

        if x_processed.is_complex():
            print("Warning: Input to DCT2DModule is complex. DCT expects real input. Taking real part.")
            x_processed = x_processed.real

        # Apply 2D DCT on the last two dimensions (freq, time)
        # Apply DCT on frequency dimension first
        dct_freq = self._dct_1d(x_processed, dim=-2)
        # Then apply DCT on time dimension
        dct_output = self._dct_1d(dct_freq, dim=-1)

        return dct_output

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 4:
            raise ValueError(
                f"Input tensor y must have 4 dimensions (batch, channels, n_freq, n_time), but got {y.ndim}")

        y = y.float()

        # Apply inverse 2D DCT on the last two dimensions
        # Apply inverse DCT on time dimension first
        idct_time = self._idct_1d(y, dim=-1)
        # Then apply inverse DCT on frequency dimension
        reconstructed_signal = self._idct_1d(idct_time, dim=-2)

        return reconstructed_signal
