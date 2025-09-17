import torch
import torch.nn as nn
from einops import rearrange


class STFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.register_buffer('window', torch.hann_window(self.win_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        original_dtype = x.dtype
        x = rearrange(x, 'b c t -> (b c) t')
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=original_dtype, device=x.device),
            center=True,
            normalized=True,
            return_complex=True
        )
        x = torch.view_as_real(x)
        x = rearrange(x, '(b c) f t_prime two -> b (c two) f t_prime', b=b, c=c)
        x = x.to(original_dtype)
        return x

    def inverse(self, x: torch.Tensor, length: int | None = None) -> torch.Tensor:
        b, c_times_two, f, t_prime = x.shape
        c = c_times_two // 2
        x = rearrange(x, 'b (c two) f t_p -> (b c) f t_p two', c=c, two=2, t_p=t_prime)
        x = x.contiguous()
        x = x.to(torch.float32)
        x = torch.view_as_complex(x)
        window_float32 = self.window.to(dtype=torch.float32, device=x.device)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window_float32,
            center=True,
            normalized=True,
            length=length
        )
        x = rearrange(x, '(b c) t -> b c t', b=b, c=c)
        return x
