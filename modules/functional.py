import sys
from dataclasses import dataclass, replace
from typing import List

import torch
from einops import rearrange, parse_shape
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from modules.fft import RFFTModule, RFFT2DModule, DCT2DModule
from modules.seq import Seq
from modules.stft import STFT

import torch.nn.functional as F


@dataclass
class CFTShape:
    c: int  # channels
    f: int  # frequency
    t: int  # time


@dataclass
class CTShape:
    c: int  # channels
    t: int  # time


@dataclass
class CFShape:
    c: int  # channels
    f: int  # time


@dataclass
class TFShape:
    c: int  # channels
    f: int  # time


class DropBlock(nn.Module):
    def __init__(self, drop_prob=0.0, *args):
        super().__init__()
        self.fn = Seq(*args)
        self.drop_prob = drop_prob

    def __repr__(self):
        return f"DropBlock(drop_prob={self.drop_prob}, fn={self.fn})"

    def forward(self, x):
        """Determines whether to drop the block and runs it if not dropped."""
        if self.training and torch.rand(1).item() < self.drop_prob:
            return x
        return self.fn(x)


class Abs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


class Residual(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)

    def forward(self, x):
        return self.fn(x) + x


class SplitSum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Split channels in half: [b, c*2, ...] -> two tensors of [b, c, ...]
        c_total = x.shape[1]
        c_half = c_total // 2

        x1 = x[:, :c_half, ...]
        x2 = x[:, c_half:, ...]

        # Sum the two halves to get [b, c, ...]
        x_split_sum = x1 + x2

        return x_split_sum


class Plus(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __repr__(self):
        return f"Plus(value={self.value})"

    def forward(self, x):
        return x + self.value


class Freeze(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)
        for param in self.fn.parameters():
            param.requires_grad = False

    def __repr__(self):
        return f"Freeze(fn={self.fn})"

    def forward(self, x):
        return self.fn(x)


class Mask(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)

    def __repr__(self):
        return f"Mask(fn={self.fn})"

    def forward(self, x):
        return self.fn(x) * x


class MaskInstruments(nn.Module):
    def __init__(self, num_instruments, *args):
        super().__init__()
        self.fn = Seq(*args)
        self.num_instruments = num_instruments

    def __repr__(self):
        return f"MaskInstruments(num_instruments={self.num_instruments}, fn={self.fn})"

    def forward(self, x):
        return self.fn(x) * x.repeat(1, self.num_instruments, 1, 1)


class Film(nn.Module):
    def __init__(self, dim=1, *args):
        super().__init__()
        self.dim = dim
        self.fn = Seq(*args)
        self.silu = nn.SiLU()

    def __repr__(self):
        return f"Film(dim={self.dim}, fn={self.fn})"

    def forward(self, x):
        residual = x
        x = self.fn(x)
        mask, bias = torch.split(x, x.shape[self.dim] // 2, dim=self.dim)
        x = residual * torch.sigmoid(mask) + bias
        return x


class SwiGLU(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        self.silu = nn.SiLU()

    def __repr__(self):
        return f"SwiGLU(dim={self.dim})"

    def forward(self, x):
        gate, value = torch.split(x, x.shape[self.dim] // 2, dim=self.dim)
        return self.silu(gate) * value


class ForgetGate(nn.Module):
    def __init__(self, dim, *args):
        super().__init__()
        self.dim = dim
        self.fn = Seq(*args)

    def __repr__(self):
        return f"ForgetGate(dim={self.dim}, fn={self.fn})"

    def forward(self, x):
        residual = x
        x = self.fn(x)
        transform, gate = torch.split(x, x.shape[self.dim] // 2, dim=self.dim)
        gate = torch.sigmoid(gate)
        return gate * transform + (1 - gate) * residual


class Repeat(nn.Module):
    def __init__(self, num_repeats, *args):
        super().__init__()
        self.num_repeats = num_repeats
        self.modules_list = nn.ModuleList([Seq(*args) for _ in range(num_repeats)])

    def __repr__(self):
        return f"Repeat(num_repeats={self.num_repeats}, modules_list={self.modules_list})"

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


class RepeatWithArgs(nn.Module):
    def __init__(self, num_repeats, block):
        super().__init__()
        self.num_repeats = num_repeats
        self.block = block
        self.blocks = nn.ModuleList([
            block(i) for i in range(num_repeats)
        ])

    def __repr__(self):
        return f"RepeatWithArgs(num_repeats={self.num_repeats}, block={self.block}, blocks={self.blocks})"

    def forward(self, x):
        for i in range(self.num_repeats):
            x = self.blocks[i](x)
        return x


class RepeatWithArgsConcat(nn.Module):
    def __init__(self, num_repeats, block):
        super().__init__()
        self.num_repeats = num_repeats
        self.block = block
        self.blocks = nn.ModuleList([
            block(i) for i in range(num_repeats)
        ])

    def __repr__(self):
        return f"RepeatWithArgsConcat(num_repeats={self.num_repeats}, block={self.block}, blocks={self.blocks})"

    def forward(self, x):
        outputs = []
        for i in range(self.num_repeats):
            output_i = self.blocks[i](x)
            outputs.append(output_i)

        concatenated_output = torch.cat(outputs, dim=1)

        return concatenated_output


class RepeatDim(nn.Module):
    def __init__(self, dim, times):
        super().__init__()
        self.dim = dim
        self.times = times

    def __repr__(self):
        return f"RepeatDim(dim={self.dim}, times={self.times})"

    def forward(self, x):
        sizes = [1] * x.ndim
        sizes[self.dim] = self.times
        x = x.repeat(*sizes)
        return x


class PadBCFT(nn.Module):
    def __init__(self, freq_pad, time_pad, fn):
        super(PadBCFT, self).__init__()
        self.freq_pad = freq_pad
        self.time_pad = time_pad
        self.fn = fn

    def __repr__(self):
        return f"PadBCFT(freq_pad={self.freq_pad}, time_pad={self.time_pad}, fn={self.fn})"

    def forward(self, x):
        b, c, f, t = x.shape
        f_pad = self.freq_pad
        t_pad = self.time_pad
        padding = (t_pad, t_pad, f_pad, f_pad)
        x = nn.functional.pad(x, padding, mode='reflect')
        x = self.fn(x)
        if f_pad > 0 or t_pad > 0:
            x = x[:, :, f_pad:f_pad + f, t_pad:t_pad + t]
        return x


class PadBCFTNearestMultiple(nn.Module):
    def __init__(self, freq_multiple, time_multiple, fn):
        super(PadBCFTNearestMultiple, self).__init__()
        self.freq_multiple = freq_multiple
        self.time_multiple = time_multiple
        self.fn = fn()

    def __repr__(self):
        return f"PadBCFTNearestMultiple(freq_multiple={self.freq_multiple}, time_multiple={self.time_multiple}, fn={self.fn})"

    def forward(self, x):
        b, c, f, t = x.shape
        f_remainder = f % self.freq_multiple
        f_pad = (self.freq_multiple - f_remainder) % self.freq_multiple
        t_remainder = t % self.time_multiple
        t_pad = (self.time_multiple - t_remainder) % self.time_multiple
        padding = (0, t_pad, 0, f_pad)
        x = nn.functional.pad(x, padding, mode='constant', value=0)
        x = self.fn(x)
        if f_pad > 0 or t_pad > 0:
            x = x[:, :, :f, :t]

        return x


class PadBCTNearestMultiple(nn.Module):
    def __init__(self, time_multiple, *args):
        super(PadBCTNearestMultiple, self).__init__()
        self.time_multiple = time_multiple
        self.fn = Seq(*args)

    def __repr__(self):
        return f"PadBCTNearestMultiple(time_multiple={self.time_multiple}, fn={self.fn})"

    def forward(self, x):
        b, c, f, t = x.shape
        t_remainder = t % self.time_multiple
        t_pad = (self.time_multiple - t_remainder) % self.time_multiple
        padding = (0, t_pad, 0, 0)
        x = nn.functional.pad(x, padding, mode='constant', value=0)
        x = self.fn(x)
        if t_pad > 0:
            x = x[:, :, :, :t]

        return x


class SoftmaxGroups(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def __repr__(self):
        return f"SoftmaxGroups(num_groups={self.num_groups})"

    def forward(self, x):
        x = rearrange(x, 'b (n c) f t -> b n c f t', n=self.num_groups)
        x = torch.softmax(x, dim=1)
        x = rearrange(x, 'b n c f t -> b (n c) f t')
        x = x
        return x


class SoftmaxMask(nn.Module):
    def __init__(self, num_instruments, *args):
        super().__init__()
        self.softmax_groups = SoftmaxGroups(num_instruments)
        self.num_instruments = num_instruments
        self.fn = Seq(*args)

    def __repr__(self):
        return f"SoftmaxMask(num_instruments={self.num_instruments}, fn={self.fn})"

    def forward(self, x):
        mixture = x
        x = self.fn(x)
        x = self.softmax_groups(x)
        num_multiplies = x.shape[1] // mixture.shape[1]
        mixture = mixture.repeat(1, num_multiplies, 1, 1)
        x = x * mixture
        return x


class Scale(nn.Module):
    def __init__(self, scale, *args):
        super().__init__()
        self.fn = Seq(*args)
        self.scale = scale

    def __repr__(self):
        return f"Scale(scale={self.scale}, fn={self.fn})"

    def forward(self, x):
        return self.fn(x) * self.scale


class Concat(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def __repr__(self):
        return f"Concat(fns={self.fns})"

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return torch.cat(outputs, dim=1)


class Shrink(nn.Module):
    def __init__(self, shape, fn):
        super().__init__()
        self.shape = shape
        self.fn = fn
        self.thing = Seq(
            nn.Conv2d(shape.c, shape.c * 2, kernel_size=(2, 1), stride=(2, 1),
                      padding=0),
            fn(replace(shape, c=shape.c * 2)),
            nn.ConvTranspose2d(shape.c * 2, shape.c, kernel_size=(2, 1),
                               stride=(2, 1)),
        )

    def __repr__(self):
        return f"Shrink(shape={self.shape}, fn={self.fn}, thing={self.thing})"

    def forward(self, x):
        return self.thing(x)


class Bandsplit(nn.Module):
    def __init__(self, shape: CFTShape, num_splits, fn):
        super().__init__()
        self.shape = shape
        self.num_splits = num_splits
        self.fn = fn
        # Pass the function that will receive the full shape
        self.fn = fn(
            CFTShape(c=num_splits * shape.c, f=shape.f // num_splits, t=shape.t)  # Adjusted for num_splits
        )

    def __repr__(self):
        return f"Bandsplit(shape={self.shape}, num_splits={self.num_splits}, fn={self.fn})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape

        remainder = f % self.num_splits
        if remainder != 0:
            padding = self.num_splits - remainder
            x = nn.functional.pad(x, (0, 0, 0, padding))
            f = f + padding

        x = rearrange(x, 'b c (n f) t -> b (n c) f t', n=self.num_splits)

        # Pass the shape to the function
        x = self.fn(x)

        x = rearrange(x, 'b (n c) f t -> b c (n f) t', n=self.num_splits)

        if remainder != 0:
            x = x[:, :, :f - padding, :]
        return x


class TimeSplit(nn.Module):
    def __init__(self, shape: CFTShape, num_splits, fn):
        super().__init__()
        self.shape = shape
        self.num_splits = num_splits
        self.fn = fn
        # Pass the function that will receive the full shape
        self.fn = fn(
            CFTShape(c=num_splits * shape.c, f=shape.f, t=shape.t // num_splits)  # Adjusted for num_splits
        )

    def __repr__(self):
        return f"TimeSplit(shape={self.shape}, num_splits={self.num_splits}, fn={self.fn})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        remainder = t % self.num_splits
        if remainder != 0:
            padding = self.num_splits - remainder
            x = nn.functional.pad(x, (0, padding, 0, 0))
            t = t + padding

        x = rearrange(x, 'b c f (n t) -> b (n c) f t', n=self.num_splits)
        x = self.fn(x)
        x = rearrange(x, 'b (n c) f t -> b c f (n t)', n=self.num_splits)

        if remainder != 0:
            x = x[:, :, :, :t - padding]
        return x


class DebugShape(nn.Module):
    def __init__(self, name: str = None):
        super().__init__()
        self.fn = nn.Identity()
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fn(x)
        name_prefix = f"{self.name}: " if self.name else ""
        print(
            f"{name_prefix}Shape: {x.shape}, Min: {x.min().item():.4f}, Max: {x.max().item():.4f}, Mean: {x.mean().item():.4f}")
        return x


class Condition(nn.Module):
    def __init__(self, condition, true_fn, false_fn=None):
        super().__init__()
        self.condition = condition
        self.true_fn = true_fn()
        self.false_fn = false_fn() if false_fn is not None else nn.Identity()

    def __repr__(self):
        return f"Condition(condition={self.condition}, true_fn={self.true_fn}, false_fn={self.false_fn})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.condition(x):
            return self.true_fn(x)
        else:
            return self.false_fn(x)


class ReshapeBCFT(nn.Module):
    def __init__(self, reshape_to, *args):
        super().__init__()
        self.fn = Seq(*args)
        self.reshape_to = reshape_to

    def __repr__(self):
        return f"ReshapeBCFT(reshape_to='{self.reshape_to}', fn={self.fn})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store shape info
        shape_dict = dict(zip(['b', 'c', 'f', 't'], x.shape))

        # Transform and process
        x_intermediate = rearrange(x, f'b c f t -> {self.reshape_to}', **shape_dict)
        x_processed = self.fn(x_intermediate)

        # Restore using stored shape
        output = rearrange(x_processed, f'{self.reshape_to} -> b c f t', **shape_dict)
        return output


class ReshapeBCT(nn.Module):
    def __init__(self, reshape_to, *args):
        super().__init__()
        self.fn = Seq(*args)
        self.reshape_to = reshape_to

    def __repr__(self):
        return f"ReshapeBCT(reshape_to='{self.reshape_to}', fn={self.fn})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape_context = parse_shape(x, 'b c t')
        x_intermediate = rearrange(x, f'b c t -> {self.reshape_to}', **original_shape_context)
        x_processed = self.fn(x_intermediate)
        processed_shape_context = parse_shape(x_processed, self.reshape_to)
        output = rearrange(x_processed, f'{self.reshape_to} -> b c t', **processed_shape_context)
        return output


class Module(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __repr__(self):
        return f"Module(fn={self.fn})"

    def forward(self, x):
        return self.fn(x)


class SideEffect(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __repr__(self):
        return f"SideEffect(fn={self.fn})"

    def forward(self, x):
        self.fn(x)
        return x


class Sum(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)

    def __repr__(self):
        return f"Sum(fn={self.fn})"

    def forward(self, x):
        x = self.fn(x)
        x = torch.sum(x, dim=1)
        return x


class ToFFT(nn.Module):
    def __init__(self, channels, fn):
        super().__init__()
        self.channels = channels
        self.fn = fn(channels * 2)
        self.fft = RFFTModule()

    def __repr__(self):
        return f"ToFFT(channels={self.channels}, fn={self.fn})"

    def forward(self, x):
        original_time_dim = x.shape[-1]
        x = self.fft(x)
        x = self.fn(x)
        x = self.fft.inverse(x, original_time_dim)
        return x


class ComplexMask(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)

    def __repr__(self):
        return f"ComplexMask(fn={self.fn})"

    def forward(self, x):
        # x shape: (b, c, f, t) where c is even, with interleaved real/imag pairs
        b, c, f, t = x.shape
        assert c % 2 == 0, "Channel dimension must be even for real/imag pairs"

        # Store original dtype and convert to float32 for complex operations
        original_dtype = x.dtype
        x_float32 = x.float()

        # Reshape to separate real/imag pairs and convert to complex
        x_complex = rearrange(x_float32, 'b (ch ri) f t -> b ch f t ri', ri=2)
        residual = torch.view_as_complex(x_complex.contiguous())

        # Apply function to original tensor (keeping original dtype)
        x_processed = self.fn(x)

        # Convert processed tensor to float32 and reshape to complex
        x_processed_float32 = x_processed.float()
        x_processed_complex = rearrange(x_processed_float32, 'b (ch ri) f t -> b ch f t ri', ri=2)
        x_processed = torch.view_as_complex(x_processed_complex.contiguous())

        # Complex multiplication
        result = residual * x_processed

        # Convert back to real and reshape to original format
        result_real = torch.view_as_real(result)  # (b, ch, f, t, 2)
        result = rearrange(result_real, 'b ch f t ri -> b (ch ri) f t')

        # Convert back to original dtype
        result = result.to(original_dtype)

        return result


class ToFFT2d(nn.Module):
    def __init__(self, shape: CFTShape, fn):
        super().__init__()
        self.shape = shape
        self.fn = fn(replace(shape, c=shape.c * 2))
        self.fft = RFFT2DModule()

    def __repr__(self):
        return f"ToFFT2d(shape={self.shape}, fn={self.fn})"

    def forward(self, x):
        original_freq_dim = x.shape[2]
        original_time_dim = x.shape[3]
        x = self.fft(x)
        x = self.fn(x)
        x = self.fft.inverse(x, original_freq_dim, original_time_dim)
        return x


class ToFFT2dOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft = RFFT2DModule()

    def __repr__(self):
        return f"ToFFT1dOnly()"

    def forward(self, x):
        x = self.fft(x)
        return x


class ToFFT1dOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft = RFFTModule()

    def __repr__(self):
        return f"ToFFT1dOnly()"

    def forward(self, x):
        x = self.fft(x)
        return x


class ToDCT2d(nn.Module):
    def __init__(self, channels, fn):
        super().__init__()
        self.channels = channels
        self.fn = fn(channels)
        self.dct = DCT2DModule()

    def __repr__(self):
        return f"ToDCT2d(channels={self.channels}, fn={self.fn})"

    def forward(self, x):
        x = self.dct(x)
        x = self.fn(x)
        x = self.dct.inverse(x)
        return x


class STFTAndInverse(nn.Module):
    def __init__(self, in_channels=2, in_samples=44100, n_fft=4096, hop_length=1024, fn=nn.Identity()):
        super().__init__()
        self.in_channels = in_channels  # Store for validation
        self.in_samples = in_samples  # Store for validation
        self.out_f = n_fft // 2
        self.hop_length = hop_length
        out_shape = CFTShape(
            c=in_channels * 2,
            f=self.out_f,
            t=(in_samples + hop_length - 1) // hop_length
        )
        self.fn = fn(out_shape)
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)

    def __repr__(self):
        return f"STFTAndInverse(in_channels={self.in_channels}, in_samples={self.in_samples}, out_f={self.out_f}, hop_length={self.hop_length}, fn={self.fn})"

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input tensor has {x.shape[1]} channels, but model expects {self.in_channels} channels"
            )
        original_length = x.shape[-1]
        x = self.stft(x.float())
        x = x[:, :, :-1, :]
        x = self.fn(x)
        nyquist_bin = torch.zeros_like(x[:, :, :1, :])
        x = torch.cat([x, nyquist_bin], dim=2)
        x = self.stft.inverse(x, original_length)
        return x


class STFTOnly(nn.Module):
    def __init__(self, in_channels=2, in_samples=44100, n_fft=2048, hop_length=512):
        super().__init__()
        self.in_channels = in_channels
        self.in_samples = in_samples
        self.out_f = n_fft // 2
        self.hop_length = hop_length
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)

    def __repr__(self):
        return f"STFTOnly(in_channels={self.in_channels}, in_samples={self.in_samples}, out_f={self.out_f}, hop_length={self.hop_length})"

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input tensor has {x.shape[1]} channels, but model expects {self.in_channels} channels"
            )
        x = self.stft(x.float())
        return x


class Bias2d(nn.Module):
    def __init__(self, channels, max_f=2048, max_t=2048):
        super(Bias2d, self).__init__()
        self.channels = channels
        self.max_f = max_f
        self.max_t = max_t
        self.pos_embed_f = nn.Parameter(torch.zeros(1, channels, max_f, 1))
        self.pos_embed_t = nn.Parameter(torch.zeros(1, channels, 1, max_t))
        nn.init.trunc_normal_(self.pos_embed_f, std=.02)
        nn.init.trunc_normal_(self.pos_embed_t, std=.02)

    def __repr__(self):
        return f"Bias2d(channels={self.channels}, max_f={self.max_f}, max_t={self.max_t})"

    def forward(self, x):
        b, c, f, t = x.shape
        return x + self.pos_embed_f[:, :, :f, :t]


class Bias2dChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 1, 1))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def __repr__(self):
        return f"Bias2dChannels(channels={self.channels})"

    def forward(self, x):
        return x * self.pos_embed


class Bias2dFreq(nn.Module):
    def __init__(self, freq):
        super().__init__()
        self.freq = freq
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, freq, 1))
        nn.init.trunc_normal_(self.pos_embed, std=.00002)

    def __repr__(self):
        return f"Bias2dFreq(freq={self.freq})"

    def forward(self, x):
        return x + self.pos_embed


class Bias1d(nn.Module):
    def __init__(self, channels, max_len=2048):
        super(Bias1d, self).__init__()
        self.channels = channels
        self.max_len = max_len
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, max_len))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def __repr__(self):
        return f"Bias1d(channels={self.channels}, max_len={self.max_len})"

    def forward(self, x):
        b, c, seq_len = x.shape
        return x + self.pos_embed[:, :, :seq_len]


class Zeroes(nn.Module):
    """Zero out the input tensor"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)  # Return a tensor of zeros with the same shape as x


class Ones(nn.Module):
    """Zero out the input tensor"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)  # Return a tensor of zeros with the same shape as x


class SplitTensor(nn.Module):
    """Split tensor along specified dimension and apply the same function to each split"""

    def __init__(self, dim=1, fn=None):
        super().__init__()
        self.dim = dim
        self.fn = fn

    def __repr__(self):
        return f"Split(dim={self.dim}, fn={self.fn})"

    def forward(self, x):
        # Split along specified dimension into separate tensors
        split_tensors = torch.unbind(x, dim=self.dim)

        # Apply the same function to each split tensor
        outputs = []
        for split_tensor in split_tensors:
            if self.fn is not None:
                # Add dimension back for processing
                split_with_dim = split_tensor.unsqueeze(self.dim)
                output = self.fn(split_with_dim)
                outputs.append(output)
            else:
                # If no function specified, just return the split with dimension restored
                outputs.append(split_tensor.unsqueeze(self.dim))

        # Concatenate results back along the same dimension
        return torch.cat(outputs, dim=self.dim)


class SplitNTensor(nn.Module):
    def __init__(self, fns: List[nn.Module], split_points: List[int], dim: int = 1):
        """
        Split tensor at specified points along a dimension, apply separate functions to each split, then concatenate.

        Args:
            fns: List of pre-created functions/modules to apply to each split
            split_points: List of indices where to split the tensor (exclusive end points)
            dim: Dimension along which to split
        """
        super().__init__()
        self.dim = dim
        self.fns = nn.ModuleList(fns)
        self.split_points = split_points
        self.n_splits = len(split_points) + 1

        if len(fns) != self.n_splits:
            raise ValueError(
                f"Number of functions ({len(fns)}) must equal number of splits "
                f"({self.n_splits}) created by {len(split_points)} split points"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim_size = x.size(self.dim)

        for i, point in enumerate(self.split_points):
            if point <= 0 or point >= dim_size:
                raise ValueError(f"Split point {point} is out of bounds for dimension size {dim_size}")
            if i > 0 and point <= self.split_points[i - 1]:
                raise ValueError(f"Split points must be in ascending order, got {self.split_points}")

        splits = []
        start = 0

        for split_point in self.split_points:
            size = split_point - start
            splits.append(x.narrow(self.dim, start, size))
            start = split_point

        final_size = dim_size - start
        splits.append(x.narrow(self.dim, start, final_size))
        outputs = [self.fns[i](split) for i, split in enumerate(splits)]
        return torch.cat(outputs, dim=self.dim)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"n_splits={self.n_splits}, "
                f"split_points={self.split_points}, "
                f"dim={self.dim})")


class Checkpoint(nn.Module):
    """
    Wrapper module that applies gradient checkpointing to the wrapped function.

    Gradient checkpointing trades compute for memory by not storing intermediate
    activations during the forward pass, and recomputing them during backward pass.
    """

    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)

    def __repr__(self):
        return f"Checkpoint(fn={self.fn})"

    def forward(self, x):
        if self.training:
            return checkpoint(self.fn, x, use_reentrant=False)
        else:
            return self.fn(x)


class WithShape(nn.Module):
    def __init__(self, shape, fn):
        super().__init__()
        self.shape = shape
        self.fn = fn(shape)

    def __repr__(self):
        return f"WithShape(shape={self.shape}, fn={self.fn})"

    def forward(self, x):
        return self.fn(x)


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='linear',
                 align_corners=None, recompute_scale_factor=None, antialias=False):
        super(Interpolate, self).__init__()

        if size is not None and scale_factor is not None:
            raise ValueError("Cannot specify both 'size' and 'scale_factor'. "
                             "Please choose one.")

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def __repr__(self):
        return f"Interpolate(size={self.size}, scale_factor={self.scale_factor}, mode='{self.mode}', align_corners={self.align_corners}, recompute_scale_factor={self.recompute_scale_factor}, antialias={self.antialias})"

    def forward(self, x):
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias
        )


class GaussianNoise(nn.Module):
    """
    Adds Gaussian noise to input tensors with shape (batch, channels, frequency, time).

    Args:
        std (float): Standard deviation of the Gaussian noise. Default: 0.1
        mean (float): Mean of the Gaussian noise. Default: 0.0
        training_only (bool): If True, noise is only added during training. Default: True
    """

    def __init__(self, std=0.25, mean=0.0, training_only=False):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        self.training_only = training_only

    def __repr__(self):
        return f"GaussianNoise(std={self.std}, mean={self.mean}, training_only={self.training_only})"

    def forward(self, x):
        """
        Forward pass that adds Gaussian noise to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, frequency, time)

        Returns:
            torch.Tensor: Noisy tensor with the same shape as input
        """
        if self.training_only and not self.training:
            return x

        # Generate Gaussian noise with the same shape as input
        noise = torch.randn_like(x) * self.std + self.mean

        return x + noise

    def extra_repr(self):
        """String representation of the module parameters."""
        return f'std={self.std}, mean={self.mean}, training_only={self.training_only}'


class ToMagnitude(nn.Module):
    def __init__(self, shape: CFTShape, fn, eps=1e-6):
        super().__init__()
        self.shape = shape
        self.fn = fn(CFTShape(c=shape.c // 2, f=shape.f, t=shape.t))
        self.eps = eps

    def __repr__(self):
        return f"ToMagnitude(shape={self.shape}, fn={self.fn}, eps={self.eps})"

    def forward(self, x):
        real_parts = x[:, 0::2, :, :]
        imag_parts = x[:, 1::2, :, :]

        magnitude = torch.sqrt(real_parts ** 2 + imag_parts ** 2 + self.eps)

        processed_magnitude = self.fn(magnitude)

        scale_factor = processed_magnitude / (magnitude + self.eps)

        new_real = real_parts * scale_factor
        new_imag = imag_parts * scale_factor

        output = torch.zeros_like(x)
        output[:, 0::2, :, :] = new_real
        output[:, 1::2, :, :] = new_imag

        return output


class ToPhase(nn.Module):
    def __init__(self, shape: CFTShape, fn, eps=1e-6, phase_mode='direct'):
        super().__init__()
        self.shape = shape
        self.fn = fn(shape)
        self.eps = eps
        self.phase_mode = phase_mode  # 'direct', 'normalized', 'wrapped', 'complex_polar'

    def __repr__(self):
        return f"ToPhase(shape={self.shape}, fn={self.fn}, eps={self.eps}, phase_mode='{self.phase_mode}')"

    def forward(self, x):
        # Ensure input tensor has consistent dtype
        real_parts = x[:, 0::2, :, :]
        imag_parts = x[:, 1::2, :, :]

        # More robust magnitude calculation
        magnitude = torch.sqrt(torch.clamp(real_parts ** 2 + imag_parts ** 2, min=self.eps))
        eps_tensor = torch.tensor(self.eps, dtype=x.dtype, device=x.device)
        original_phase = torch.atan2(imag_parts, real_parts + eps_tensor)

        if self.phase_mode == 'direct':
            # Method 1: Direct phase prediction (often easier for networks)
            # Normalize phase to [0, 1] range for better learning
            normalized_phase = (original_phase + torch.pi) / (2 * torch.pi)

            # Let network predict normalized phase directly
            predicted_normalized = self.fn(normalized_phase)

            # Convert back to radians
            predicted_phase = predicted_normalized * (2 * torch.pi) - torch.pi

        elif self.phase_mode == 'normalized':
            # Method 2: Normalized phase with bounded output
            # Normalize to [-1, 1] and use tanh activation implicitly
            normalized_phase = original_phase / torch.pi
            predicted_normalized = self.fn(normalized_phase)
            predicted_phase = torch.tanh(predicted_normalized) * torch.pi

        elif self.phase_mode == 'wrapped':
            # Method 3: Unwrapped phase for continuity
            unwrapped_phase = torch.cumsum(torch.diff(original_phase, dim=-1, prepend=original_phase[..., :1]), dim=-1)
            predicted_unwrapped = self.fn(unwrapped_phase)
            # Wrap back to [-π, π]
            predicted_phase = torch.remainder(predicted_unwrapped + torch.pi, 2 * torch.pi) - torch.pi

        elif self.phase_mode == 'complex_polar':
            # Method 4: Work in complex exponential space with view_as_real
            # e^(iθ) representation - often more stable
            complex_unit = torch.complex(torch.cos(original_phase), torch.sin(original_phase))

            # Reshape to interleaved real/imag format for the network
            # Input shape: (b, c//2, f, t) -> Output shape: (b, c, f, t)
            b, c_half, f, t = complex_unit.shape

            # Convert complex tensor to interleaved real/imag format
            complex_input_interleaved = torch.zeros(b, c_half * 2, f, t, dtype=x.dtype, device=x.device)
            complex_input_interleaved[:, 0::2, :, :] = complex_unit.real
            complex_input_interleaved[:, 1::2, :, :] = complex_unit.imag

            # Network predicts new interleaved real/imag values
            predicted_interleaved = self.fn(complex_input_interleaved)

            # Extract real and imaginary parts from network output
            predicted_real = predicted_interleaved[:, 0::2, :, :]
            predicted_imag = predicted_interleaved[:, 1::2, :, :]

            # Reconstruct complex tensor
            predicted_complex = torch.complex(predicted_real, predicted_imag)

            # Normalize to unit circle (crucial for phase representation)
            predicted_complex = predicted_complex / (torch.abs(predicted_complex) + self.eps)

            # Extract phase from normalized complex number
            predicted_phase = torch.angle(predicted_complex)

        else:
            raise ValueError(f"Unknown phase_mode: {self.phase_mode}")

        # Final output calculation
        new_real = magnitude * torch.cos(predicted_phase)
        new_imag = magnitude * torch.sin(predicted_phase)

        # Safety net for NaN values
        new_real = torch.where(torch.isnan(new_real), torch.zeros_like(new_real), new_real)
        new_imag = torch.where(torch.isnan(new_imag), torch.zeros_like(new_imag), new_imag)

        # Assemble output
        output = torch.zeros_like(x)
        output[:, 0::2, :, :] = new_real
        output[:, 1::2, :, :] = new_imag

        return output


class ToMagnitudeOnly(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def __repr__(self):
        return f"ToMagnitudeOnly(eps={self.eps})"

    def forward(self, x):
        real_parts = x[:, 0::2, :, :]
        imag_parts = x[:, 1::2, :, :]

        magnitude = torch.sqrt(real_parts ** 2 + imag_parts ** 2 + self.eps)

        return magnitude


import torch
import torch.nn as nn


class ToPhaseOnly(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def __repr__(self):
        return f"ToPhaseOnly(eps={self.eps})"

    def forward(self, x):
        # Extract real and imaginary parts
        real_parts = x[:, 0::2, :, :]
        imag_parts = x[:, 1::2, :, :]

        # Compute and return phase
        phase = torch.atan2(imag_parts, real_parts + self.eps)

        return phase


class ToPhaseOnlySinusoidal(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def __repr__(self):
        return f"ToPhaseOnlySinusoidal(eps={self.eps})"

    def forward(self, x):
        # Extract real and imaginary parts
        real_parts = x[:, 0::2, :, :]
        imag_parts = x[:, 1::2, :, :]

        # Compute phase
        phase = torch.atan2(imag_parts, real_parts + self.eps)

        # Convert to sinusoidal representation
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Interleave cos and sin along channel dimension
        # cos_phase and sin_phase have shape [b, c//2, f, t]
        # We want output shape [b, c, f, t] with cos/sin interleaved
        b, c_half, f, t = cos_phase.shape
        output = torch.empty(b, c_half * 2, f, t, device=x.device, dtype=x.dtype)
        output[:, 0::2, :, :] = cos_phase  # Even indices get cos
        output[:, 1::2, :, :] = sin_phase  # Odd indices get sin

        return output


class PowerLawProcessor(nn.Module):
    def __init__(self, gamma, *args):
        super().__init__()
        self.gamma = gamma
        self.fn = Seq(*args)
        self.eps = 1e-8

    def __repr__(self):
        return f"PowerLawProcessor(gamma={self.gamma}, fn={self.fn}, eps={self.eps})"

    def forward(self, x):
        # Ensure positive values and clamp to avoid numerical issues
        x_safe = torch.clamp(x, min=self.eps)

        # Compress: x^(1/gamma)
        compressed = torch.pow(x_safe, 1.0 / self.gamma)

        # Process
        processed = self.fn(compressed)

        # Ensure processed values are safe for decompression
        processed_safe = torch.clamp(processed, min=self.eps)

        # Decompress: x^gamma
        decompressed = torch.pow(processed_safe, self.gamma)

        return decompressed


class ExpOnly(nn.Module):
    def __init__(self, factor, eps=1e-8):
        super().__init__()
        self.factor = factor
        self.eps = eps

    def __repr__(self):
        return f"ExpOnly(factor={self.factor}, eps={self.eps})"

    def forward(self, x):
        # Apply power law compression: x^(1/factor)
        # Handle negative values by preserving sign
        sign = torch.sign(x)
        abs_x = torch.abs(x) + self.eps

        # Compress using power law
        compressed = sign * torch.pow(abs_x, 1.0 / self.factor)

        return compressed
