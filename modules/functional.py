from dataclasses import dataclass, replace
from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange, parse_shape
from torch import nn
from torch.utils.checkpoint import checkpoint

from modules.fft import RFFTModule, RFFT2DModule
from modules.seq import Seq
from modules.stft import STFT


@dataclass
class CFTShape:
    c: int
    f: int
    t: int


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


class Residual(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)

    def forward(self, x):
        return self.fn(x) + x


class Mask(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)

    def __repr__(self):
        return f"Mask(fn={self.fn})"

    def forward(self, x):
        return self.fn(x) * x


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


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


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


class SplitSum(nn.Module):
    def forward(self, x):
        c_total = x.shape[1]
        c_half = c_total // 2
        x1 = x[:, :c_half, ...]
        x2 = x[:, c_half:, ...]
        return x1 + x2


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


class CopyDim(nn.Module):
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
        shape_dict = dict(zip(['b', 'c', 'f', 't'], x.shape))
        x_intermediate = rearrange(x, f'b c f t -> {self.reshape_to}', **shape_dict)
        x_processed = self.fn(x_intermediate)
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


class ComplexMask(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)

    def __repr__(self):
        return f"ComplexMask(fn={self.fn})"

    def forward(self, x):
        b, c, f, t = x.shape
        assert c % 2 == 0, "Channel dimension must be even for real/imag pairs"
        original_dtype = x.dtype
        x_float32 = x.float()
        x_complex = rearrange(x_float32, 'b (ch ri) f t -> b ch f t ri', ri=2)
        residual = torch.view_as_complex(x_complex.contiguous())
        x_processed = self.fn(x)
        x_processed_float32 = x_processed.float()
        x_processed_complex = rearrange(x_processed_float32, 'b (ch ri) f t -> b ch f t ri', ri=2)
        x_processed = torch.view_as_complex(x_processed_complex.contiguous())
        result = residual * x_processed
        result_real = torch.view_as_real(result)
        result = rearrange(result_real, 'b ch f t ri -> b (ch ri) f t')
        result = result.to(original_dtype)
        return result


class FFT1dAndInverse(nn.Module):
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


class FFT2dAndInverse(nn.Module):
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


class FFT1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft = RFFTModule()

    def __repr__(self):
        return f"ToFFT1dOnly()"

    def forward(self, x):
        x = self.fft(x)
        return x


class FFT2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft = RFFT2DModule()

    def __repr__(self):
        return f"ToFFT1dOnly()"

    def forward(self, x):
        x = self.fft(x)
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


class ToSTFT(nn.Module):
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


class Zeroes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class Ones(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)


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


class ToMagnitudeAndInverse(nn.Module):
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


class ToMagnitude(nn.Module):
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
