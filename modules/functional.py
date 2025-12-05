import math
from dataclasses import dataclass, replace
from typing import List, Callable

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


class ReshapeBCFT(nn.Module):
    def __init__(self, reshape_to, *args):
        super().__init__()
        self.fn = Seq(*args)
        self.reshape_to = reshape_to

    def __repr__(self):
        return f"ReshapeBCFT(reshape_to='{self.reshape_to}', fn={self.fn})"

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        shape_dict = dict(zip(['b', 'c', 'f', 't'], x.shape))
        x = rearrange(x, f'b c f t -> {self.reshape_to}', **shape_dict)
        x_processed = self.fn(x, **kwargs)
        output = rearrange(x_processed, f'{self.reshape_to} -> b c f t', **shape_dict)
        return output


class DebugShape(nn.Module):
    def __init__(self, name: str = None):
        super().__init__()
        self.fn = Seq(nn.Identity())
        self.name = name

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.fn(x, **kwargs)
        name_prefix = f"{self.name}: " if self.name else ""
        print(
            f"{name_prefix}Shape: {x.shape}, Min: {x.min().item():.4f}, Max: {x.max().item():.4f}, Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
        return x


class Residual(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)

    def forward(self, x, **kwargs):
        x = x + self.fn(x, **kwargs)
        return x


class Mask(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)

    def __repr__(self):
        return f"Mask(fn={self.fn})"

    def forward(self, x, **kwargs):
        x = x * self.fn(x, **kwargs)
        return x


class Scale(nn.Module):
    def __init__(self, scale, *args):
        super().__init__()
        self.fn = Seq(*args)
        self.scale = scale

    def __repr__(self):
        return f"Scale(scale={self.scale}, fn={self.fn})"

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class Concat(nn.Module):
    def __init__(self, dim=1, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)
        self.dim = dim

    def __repr__(self):
        return f"Concat(fns={self.fns}, dim={self.dim})"

    def forward(self, x, **kwargs):
        outputs = [fn(x, **kwargs) for fn in self.fns]
        return torch.cat(outputs, dim=self.dim)


class Abs(nn.Module):
    def forward(self, x, **kwargs):
        return torch.abs(x)


class Film(nn.Module):
    def __init__(self, dim=1, *args):
        super().__init__()
        self.dim = dim
        self.fn = Seq(*args)
        self.silu = nn.SiLU()

    def __repr__(self):
        return f"Film(dim={self.dim}, fn={self.fn})"

    def forward(self, x, **kwargs):
        residual = x
        x = self.fn(x, **kwargs)
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

    def forward(self, x, **kwargs):
        gate, value = torch.split(x, x.shape[self.dim] // 2, dim=self.dim)
        return self.silu(gate) * value


class SplitSum(nn.Module):
    def forward(self, x, **kwargs):
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

    def forward(self, x, **kwargs):
        """Determines whether to drop the block and runs it if not dropped."""
        if self.training and torch.rand(1).item() < self.drop_prob:
            return x
        return self.fn(x, **kwargs)


class Freeze(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)
        for param in self.fn.parameters():
            param.requires_grad = False

    def __repr__(self):
        return f"Freeze(fn={self.fn})"

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs)


class Repeat(nn.Module):
    def __init__(self, num_repeats, *args):
        super().__init__()
        self.num_repeats = num_repeats
        self.modules_list = nn.ModuleList([Seq(*args) for _ in range(num_repeats)])

    def __repr__(self):
        return f"Repeat(num_repeats={self.num_repeats}, modules_list={self.modules_list})"

    def forward(self, x, **kwargs):
        for module in self.modules_list:
            x = module(x, **kwargs)
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

    def forward(self, x, **kwargs):
        for i in range(self.num_repeats):
            x = self.blocks[i](x, **kwargs)
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

    def forward(self, x, **kwargs):
        outputs = []
        for i in range(self.num_repeats):
            output_i = self.blocks[i](x, **kwargs)
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

    def forward(self, x, **kwargs):
        sizes = [1] * x.ndim
        sizes[self.dim] = self.times
        x = x.repeat(*sizes)
        return x


class PadBCFT(nn.Module):
    def __init__(self, shape: CFTShape, target_time=None, fn=None, mode='constant', value=0):
        """
        Pad BCFT tensors from input shape to target dimensions.

        Args:
            shape: CFTShape object with input dimensions (c, f, t)
            target_time: Target time dimension (if None, use shape.t)
            fn: Function constructor that will receive the padded shape
            mode: Padding mode ('reflect', 'constant', 'replicate', etc.)
            value: Fill value for constant padding (default: 0)
        """
        super(PadBCFT, self).__init__()
        self.shape = shape
        self.target_time = target_time if target_time is not None else shape.t
        self.mode = mode
        self.value = value

        # Create the padded shape and initialize fn with it
        padded_shape = CFTShape(
            c=shape.c,
            f=shape.f,
            t=self.target_time
        )
        self.fn = fn(padded_shape) if fn is not None else None

    def __repr__(self):
        return f"PadBCFT(shape={self.shape}, target_time={self.target_time}, fn={self.fn}, mode='{self.mode}', value={self.value})"

    def forward(self, x, **kwargs):
        b, c, f, t = x.shape

        t_pad_right = self.target_time - t

        # Apply padding if needed
        padding = (0, t_pad_right, 0, 0)
        x = nn.functional.pad(x, padding, mode=self.mode, value=self.value)

        # Apply function if provided
        if self.fn is not None:
            x = self.fn(x, **kwargs)

        if t_pad_right > 0:
            x = x[:, :, :, :t]

        return x


class PadBCFTNearestMultiple(nn.Module):
    def __init__(self, shape: CFTShape, freq_multiple: int, time_multiple: int, fn: Callable):
        super(PadBCFTNearestMultiple, self).__init__()
        self.shape = shape
        self.freq_multiple = freq_multiple
        self.time_multiple = time_multiple

        # Calculate padded shape
        f_remainder = shape.f % freq_multiple
        f_pad = (freq_multiple - f_remainder) % freq_multiple
        t_remainder = shape.t % time_multiple
        t_pad = (time_multiple - t_remainder) % time_multiple

        padded_shape = replace(
            shape,
            f=shape.f + f_pad,
            t=shape.t + t_pad
        )

        self.fn = fn(padded_shape)

    def __repr__(self):
        return (f"PadBCFTNearestMultiple("
                f"shape={self.shape}, "
                f"freq_multiple={self.freq_multiple}, "
                f"time_multiple={self.time_multiple}, "
                f"fn={self.fn})")

    def forward(self, x, **kwargs):
        b, c, f, t = x.shape
        f_remainder = f % self.freq_multiple
        f_pad = (self.freq_multiple - f_remainder) % self.freq_multiple
        t_remainder = t % self.time_multiple
        t_pad = (self.time_multiple - t_remainder) % self.time_multiple
        padding = (0, t_pad, 0, f_pad)
        x = nn.functional.pad(x, padding, mode='constant', value=0)
        x = self.fn(x, **kwargs)
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

    def forward(self, x, **kwargs):
        b, c, f, t = x.shape
        t_remainder = t % self.time_multiple
        t_pad = (self.time_multiple - t_remainder) % self.time_multiple
        padding = (0, t_pad, 0, 0)
        x = nn.functional.pad(x, padding, mode='constant', value=0)
        x = self.fn(x, **kwargs)
        if t_pad > 0:
            x = x[:, :, :, :t]
        return x


class SoftmaxGroups(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def __repr__(self):
        return f"SoftmaxGroups(num_groups={self.num_groups})"

    def forward(self, x, **kwargs):
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

    def forward(self, x, **kwargs):
        mixture = x
        x = self.fn(x, **kwargs)
        x = self.softmax_groups(x)
        num_multiplies = x.shape[1] // mixture.shape[1]
        mixture = mixture.repeat(1, num_multiplies, 1, 1)
        x = x * mixture
        return x


class TanhMask(nn.Module):
    def __init__(self, num_instruments, *args):
        super().__init__()
        # Tanh outputs values between -1 and 1
        self.activation = nn.Tanh()
        self.num_instruments = num_instruments
        self.fn = Seq(*args)

    def __repr__(self):
        return f"TanhMask(num_instruments={self.num_instruments}, fn={self.fn})"

    def forward(self, x, **kwargs):
        mixture = x

        # Forward pass through the sub-network
        x = self.fn(x, **kwargs)

        # Apply Tanh (-1 to 1)
        # We do not need SoftmaxGroups here because Tanh is element-wise
        # and does not require grouping dimensions to normalize.
        x = self.activation(x)

        # Expand mixture to match the number of output channels
        # (e.g., if input is stereo and output is 4 instruments * stereo)
        num_multiplies = x.shape[1] // mixture.shape[1]
        mixture = mixture.repeat(1, num_multiplies, 1, 1)

        # Apply the mask
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

    def forward(self, x, **kwargs):
        return self.thing(x, **kwargs)


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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        b, c, f, t = x.shape

        remainder = f % self.num_splits
        if remainder != 0:
            padding = self.num_splits - remainder
            x = nn.functional.pad(x, (0, 0, 0, padding))
            f = f + padding

        x = rearrange(x, 'b c (n f) t -> b (n c) f t', n=self.num_splits)
        x = self.fn(x, **kwargs)
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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        b, c, f, t = x.shape
        remainder = t % self.num_splits
        if remainder != 0:
            padding = self.num_splits - remainder
            x = nn.functional.pad(x, (0, padding, 0, 0))
            t = t + padding

        x = rearrange(x, 'b c f (n t) -> b (n c) f t', n=self.num_splits)
        x = self.fn(x, **kwargs)
        x = rearrange(x, 'b (n c) f t -> b c f (n t)', n=self.num_splits)

        if remainder != 0:
            x = x[:, :, :, :t - padding]
        return x


class Condition(nn.Module):
    def __init__(self, condition, true_fn, false_fn=None):
        super().__init__()
        self.condition = condition
        self.true_fn = Seq(true_fn())
        self.false_fn = Seq(false_fn()) if false_fn is not None else None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.condition(x):
            return self.true_fn(x, **kwargs)
        elif self.false_fn is not None:
            return self.false_fn(x, **kwargs)
        else:
            return x


class ReshapeBCT(nn.Module):
    def __init__(self, reshape_to, *args):
        super().__init__()
        self.fn = Seq(*args)
        self.reshape_to = reshape_to

    def __repr__(self):
        return f"ReshapeBCT(reshape_to='{self.reshape_to}', fn={self.fn})"

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        original_shape_context = parse_shape(x, 'b c t')
        x = rearrange(x, f'b c t -> {self.reshape_to}', **original_shape_context)
        x_processed = self.fn(x, **kwargs)
        processed_shape_context = parse_shape(x_processed, self.reshape_to)
        output = rearrange(x_processed, f'{self.reshape_to} -> b c t', **processed_shape_context)
        return output


class Module(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __repr__(self):
        return f"Module(fn={self.fn})"

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs)


class SideEffect(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __repr__(self):
        return f"SideEffect(fn={self.fn})"

    def forward(self, x, **kwargs):
        self.fn(x, **kwargs)
        return x


class ComplexMask(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)

    def forward(self, x, **kwargs):
        b, c, f, t = x.shape
        mask_out = self.fn(x, **kwargs)

        # Repeat x to match mask_out channels
        num_repeats = mask_out.shape[1] // c
        x_repeated = x.repeat(1, num_repeats, 1, 1)

        mask_complex = torch.view_as_complex(
            rearrange(mask_out.float(), 'b (ch ri) f t -> b ch f t ri', ri=2).contiguous()
        )
        x_complex = torch.view_as_complex(
            rearrange(x_repeated.float(), 'b (ch ri) f t -> b ch f t ri', ri=2).contiguous()
        )
        result_complex = x_complex * mask_complex
        result_real = torch.view_as_real(result_complex)
        result = rearrange(result_real, 'b ch f t ri -> b (ch ri) f t')
        return result.type_as(x)


class FFT1dAndInverse(nn.Module):
    def __init__(self, channels, fn):
        super().__init__()
        self.channels = channels
        self.fn = fn(channels * 2)
        self.fft = RFFTModule()

    def __repr__(self):
        return f"ToFFT(channels={self.channels}, fn={self.fn})"

    def forward(self, x, **kwargs):
        original_time_dim = x.shape[-1]
        x = self.fft(x)
        x = self.fn(x, **kwargs)
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

    def forward(self, x, **kwargs):
        original_freq_dim = x.shape[2]
        original_time_dim = x.shape[3]
        x = self.fft(x)
        x = self.fn(x, **kwargs)
        x = self.fft.inverse(x, original_freq_dim, original_time_dim)
        return x


class FFT1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft = RFFTModule()

    def __repr__(self):
        return f"ToFFT1dOnly()"

    def forward(self, x, **kwargs):
        x = self.fft(x)
        return x


class FFT2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft = RFFT2DModule()

    def __repr__(self):
        return f"ToFFT1dOnly()"

    def forward(self, x, **kwargs):
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
        self.stft = ToSTFT(n_fft=n_fft, hop_length=hop_length)
        self.inverse_stft = InverseSTFT(n_fft=n_fft, hop_length=hop_length)

    def __repr__(self):
        return f"STFTAndInverse(in_channels={self.in_channels}, in_samples={self.in_samples}, out_f={self.out_f}, hop_length={self.hop_length}, fn={self.fn})"

    def forward(self, x, **kwargs):
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input tensor has {x.shape[1]} channels, but model expects {self.in_channels} channels"
            )
        original_length = x.shape[-1]
        x = self.stft(x)
        x = self.fn(x, **kwargs)
        x = self.inverse_stft(x, original_length)
        return x


class ToSTFT(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.out_f = n_fft // 2
        self.hop_length = hop_length
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)

    def __repr__(self):
        return f"ToSTFT(out_f={self.out_f}, hop_length={self.hop_length})"

    def forward(self, x, **kwargs):
        x = self.stft(x.float())
        return x


class InverseSTFT(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)

    def __repr__(self):
        return f"InverseSTFT(n_fft={self.n_fft}, hop_length={self.hop_length})"

    def forward(self, x, length=None, **kwargs):
        # x shape: (batch, channels*2, freq, time) - complex as real/imag channels
        target_length = length if length is not None else self.out_samples
        x = self.stft.inverse(x, target_length)
        return x


class Zeroes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros_like(x)


class Ones(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.ones_like(x)


class SplitNTensor(nn.Module):
    def __init__(self, shape: CFTShape, fns: List[Callable], split_points: List[int], dim: int = 1,
                 concat_dim=1):
        super().__init__()
        self.shape = shape
        self.dim = dim
        self.concat_dim = concat_dim if concat_dim is not None else dim
        self.split_points = split_points
        self.n_splits = len(split_points) + 1

        if len(fns) != self.n_splits:
            raise ValueError(
                f"Number of functions ({len(fns)}) must equal number of splits "
                f"({self.n_splits}) created by {len(split_points)} split points"
            )

        if dim == 1:
            total_size = shape.c
        elif dim == 2:
            total_size = shape.f
        elif dim == 3:
            total_size = shape.t
        else:
            raise ValueError(f"Unsupported dimension {dim} for CFTShape")

        for i, point in enumerate(self.split_points):
            if point <= 0 or point >= total_size:
                raise ValueError(f"Split point {point} is out of bounds for dimension size {total_size}")
            if i > 0 and point <= self.split_points[i - 1]:
                raise ValueError(f"Split points must be in ascending order, got {self.split_points}")

        self.fns = nn.ModuleList()
        start = 0

        for i, split_point in enumerate(self.split_points):
            size = split_point - start
            split_shape = self._create_split_shape(shape, size, dim)
            self.fns.append(fns[i](split_shape, i))
            start = split_point

        final_size = total_size - start
        final_split_shape = self._create_split_shape(shape, final_size, dim)
        self.fns.append(fns[-1](final_split_shape, len(self.split_points)))

        self.streams = [torch.cuda.Stream() for _ in range(len(self.fns) - 1)]

    def _create_split_shape(self, original_shape: CFTShape, size: int, dim: int) -> CFTShape:
        if dim == 1:
            return replace(original_shape, c=size)
        elif dim == 2:
            return replace(original_shape, f=size)
        elif dim == 3:
            return replace(original_shape, t=size)
        else:
            raise ValueError(f"Unsupported dimension {dim}")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        dim_size = x.size(self.dim)

        splits = []
        start = 0
        for split_point in self.split_points:
            splits.append(x.narrow(self.dim, start, split_point - start))
            start = split_point
        splits.append(x.narrow(self.dim, start, dim_size - start))

        outputs = [torch.empty(0)] * len(splits)

        default_stream = torch.cuda.current_stream()

        input_ready_event = torch.cuda.Event()
        input_ready_event.record(default_stream)

        outputs[0] = self.fns[0](splits[0], **kwargs)

        side_stream_events = []

        for i, stream in enumerate(self.streams):
            split_idx = i + 1
            with torch.cuda.stream(stream):
                stream.wait_event(input_ready_event)
                outputs[split_idx] = self.fns[split_idx](splits[split_idx], **kwargs)
                event = torch.cuda.Event()
                event.record(stream)
                side_stream_events.append(event)

        for event in side_stream_events:
            default_stream.wait_event(event)

        return torch.cat(outputs, dim=self.concat_dim)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"shape={self.shape}, "
                f"n_splits={self.n_splits}, "
                f"split_points={self.split_points}, "
                f"dim={self.dim}, "
                f"concat_dim={self.concat_dim})")


class Checkpoint(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.fn = Seq(*args)

    def __repr__(self):
        return f"Checkpoint(fn={self.fn})"

    def forward(self, x, **kwargs):
        if self.training:
            return checkpoint(self.fn, x, use_reentrant=False)
        else:
            return self.fn(x, **kwargs)


class WithShape(nn.Module):
    def __init__(self, shape, fn):
        super().__init__()
        self.shape = shape
        self.fn = fn(shape)

    def __repr__(self):
        return f"WithShape(shape={self.shape}, fn={self.fn})"

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs)


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

    def forward(self, x, **kwargs):
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias
        )


class ToMagnitude(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def __repr__(self):
        return f"ToMagnitude(eps={self.eps})"

    def forward(self, x, **kwargs):
        real_parts = x[:, 0::2, :, :]
        imag_parts = x[:, 1::2, :, :]
        magnitude = torch.sqrt(real_parts ** 2 + imag_parts ** 2 + self.eps)
        return magnitude


class FromMagnitude(nn.Module):
    def __init__(self, eps=1e-6, retain_phase=True):
        super().__init__()
        self.eps = eps
        self.retain_phase = retain_phase

    def __repr__(self):
        return f"FromMagnitude(eps={self.eps}, retain_phase={self.retain_phase})"

    def forward(self, processed_magnitude, original_complex, **kwargs):
        real_parts = original_complex[:, 0::2, :, :]
        imag_parts = original_complex[:, 1::2, :, :]
        magnitude = torch.sqrt(real_parts ** 2 + imag_parts ** 2 + self.eps)

        expansion = processed_magnitude.shape[1] // magnitude.shape[1]

        if self.retain_phase:
            phase = torch.atan2(imag_parts, real_parts)
            phase_expanded = phase.repeat_interleave(expansion, dim=1)
            new_real = processed_magnitude * torch.cos(phase_expanded)
            new_imag = processed_magnitude * torch.sin(phase_expanded)
        else:
            mask = processed_magnitude / (magnitude.repeat_interleave(expansion, dim=1) + self.eps)
            real_expanded = real_parts.repeat_interleave(expansion, dim=1)
            imag_expanded = imag_parts.repeat_interleave(expansion, dim=1)
            new_real = real_expanded * mask
            new_imag = imag_expanded * mask

        batch, out_channels, freq, time = new_real.shape
        output = torch.zeros(batch, out_channels * 2, freq, time,
                             dtype=new_real.dtype, device=new_real.device)
        output[:, 0::2, :, :] = new_real
        output[:, 1::2, :, :] = new_imag
        return output


class ToMagnitudeAndInverse(nn.Module):
    def __init__(self, shape: CFTShape, fn, eps=1e-6, retain_phase=True):
        super().__init__()
        self.shape = shape
        self.fn = fn(CFTShape(c=shape.c // 2, f=shape.f, t=shape.t))
        self.eps = eps
        self.retain_phase = retain_phase
        self.to_mag = ToMagnitude(eps=eps)
        self.from_mag = FromMagnitude(eps=eps, retain_phase=retain_phase)

    def __repr__(self):
        return f"ToMagnitudeAndInverse(shape={self.shape}, fn={self.fn}, eps={self.eps}, retain_phase={self.retain_phase})"

    def forward(self, x, **kwargs):
        magnitude = self.to_mag(x[:, :4, :, :], **kwargs)
        processed_magnitude = self.fn(magnitude, **kwargs)
        return self.from_mag(processed_magnitude, x, **kwargs)


class Gamma(nn.Module):
    def __init__(self, gamma=2.2, eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def __repr__(self):
        return f"Gamma(gamma={self.gamma}, eps={self.eps})"

    def forward(self, x, **kwargs):
        x_normalized = torch.abs(x) + self.eps
        x_gamma = torch.pow(x_normalized, 1.0 / self.gamma)
        return torch.sign(x) * x_gamma


class FlowMatchingNoise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            targets: torch.Tensor,
            timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = targets.shape[0]
        device = targets.device

        if timestep is None:
            timestep = torch.rand(b, device=device)

        if timestep.dim() == 1:
            t = timestep.view(b, *([1] * (targets.dim() - 1)))
        elif timestep.dim() == 3 and timestep.shape[1:] == (1, 1):
            if targets.dim() == 4:
                t = timestep.unsqueeze(1)
            else:
                t = timestep
        else:
            t = timestep

        noise = torch.randn_like(targets)
        noisy = (1 - t) * noise + t * targets
        target_velocity = targets - noise

        return noisy, target_velocity, timestep
