import torch
from torch import nn

from modules.functional import CFTShape
from modules.seq import Seq


class UNet(nn.Module):
    def __init__(
            self,
            input_shape: CFTShape,
            channels,
            stride,
            output_channels,
            bottleneck_fn=None,
            skip_fn=None,
            pre_downsample_fn=None,
            post_downsample_fn=None,
            post_upsample_fn=None,
            post_skip_fn=None,
            dropout_rate: float = 0.0,
            dropout_after_downsample: bool = False,
            dropout_after_upsample: bool = True,
            dropout_in_bottleneck: bool = True,
    ) -> None:
        super(UNet, self).__init__()
        self.input_shape = input_shape
        self.channels = channels
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        self.output_channels = output_channels
        self.bottleneck_fn = bottleneck_fn
        self.skip_fn = skip_fn if skip_fn is not None else lambda x, skip: x + skip
        self.pre_downsample_fn = pre_downsample_fn
        self.post_downsample_fn = post_downsample_fn
        self.post_upsample_fn = post_upsample_fn
        self.post_skip_fn = post_skip_fn

        self.dropout_rate = dropout_rate
        self.dropout_after_downsample = dropout_after_downsample
        self.dropout_after_upsample = dropout_after_upsample
        self.dropout_in_bottleneck = dropout_in_bottleneck

        self.downsample_layers = nn.ModuleList()
        self.pre_downsample_modules = nn.ModuleList() if pre_downsample_fn else None
        self.post_downsample_modules = nn.ModuleList() if post_downsample_fn else None
        self.downsample_dropout = nn.ModuleList() if dropout_rate > 0 and dropout_after_downsample else None

        num_downsamples = len(self.channels) - 1
        freq_multiple = self.stride[0] ** num_downsamples
        time_multiple = self.stride[1] ** num_downsamples

        f_remainder = input_shape.f % freq_multiple
        f_pad = (freq_multiple - f_remainder) % freq_multiple
        padded_f = input_shape.f + f_pad

        t_remainder = input_shape.t % time_multiple
        t_pad = (time_multiple - t_remainder) % time_multiple
        padded_t = input_shape.t + t_pad

        current_shape = CFTShape(c=input_shape.c, f=padded_f, t=padded_t)
        encoder_shapes = []

        for i in range(len(self.channels) - 1):
            encoder_shapes.append(current_shape)
            if self.pre_downsample_modules is not None:
                self.pre_downsample_modules.append(pre_downsample_fn(current_shape))

            output_shape = CFTShape(
                c=self.channels[i + 1],
                f=current_shape.f // self.stride[0],
                t=current_shape.t // self.stride[1]
            )
            self.downsample_layers.append(
                Seq(
                    nn.Conv2d(
                        in_channels=self.channels[i],
                        out_channels=self.channels[i + 1],
                        kernel_size=self.stride,
                        stride=self.stride
                    ),
                    nn.GELU(),
                ),
            )
            if self.post_downsample_modules is not None:
                self.post_downsample_modules.append(post_downsample_fn(output_shape))

            if self.downsample_dropout is not None:
                self.downsample_dropout.append(nn.Dropout2d(dropout_rate))

            current_shape = output_shape

        self.bottleneck_fn = bottleneck_fn(current_shape) if bottleneck_fn else nn.Identity()

        self.bottleneck_dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 and dropout_in_bottleneck else None

        self.upsample_layers = nn.ModuleList()
        self.post_upsample_modules = nn.ModuleList() if post_upsample_fn else None
        self.post_skip_modules = nn.ModuleList() if post_skip_fn else None
        self.upsample_dropout = nn.ModuleList() if dropout_rate > 0 and dropout_after_upsample else None

        for i in range(len(self.channels) - 1, 0, -1):
            self.upsample_layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i - 1],
                    kernel_size=self.stride,
                    stride=self.stride
                )
            )
            output_shape = encoder_shapes[i - 1]
            if self.post_upsample_modules is not None:
                self.post_upsample_modules.append(post_upsample_fn(output_shape))
            if self.post_skip_modules is not None:
                self.post_skip_modules.append(post_skip_fn(output_shape))

            if self.upsample_dropout is not None:
                self.upsample_dropout.append(nn.Dropout2d(dropout_rate))

        self.final_conv = nn.Conv2d(self.channels[0], self.output_channels, kernel_size=1)

    def __repr__(self):
        return (f"UNet(input_shape={self.input_shape}, channels={self.channels}, "
                f"stride={self.stride}, output_channels={self.output_channels}, "
                f"bottleneck_fn={self.bottleneck_fn}, skip_fn={self.skip_fn}, "
                f"pre_downsample_fn={self.pre_downsample_fn}, "
                f"post_downsample_fn={self.post_downsample_fn}, "
                f"post_upsample_fn={self.post_upsample_fn}, "
                f"post_skip_fn={self.post_skip_fn}, "
                f"dropout_rate={self.dropout_rate}, "
                f"dropout_after_downsample={self.dropout_after_downsample}, "
                f"dropout_after_upsample={self.dropout_after_upsample}, "
                f"dropout_in_bottleneck={self.dropout_in_bottleneck})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape

        num_downsamples = len(self.channels) - 1
        freq_multiple = self.stride[0] ** num_downsamples
        time_multiple = self.stride[1] ** num_downsamples

        f_remainder = f % freq_multiple
        f_pad = (freq_multiple - f_remainder) % freq_multiple
        t_remainder = t % time_multiple
        t_pad = (time_multiple - t_remainder) % time_multiple

        padding = (0, t_pad, 0, f_pad)
        if f_pad > 0 or t_pad > 0:
            x = nn.functional.pad(x, padding, mode='constant', value=0)

        encoder_outputs = []
        current = x

        for i, downsample in enumerate(self.downsample_layers):
            encoder_outputs.append(current)
            if self.pre_downsample_modules is not None:
                current = self.pre_downsample_modules[i](current)
            current = downsample(current)
            if self.post_downsample_modules is not None:
                current = self.post_downsample_modules[i](current)
            if self.downsample_dropout is not None:
                current = self.downsample_dropout[i](current)

        current = self.bottleneck_fn(current)
        if self.bottleneck_dropout is not None:
            current = self.bottleneck_dropout(current)

        for i, upsample in enumerate(self.upsample_layers):
            current = upsample(current)
            if self.post_upsample_modules is not None:
                current = self.post_upsample_modules[i](current)
            if self.upsample_dropout is not None:
                current = self.upsample_dropout[i](current)
            skip = encoder_outputs[len(encoder_outputs) - 1 - i]
            current = self.skip_fn(current, skip)
            if self.post_skip_modules is not None:
                current = self.post_skip_modules[i](current)

        output = self.final_conv(current)

        if f_pad > 0 or t_pad > 0:
            output = output[:, :, :f, :t]

        return output
