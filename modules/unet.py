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
        self.encoder_shapes = []

        for i in range(len(self.channels) - 1):
            self.encoder_shapes.append(current_shape)
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
                    nn.SiLU(),
                ),
            )
            if self.post_downsample_modules is not None:
                self.post_downsample_modules.append(post_downsample_fn(output_shape))

            if self.downsample_dropout is not None:
                self.downsample_dropout.append(nn.Dropout2d(dropout_rate))

            current_shape = output_shape

        self.latent_shape = current_shape
        self._bottleneck = bottleneck_fn(current_shape) if bottleneck_fn else Seq(nn.Identity())

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
            output_shape = self.encoder_shapes[i - 1]
            if self.post_upsample_modules is not None:
                self.post_upsample_modules.append(post_upsample_fn(output_shape))
            if self.post_skip_modules is not None:
                self.post_skip_modules.append(post_skip_fn(output_shape))

            if self.upsample_dropout is not None:
                self.upsample_dropout.append(nn.Dropout2d(dropout_rate))

        self.final_conv = nn.Conv2d(self.channels[0], self.output_channels, kernel_size=1)

    def __repr__(self):
        return f"UNet(input_shape={self.input_shape}, channels={self.channels}"

    def _compute_padding(self, f, t_dim):
        """Compute padding needed for input dimensions."""
        num_downsamples = len(self.channels) - 1
        freq_multiple = self.stride[0] ** num_downsamples
        time_multiple = self.stride[1] ** num_downsamples

        f_remainder = f % freq_multiple
        f_pad = (freq_multiple - f_remainder) % freq_multiple
        t_remainder = t_dim % time_multiple
        t_pad = (time_multiple - t_remainder) % time_multiple

        return f_pad, t_pad

    def _pad_input(self, x):
        """Pad input to be divisible by stride^num_downsamples."""
        b, c, f, t_dim = x.shape
        f_pad, t_pad = self._compute_padding(f, t_dim)

        if f_pad > 0 or t_pad > 0:
            x = nn.functional.pad(x, (0, t_pad, 0, f_pad), mode='constant', value=0)

        return x, (f, t_dim)

    def encode(self, x: torch.Tensor, include_skips: bool = False, **kwargs):
        """
        Run only the encoder portion of the UNet.

        Args:
            x: Input tensor of shape (B, C, F, T)
            include_skips: If True, return skip connections for UNet-style decoding
            **kwargs: Additional arguments passed to submodules

        Returns:
            If include_skips=False: latent tensor of shape (B, channels[-1], F', T')
            If include_skips=True: (latent, encoder_outputs) tuple
        """
        x, original_dims = self._pad_input(x)
        encoder_outputs = []
        current = x

        for i, downsample in enumerate(self.downsample_layers):
            if include_skips:
                encoder_outputs.append(current)
            if self.pre_downsample_modules is not None:
                current = self.pre_downsample_modules[i](current, **kwargs)
            current = downsample(current, **kwargs)
            if self.post_downsample_modules is not None:
                current = self.post_downsample_modules[i](current, **kwargs)
            if self.downsample_dropout is not None:
                current = self.downsample_dropout[i](current)

        current = self._bottleneck(current, **kwargs)
        if self.bottleneck_dropout is not None:
            current = self.bottleneck_dropout(current)

        if include_skips:
            return current, encoder_outputs, original_dims
        return current

    def decode(self, z: torch.Tensor, encoder_outputs=None, original_dims=None, **kwargs):
        """
        Run only the decoder portion of the UNet.

        Args:
            z: Latent tensor from encoder
            encoder_outputs: Skip connections from encoder (optional, for UNet mode)
            original_dims: (f, t_dim) tuple for cropping output to original size
            **kwargs: Additional arguments passed to submodules

        Returns:
            Decoded tensor
        """
        current = z

        for i, upsample in enumerate(self.upsample_layers):
            current = upsample(current)
            if self.post_upsample_modules is not None:
                current = self.post_upsample_modules[i](current, **kwargs)
            if self.upsample_dropout is not None:
                current = self.upsample_dropout[i](current)

            if encoder_outputs is not None:
                skip = encoder_outputs[len(encoder_outputs) - 1 - i]
                current = self.skip_fn(current, skip)
                if self.post_skip_modules is not None:
                    current = self.post_skip_modules[i](current, **kwargs)

        output = self.final_conv(current)

        if original_dims is not None:
            f, t_dim = original_dims
            output = output[:, :, :f, :t_dim]

        return output

    def get_encoder(self):
        """
        Returns a callable encoder module that shares weights with this UNet.
        Useful for autoencoder training or feature extraction.
        """
        unet = self

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self._unet = unet

            def forward(self, x, **kwargs):
                return self._unet.encode(x, include_skips=False, **kwargs)

            @property
            def latent_shape(self):
                return self._unet.latent_shape

        return Encoder()

    def get_decoder(self):
        """
        Returns a callable decoder module that shares weights with this UNet.
        Note: This decoder does NOT use skip connections (autoencoder mode).
        """
        unet = self

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self._unet = unet

            def forward(self, z, original_dims=None, **kwargs):
                return self._unet.decode(z, encoder_outputs=None,
                                         original_dims=original_dims, **kwargs)

        return Decoder()

    def get_autoencoder(self):
        """
        Returns an autoencoder (encoder + decoder without skip connections).
        Shares weights with this UNet.
        """
        unet = self

        class AutoEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self._unet = unet
                self.encoder = unet.get_encoder()
                self.decoder = unet.get_decoder()

            def forward(self, x, **kwargs):
                x_padded, original_dims = self._unet._pad_input(x)
                z = self._unet.encode(x_padded, include_skips=False, **kwargs)
                return self._unet.decode(z, original_dims=original_dims, **kwargs)

            def encode(self, x, **kwargs):
                return self._unet.encode(x, include_skips=False, **kwargs)

            def decode(self, z, original_dims=None, **kwargs):
                return self._unet.decode(z, original_dims=original_dims, **kwargs)

            @property
            def latent_shape(self):
                return self._unet.latent_shape

        return AutoEncoder()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        z, encoder_outputs, original_dims = self.encode(x, include_skips=True, **kwargs)
        return self.decode(z, encoder_outputs=encoder_outputs,
                           original_dims=original_dims, **kwargs)
