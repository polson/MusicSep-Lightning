import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from model.base_model import BaseModel, SeparationMode
from modules.functional import ReshapeBCFT, WithShape, CFTShape, Residual, ToMagnitude, FromMagnitude
from modules.seq import Seq
from modules.stft import STFT


class Bottleneck(nn.Module):
    def __init__(self, latent_f):
        super().__init__()
        self.latent_f = latent_f
        self.compress = Seq(
            Rearrange("b c f t -> b c t f"),
            nn.Linear(self.latent_f, self.latent_f),
            Rearrange("b c t f -> b c f t"),
        )
        self.expand = Seq(
            Rearrange("b c f t -> b c t f"),
            nn.Linear(self.latent_f, self.latent_f),
            Rearrange("b c t f -> b c f t"),
        )

    def forward(self, x):
        z = self.compress(x)
        return self.expand(z)

    def encode(self, x):
        return self.compress(x)

    def decode(self, z):
        return self.expand(z)


class AutoencoderModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_instruments = len(config.training.target_sources)
        self.n_fft = config.model.n_fft
        self.hop_length = config.model.hop_length
        self.loss_factory = LossFactory.create(LossType.MULTI_STFT)

        self.stft = STFT(n_fft=self.n_fft, hop_length=self.hop_length)
        self.to_magnitude = ToMagnitude()
        self.from_magnitude = FromMagnitude()

        self.embed_dim = 64
        num_splits = 64

        num_samples = config.training.duration * 44100
        t = (num_samples + self.hop_length - 1) // self.hop_length
        self.initial_shape = CFTShape(c=2, f=self.n_fft // 2, t=t)

        initial_shape = self.initial_shape
        embed_dim = self.embed_dim
        self.latent_f = initial_shape.f * initial_shape.c // num_splits

        self.bottleneck = Bottleneck(latent_f=embed_dim)

        self.encoder = Seq(
            Rearrange('b c f t -> b 1 (c f) t'),
            Rearrange('b c (n f) t -> b (n c) f t', n=num_splits),
            Rearrange("b c f t -> b c t f"),
            nn.Linear(self.latent_f, embed_dim),
            nn.RMSNorm(embed_dim),
            Rearrange("b c t f -> b c f t"),
        )

        self.decoder = Seq(
            Rearrange("b c f t -> b c t f"),
            nn.Linear(embed_dim, self.latent_f),
            Rearrange("b c t f -> b c f t"),
            Rearrange('b (n c) f t -> b c (n f) t', n=num_splits),
            Rearrange('b 1 (c f) t -> b c f t', f=initial_shape.f),
        )

        self.model = Seq(
            WithShape(
                shape=initial_shape,
                fn=lambda shape: Seq(
                    self.encoder,
                    self.bottleneck,
                    self.decoder,
                    self.visualize("mask"),
                )
            ),
        )

        self.original_length = None
        self.original_complex = None

    def get_mode(self):
        return SeparationMode.ONE_SHOT

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        self.original_length = waveform.shape[-1]
        stft = self.stft(waveform)
        self.original_complex = stft
        mag = self.to_magnitude(stft)
        return mag

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        stft = self.from_magnitude(encoded, self.original_complex)
        waveform = self.stft.inverse(stft, self.original_length)
        return waveform

    def process(self, x, mixture, t):
        return self.model(x)

    def loss(self, pred, targets, mixture):
        pred = self.decode(pred)
        targets = self.decode(targets)
        recon_loss = self.loss_factory.calculate(pred, targets)
        return recon_loss

    def encode_to_latent(self, waveform: torch.Tensor) -> torch.Tensor:
        self.original_length = waveform.shape[-1]
        stft = self.stft(waveform)
        x = self.encoder(stft)
        z = self.bottleneck.encode(x)
        return z

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck.decode(z)
        x = self.decoder(x)
        return self.stft.inverse(x, self.original_length)

    def reconstruct(self, waveform: torch.Tensor) -> torch.Tensor:
        z = self.encode_to_latent(waveform)
        return self.decode_from_latent(z)