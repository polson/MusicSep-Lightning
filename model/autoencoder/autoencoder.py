import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from model.base_model import BaseModel, SeparationMode
from model.magsep.rwkv import BiRWKVLayer
from modules.functional import ReshapeBCFT, WithShape, CFTShape, Residual, ToMagnitude, FromMagnitude
from modules.seq import Seq
from modules.stft import STFT


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

        self.embed_dim = 128
        num_splits = 64

        num_samples = config.training.duration * 44100
        t = (num_samples + self.hop_length - 1) // self.hop_length
        self.initial_shape = CFTShape(c=4, f=self.n_fft // 2, t=t)

        initial_shape = self.initial_shape
        embed_dim = self.embed_dim
        self.latent_f = initial_shape.f * initial_shape.c // num_splits

        self.rwkv = lambda dim, reshape: Seq(
            ReshapeBCFT(
                reshape,
                Residual(
                    nn.LayerNorm(dim),
                    BiRWKVLayer(dim),
                ),
                Residual(
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(0.1),
                ),
            ),
        )

        self.encoder = Seq(
            Rearrange('b c f t -> b 1 (c f) t'),
            Rearrange('b c (n f) t -> b (n c) f t', n=num_splits),
            Rearrange("b c f t -> b c t f"),
            nn.Linear(self.latent_f, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.SiLU(),
            Rearrange("b c t f -> b c f t"),
        )

        self.decoder = Seq(
            Rearrange("b c f t -> b c t f"),
            nn.Linear(embed_dim, self.latent_f * 2),
            nn.GLU(dim=-1),
            Rearrange("b c t f -> b c f t"),
            Rearrange('b (n c) f t -> b c (n f) t', n=num_splits),
            Rearrange('b 1 (c f) t -> b c f t', f=initial_shape.f),
        )

        self.model = Seq(
            WithShape(
                shape=initial_shape,
                fn=lambda shape: Seq(
                    self.encoder,
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
        # mag = self.to_magnitude(stft)
        return stft

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        # stft = self.from_magnitude(encoded, self.original_complex)
        waveform = self.stft.inverse(encoded, self.original_length)
        return waveform

    def process(self, x, mixture, t):
        return self.model(x)

    def additivity_loss(self, x):
        """Split batch in half, treat as two sources"""
        b = x.shape[0]
        if b < 2:
            return 0.0

        mid = b // 2
        x1 = x[:mid]
        x2 = x[mid:mid * 2]  # ensure same size if odd batch

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z_sum = self.encoder(x1 + x2)

        return torch.nn.functional.mse_loss(z1 + z2, z_sum)

    def loss(self, pred, targets, mixture):
        pred_decoded = self.decode(pred)
        targets_decoded = self.decode(targets)
        recon_loss = self.loss_factory.calculate(pred_decoded, targets_decoded)

        # Additivity constraint on the encoded targets
        add_loss = self.additivity_loss(targets)

        print(f"Reconstruction Loss: {recon_loss.item():.6f}, Additivity Loss: {add_loss.item():.6f}")

        return recon_loss + 0.1 * add_loss  # tune the weight

    def encode_to_latent(self, waveform: torch.Tensor) -> torch.Tensor:
        self.original_length = waveform.shape[-1]
        stft = self.stft(waveform)
        z = self.encoder(stft)
        return z

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        z = self.decoder(z)
        return self.stft.inverse(z, self.original_length)

    def reconstruct(self, waveform: torch.Tensor) -> torch.Tensor:
        z = self.encode_to_latent(waveform)
        return self.decode_from_latent(z)
