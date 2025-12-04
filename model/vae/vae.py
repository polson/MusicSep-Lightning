import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from model.base_model import BaseModel, SeparationMode
from model.magsep.rwkv import BiRWKVLayer
from modules.functional import ReshapeBCFT, WithShape, CFTShape, Residual, Repeat, ComplexMask, ToSTFT, InverseSTFT, \
    Mask, DebugShape, Bandsplit
from modules.seq import Seq
from modules.stft import STFT
from modules.unet import UNet


class VAEBottleneck(nn.Module):
    """VAE bottleneck with reparameterization trick."""

    def __init__(self, in_channels, latent_channels):
        super().__init__()
        self.to_mu = nn.Conv2d(in_channels, latent_channels, 1)
        self.to_logvar = nn.Conv2d(in_channels, latent_channels, 1)
        self.from_latent = nn.Conv2d(latent_channels, in_channels, 1)

        # Critical: Initialize logvar to produce small initial variances
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, -2.0)

        # Small init for other layers too
        nn.init.xavier_uniform_(self.to_mu.weight, gain=0.1)
        nn.init.xavier_uniform_(self.from_latent.weight, gain=0.1)

        self.mu = None
        self.logvar = None

        # Clamping bounds
        self.logvar_min = -20.0
        self.logvar_max = 10.0

    def reparameterize(self, mu, logvar):
        if self.training:
            # Clamp to prevent extreme values
            logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(self, x):
        self.mu = self.to_mu(x)
        self.logvar = self.to_logvar(x)
        # Clamp logvar immediately
        self.logvar = torch.clamp(self.logvar, self.logvar_min, self.logvar_max)
        z = self.reparameterize(self.mu, self.logvar)
        return self.from_latent(z)

    def kl_divergence(self, device=None):
        """KL(q(z|x) || p(z)) where p(z) = N(0,1)"""
        if self.mu is None or self.logvar is None:
            if device is None:
                device = next(self.parameters()).device
            return torch.zeros(1, device=device).squeeze()

        # logvar is already clamped in forward()
        kl = -0.5 * torch.mean(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())

        # Safety check
        if torch.isnan(kl) or torch.isinf(kl):
            return torch.zeros(1, device=self.mu.device).squeeze()

        return kl


class VAEModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        dropout = 0.1

        self.num_instruments = len(config.training.target_sources)
        self.n_fft = config.model.n_fft
        self.hop_length = config.model.hop_length
        self.loss_factory = LossFactory.create(LossType.MULTI_STFT)

        # VAE hyperparameters
        self.latent_channels = config.model.latent_channels
        self.kl_weight = float(config.model.kl_weight)

        # VAE bottleneck - will be set in unet construction
        self.vae_bottleneck = None

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
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout),
                ),
            ),
        )

        self.unet = lambda shape: UNet(
            input_shape=shape,
            channels=[shape.c, 128, 256],
            stride=(4, 4),
            output_channels=shape.c * self.num_instruments,
            post_downsample_fn=lambda shape: Seq(),
            bottleneck_fn=lambda shape: self._make_vae_bottleneck(shape),
            post_upsample_fn=lambda shape: Seq(),
            post_skip_fn=lambda shape: Seq()
        )

        self.model = Seq(
            WithShape(
                shape=CFTShape(c=4, f=self.n_fft // 2, t=87),
                fn=lambda shape: Seq(
                    Bandsplit(
                        shape=shape,
                        num_splits=1,
                        fn=lambda shape: Seq(
                            self.unet(shape=shape),
                        )
                    ),
                    self.visualize("mask")
                ),
            ),
        )

        self.stft = STFT(n_fft=self.n_fft, hop_length=self.hop_length)
        self.original_length = None

    def _make_vae_bottleneck(self, shape):
        """Create and store VAE bottleneck for the UNet."""
        # shape.c is the channel dimension at the bottleneck
        self.vae_bottleneck = VAEBottleneck(shape.c, self.latent_channels)
        return self.vae_bottleneck

    def get_mode(self):
        return SeparationMode.ONE_SHOT

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        self.original_length = waveform.shape[-1]
        return self.stft(waveform)

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.stft.inverse(encoded, self.original_length)

    def process(self, x, mixture, t):
        return self.model(x)

    def loss(self, pred, targets, mixture):
        pred = self.decode(pred)
        targets = self.decode(targets)

        recon_loss = self.loss_factory.calculate(pred, targets)

        # Debug prints
        if torch.isnan(recon_loss):
            print(f"NaN in recon_loss! pred range: [{pred.min():.3f}, {pred.max():.3f}]")

        kl_loss = torch.zeros(1, device=pred.device).squeeze()
        if self.vae_bottleneck is not None:
            kl_loss = self.vae_bottleneck.kl_divergence(device=pred.device)
            if torch.isnan(kl_loss):
                print(f"NaN in KL! mu: [{self.vae_bottleneck.mu.min():.3f}, {self.vae_bottleneck.mu.max():.3f}], "
                      f"logvar: [{self.vae_bottleneck.logvar.min():.3f}, {self.vae_bottleneck.logvar.max():.3f}]")

        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss

    def sample(self, shape, device='cuda'):
        """Sample from the prior p(z) = N(0,1) and decode."""
        # You'll need to implement this based on your decoder structure
        # This is a placeholder showing the concept
        z = torch.randn(shape, device=device)
        # Route through decoder portion of UNet...
        raise NotImplementedError("Sampling requires decoder-only forward pass")
