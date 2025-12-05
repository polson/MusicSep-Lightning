import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from loss import LossFactory, LossType
from model.base_model import BaseModel, SeparationMode
from model.magsep.rwkv import BiRWKVLayer
from modules.functional import ReshapeBCFT, WithShape, CFTShape, Residual, DebugShape, Bandsplit, ToMagnitude, \
    ToMagnitudeAndInverse, FromMagnitude
from modules.seq import Seq
from modules.stft import STFT


class KLScheduler:
    def __init__(self, total_steps, cycle_count=1, ratio=0.5, start=0.0, stop=1.0):
        self.total_steps = total_steps
        self.cycle_count = cycle_count
        self.ratio = ratio
        self.start = start
        self.stop = stop

    def get_beta(self, step):
        if self.total_steps == 0:
            return self.stop

        L = self.total_steps / self.cycle_count
        tau = (step % L) / L

        if tau > self.ratio:
            return self.stop
        else:
            progress = tau / self.ratio
            return self.start + (self.stop - self.start) * progress


class VAEBottleneck(nn.Module):
    def __init__(self, in_channels, latent_channels, learnable_prior=True):
        super().__init__()
        self.latent_channels = latent_channels
        self.learnable_prior = learnable_prior

        self.latent_f = 64
        self.thing = lambda: Seq(
            Rearrange("b c f t -> b c t f"),
            nn.Linear(self.latent_f, self.latent_f),
            Rearrange("b c t f -> b c f t"),
        )

        # Encoder / Decoder
        self.to_mu = self.thing()
        self.to_logvar = self.thing()
        self.from_latent = self.thing()

        # Learnable prior
        if learnable_prior:
            self.prior_mu = nn.Parameter(torch.zeros(1, latent_channels, 1, 1))
            self.prior_logvar = nn.Parameter(torch.zeros(1, latent_channels, 1, 1))

        self._kl = torch.tensor(0.0)

    def forward(self, x):
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)

        # Reparameterization
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu

        # KL divergence
        if self.learnable_prior:
            p_logvar = self.prior_logvar
            self._kl = 0.5 * (p_logvar - logvar + (logvar.exp() + (mu - self.prior_mu) ** 2) / p_logvar.exp() - 1)
        else:
            self._kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1)
        self._kl = self._kl.sum(dim=[1, 2, 3]).mean()

        return self.from_latent(z)

    def kl_divergence(self):
        return self._kl

    def sample_prior(self, batch_size, freq_dim, time_dim, device='cuda', temperature=1.0):
        shape = (batch_size, self.latent_channels, freq_dim, time_dim)
        if self.learnable_prior:
            mu = self.prior_mu.expand(*shape)
            std = torch.exp(0.5 * self.prior_logvar).expand(*shape)
            return mu + std * torch.randn(*shape, device=device) * temperature
        return torch.randn(*shape, device=device) * temperature


class VAEModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        dropout = 0.1
        self.config = config

        self.num_instruments = len(config.training.target_sources)
        self.n_fft = config.model.n_fft
        self.hop_length = config.model.hop_length
        self.loss_factory = LossFactory.create(LossType.MULTI_STFT)

        self.kl_scheduler = KLScheduler(
            total_steps=10000,
            cycle_count=1,
            ratio=0.5,
            start=0.0,
            stop=1.0
        )

        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        self.stft = STFT(n_fft=self.n_fft, hop_length=self.hop_length)
        self.to_magnitude = ToMagnitude()
        self.from_magnitude = FromMagnitude()

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

        # PROJECT SETUP
        self.embed_dim = 64
        num_splits = 64

        num_samples = config.training.duration * 44100
        t = (num_samples + self.hop_length - 1) // self.hop_length
        self.initial_shape = CFTShape(c=2, f=self.n_fft // 2, t=t)

        initial_shape = self.initial_shape
        embed_dim = self.embed_dim

        self.bottleneck = VAEBottleneck(
            in_channels=num_splits,
            latent_channels=config.model.latent_channels,
            learnable_prior=True
        )

        self.latent_f = initial_shape.f * initial_shape.c // num_splits

        self.encoder = Seq(
            Rearrange('b c f t -> b 1 (c f) t'),  # mega-stft

            Rearrange('b c (n f) t -> b (n c) f t', n=num_splits),  # Bandsplit

            Rearrange("b c f t -> b c t f"),
            nn.Linear(self.latent_f, embed_dim),
            nn.RMSNorm(embed_dim),
            Rearrange("b c t f -> b c f t"),
            # self.rwkv(embed_dim, "(b t) c f"),
            # self.rwkv(embed_dim, "(b c) t f"),
            self.bottleneck
        )

        self.decoder = Seq(
            # self.rwkv(embed_dim, "(b t) c f"),
            # self.rwkv(embed_dim, "(b c) t f"),

            Rearrange("b c f t -> b c t f"),
            nn.Linear(embed_dim, self.latent_f),
            # nn.Linear(embed_dim, embed_dim * 2),
            # nn.GELU(),
            # nn.Linear(embed_dim * 2, self.latent_f * 2),
            # nn.GLU(dim=-1),
            Rearrange("b c t f -> b c f t"),

            Rearrange('b (n c) f t -> b c (n f) t', n=num_splits),  # Undo bandsplit

            Rearrange('b 1 (c f) t -> b c f t', f=initial_shape.f),  # undo mega-stft
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
        self.sample_visualizer = self.visualize("sample", transform=ToMagnitude(), force=True)

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
        kl_loss = self.bottleneck.kl_divergence()
        beta = self.kl_scheduler.get_beta(self.global_step.item())
        kl_weight = float(self.config.model.kl_weight)
        current_weight = kl_weight * beta
        weighted_kl = current_weight * kl_loss

        if self.training and self.global_step % 1 == 0:
            self._sample_and_visualize(device=pred.device)

        self.global_step += 1
        total_loss = recon_loss + weighted_kl

        print(f"KL Loss: {weighted_kl.item()}, recon loss: {recon_loss.item()}")
        return total_loss

    def _sample_and_visualize(self, device):
        with torch.no_grad():
            sample_wav = self.sample(duration=5.0)
            sample_stft = self.stft(sample_wav)
            self.sample_visualizer(sample_stft)

    def _samples_to_stft_time(self, num_samples):
        return (num_samples + self.hop_length - 1) // self.hop_length

    def encode_to_latent(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        self.original_length = waveform.shape[-1]
        stft = self.stft(waveform)

        # TODO: why is there an underscore here?
        z, _, original_dims = self.encoder(stft)
        mu = self.bottleneck.mu.clone()
        var = self.bottleneck.var.clone()
        return mu, var, original_dims

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        print(f"Decoding from latent shape: {z.shape}")
        z = self.bottleneck.from_latent(z)
        z = self.decoder(z)
        return self.stft.inverse(z, self.original_length)

    def sample(self, duration, batch_size=1, temperature=1.0) -> torch.Tensor:
        num_samples = int(duration * 44100)
        t = self._samples_to_stft_time(num_samples)
        f = self.embed_dim

        # Sample from the (learnable) prior
        z = self.bottleneck.sample_prior(
            batch_size=batch_size,
            freq_dim=f,
            time_dim=t,
            device='cuda',
            temperature=temperature
        )
        return self.decode_from_latent(z)

    def reconstruct(self, waveform: torch.Tensor, use_mean: bool = False) -> torch.Tensor:
        mu, var, original_dims = self.encode_to_latent(waveform)

        if use_mean:
            z = mu
        else:
            z = self.bottleneck.reparameterize(mu, var)

        return self.decode_from_latent(z)
