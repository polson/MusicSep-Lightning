import enum
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from modules.functional import STFT, ToMagnitude, FFT2d, ToSTFT
from modules.seq import Seq


class LossInterface(ABC):
    @abstractmethod
    def calculate(self, prediction, target):
        pass


class LossType(enum.Enum):
    MASKED_MSE = "masked_mse"
    STFT_RMSE = "stft_rmse"
    MULTI_STFT = "multi_stft"  # Added Enum member


class StftRmseLoss(LossInterface, nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = Seq(
            Rearrange("b n c t -> (b n) c t"),
            ToSTFT(),
        )

    def calculate(self, prediction, target):
        prediction = self.stft(prediction)
        target = self.stft(target)
        loss = F.mse_loss(prediction, target)
        return loss.sqrt() * 1000


class MultiStftLoss(LossInterface, nn.Module):
    def __init__(
            self,
            fft_sizes=[4096, 2048, 1024, 512, 256],
            hop_sizes=[147, 147, 147, 147, 147]
    ):
        super().__init__()
        self.stft_losses = nn.ModuleList()

        # Ensure consistent list lengths
        assert len(fft_sizes) == len(hop_sizes), "FFT and Hop size lists must be equal length"

        for n_fft, hop in zip(fft_sizes, hop_sizes):
            # Create a sequence for each resolution
            layer = Seq(
                Rearrange("b n c t -> (b n) c t"),
                ToSTFT(n_fft=n_fft, hop_length=hop)
            )
            self.stft_losses.append(layer)

    def calculate(self, prediction, target):
        total_loss = 0.0

        # Accumulate loss over all resolutions
        for stft_layer in self.stft_losses:
            p_stft = stft_layer(prediction)
            t_stft = stft_layer(target)

            # Calculate RMSE for this specific resolution
            loss = F.mse_loss(p_stft, t_stft).sqrt()
            total_loss += loss

        # Average the loss across resolutions and apply scaling
        return (total_loss / len(self.stft_losses)) * 1000


class EasyMiningMSE(LossInterface):
    def __init__(self, q=0.5):
        self.q = q

    def apply_mask(self, loss):
        batch_size = loss.shape[0]
        loss_flat = loss.view(batch_size, -1)

        quantiles = torch.quantile(loss_flat.detach(), self.q, dim=1, keepdim=True)
        mask = loss_flat < quantiles

        masked_losses = []
        for i in range(batch_size):
            if mask[i].sum() == 0:
                masked_losses.append(loss_flat[i].mean())
            else:
                masked_losses.append(loss_flat[i][mask[i]].mean())

        return torch.stack(masked_losses).mean()

    def calculate(self, prediction, target):
        loss = torch.nn.MSELoss(reduction='none')(prediction, target)
        masked_loss = self.apply_mask(loss)
        return masked_loss


class LossFactory:
    @staticmethod
    def create(loss_type):
        if loss_type == LossType.MASKED_MSE:
            return EasyMiningMSE()
        elif loss_type == LossType.STFT_RMSE:
            return StftRmseLoss()
        elif loss_type == LossType.MULTI_STFT:
            return MultiStftLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
