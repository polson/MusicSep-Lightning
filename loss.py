import enum
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from modules.functional import STFT, ToMagnitude, FFT2d, ToSTFT, DebugShape, Condition
from modules.seq import Seq


class LossInterface(ABC):
    @abstractmethod
    def calculate(self, prediction, target):
        pass


class LossType(enum.Enum):
    MSE = "mse"
    MASKED_MSE = "masked_mse"
    STFT_RMSE = "stft_rmse"
    MULTI_STFT = "multi_stft"
    MULTI_STFT_L1 = "multi_stft_l1"


class MSELoss(LossInterface, nn.Module):
    """Simple MSE loss."""

    def __init__(self):
        super().__init__()

    def calculate(self, prediction, target):
        return F.mse_loss(prediction, target)


class StftRmseLoss(LossInterface, nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = Seq(
            Condition(
                condition=lambda x: x.ndim == 3,
                true_fn=lambda: ToSTFT()
            )
        )

    def calculate(self, prediction, target):
        loss = F.mse_loss(prediction, target).sqrt()
        return loss


class MultiStftLoss(LossInterface, nn.Module):
    def __init__(
            self,
            fft_sizes=[4096, 2048, 1024, 512, 256],
            hop_sizes=[147, 147, 147, 147, 147]
    ):
        super().__init__()
        self.stft_losses = nn.ModuleList()

        assert len(fft_sizes) == len(hop_sizes), "FFT and Hop size lists must be equal length"

        for n_fft, hop in zip(fft_sizes, hop_sizes):
            layer = Seq(
                ToSTFT(n_fft=n_fft, hop_length=hop)
            )
            self.stft_losses.append(layer)

    def calculate(self, prediction, target):
        total_loss = 0.0
        for stft_layer in self.stft_losses:
            p_stft = stft_layer(prediction)
            t_stft = stft_layer(target)

            # MSE + Sqrt (L2 Distance)
            loss = F.mse_loss(p_stft, t_stft).sqrt()
            total_loss += loss

        return (total_loss / len(self.stft_losses))


class MultiStftL1Loss(LossInterface, nn.Module):
    """
    Aggressive separation loss.
    Uses L1 (Manhattan Distance) on Multi-Resolution STFTs.
    Encourages sparsity (removing background noise entirely).
    """

    def __init__(
            self,
            fft_sizes=[4096, 2048, 1024, 512, 256],
            hop_sizes=[147, 147, 147, 147, 147]
    ):
        super().__init__()
        self.stft_losses = nn.ModuleList()

        assert len(fft_sizes) == len(hop_sizes), "FFT and Hop size lists must be equal length"

        for n_fft, hop in zip(fft_sizes, hop_sizes):
            layer = Seq(
                Rearrange("b n c t -> (b n) c t"),
                ToSTFT(n_fft=n_fft, hop_length=hop)
            )
            self.stft_losses.append(layer)

    def calculate(self, prediction, target):
        total_loss = 0.0
        for stft_layer in self.stft_losses:
            p_stft = stft_layer(prediction)
            t_stft = stft_layer(target)

            # L1 Loss (No Sqrt needed)
            loss = F.l1_loss(p_stft, t_stft)
            total_loss += loss

        return (total_loss / len(self.stft_losses)) * 1000


class EasyMiningMSE(LossInterface, nn.Module):
    def __init__(self, q=0.5):
        super().__init__()
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
        # Convert string to Enum if passed as string from config
        if isinstance(loss_type, str):
            try:
                loss_type = LossType(loss_type)
            except ValueError:
                pass

        if loss_type == LossType.MSE:
            return MSELoss()
        elif loss_type == LossType.MASKED_MSE:
            return EasyMiningMSE()
        elif loss_type == LossType.STFT_RMSE:
            return StftRmseLoss()
        elif loss_type == LossType.MULTI_STFT:
            return MultiStftLoss()
        elif loss_type == LossType.MULTI_STFT_L1:
            return MultiStftL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
