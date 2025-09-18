import enum
from abc import ABC, abstractmethod

import torch
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


class StftRmseLoss(LossInterface):
    def __init__(self):
        self.stft = Seq(
            Rearrange("b n c t -> (b n) c t"),
            ToSTFT()
        )

    def calculate(self, prediction, target):
        prediction = self.stft(prediction)
        target = self.stft(target)
        loss = F.mse_loss(prediction, target)
        return loss.sqrt() * 1000


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
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
