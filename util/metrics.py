import torch


def sdr(targets: torch.Tensor, predictions: torch.Tensor, use_mean: bool = True) -> float:
    eps: float = 1e-8
    targets = targets.float()
    predictions = predictions.float()

    if use_mean:
        predictions = torch.mean(predictions, dim=0, keepdim=True)
        targets = torch.mean(targets, dim=0, keepdim=True)

    num = torch.sum(torch.square(targets), dim=(1, 2)) + eps
    den = torch.sum(torch.square(targets - predictions), dim=(1, 2)) + eps

    sdr_per_batch = 10 * torch.log10(num / den)
    return torch.mean(sdr_per_batch).item()
