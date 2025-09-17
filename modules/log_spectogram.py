import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogSpectrogramProcessor(nn.Module):
    def __init__(self, in_freqs: int, out_freqs: int, process_fn: nn.Module):
        super().__init__()
        self.in_freqs = in_freqs
        self.out_freqs = out_freqs
        self.process_fn = process_fn

        log_indices = torch.logspace(start=0, end=math.log10(in_freqs), steps=out_freqs, dtype=torch.float32) - 1

        forward_y_coords = (2 * log_indices / (in_freqs - 1)) - 1
        self.register_buffer('forward_y_coords', forward_y_coords)

        linear_indices = torch.arange(in_freqs, dtype=torch.float32)
        j = torch.searchsorted(log_indices, linear_indices).clamp(1, len(log_indices) - 1)

        x1 = log_indices[j - 1]
        x2 = log_indices[j]

        denom = x2 - x1
        fraction = (linear_indices - x1) / denom.where(denom != 0, torch.tensor(1.0))

        interp_log_indices = (j - 1) + fraction
        interp_log_indices = torch.nan_to_num(interp_log_indices)

        inverse_y_coords = (2 * interp_log_indices / (out_freqs - 1)) - 1
        self.register_buffer('inverse_y_coords', inverse_y_coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape

        if f != self.in_freqs:
            raise ValueError(
                f"Input frequency dimension ({f}) does not match the module's "
                f"expected in_freqs ({self.in_freqs})."
            )

        x_coords_log = torch.linspace(-1, 1, t, device=x.device)
        grid_y_log = self.forward_y_coords.view(1, -1, 1).expand(b, -1, t)
        grid_x_log = x_coords_log.view(1, 1, -1).expand(b, self.out_freqs, -1)
        forward_grid = torch.stack([grid_x_log, grid_y_log], dim=-1)

        log_spec = F.grid_sample(
            input=x,
            grid=forward_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        processed_log_spec = self.process_fn(log_spec)

        if processed_log_spec.shape[2] != self.out_freqs or processed_log_spec.shape[3] != t:
            raise ValueError(
                "The process_fn must not change the frequency or time dimensions of the tensor."
            )

        x_coords_linear = torch.linspace(-1, 1, t, device=x.device)
        grid_y_linear = self.inverse_y_coords.view(1, -1, 1).expand(b, -1, t)
        grid_x_linear = x_coords_linear.view(1, 1, -1).expand(b, self.in_freqs, -1)
        inverse_grid = torch.stack([grid_x_linear, grid_y_linear], dim=-1)

        restored_spec = F.grid_sample(
            input=processed_log_spec,
            grid=inverse_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return restored_spec
