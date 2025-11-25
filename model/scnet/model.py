# from dataclasses import replace
#
# import numpy as np
# import torch.nn as nn
# from einops.layers.torch import Rearrange
#
# from model.base_model import BaseModel
# from modules.functional import STFTAndInverse, Residual, ReshapeBCFT, Repeat, \
#     Mask, SplitNTensor, RepeatWithArgs, ComplexMask
# from modules.self_attention import SelfAttention
# from modules.seq import Seq
#
#
# # @torch.compile()
# class SCNet(BaseModel):
#
#     def __init__(self,
#                  num_instruments=1,
#                  n_fft=2048,
#                  hop_length=512,
#                  layers=1,
#                  mask_layers=1,
#                  dropout=0.1,
#                  embed_dim=64
#                  ):
#         super().__init__()
#
#         self.bottleneck = lambda dim: Seq(
#             Residual(
#                 nn.RMSNorm(dim),  # TODO: groupnorm
#                 nn.LSTM(dim, 128, num_layers=1, batch_first=True, bidirectional=True),
#                 nn.Linear(128 * 2, dim),
#             ),
#         )
#
#         self.model = Seq(
#             STFTAndInverse(
#                 in_channels=2,
#                 n_fft=n_fft,
#                 hop_length=hop_length,
#                 fn=lambda shape:
#                 Residual(
#                     # Split frequency dim at specified points and apply embedding layer
#                     SplitNTensor(
#                         shape,
#                         fns=[self.embed] * num_splits,
#                         split_points=freqs_per_bands_cumsum[:-1],
#                         dim=2,  # Split on frequency dim
#                         concat_dim=1  # Concat on channel dim
#                     ),
#
#                     # We now have shape (b, num_splits, embed_dim, t)
#                     Repeat(
#                         layers,
#                         ReshapeBCFT(
#                             "(b c) t f",
#                             self.transformer(dim=embed_dim)
#                         ),
#                         ReshapeBCFT(
#                             "(b t) c f",
#                             self.transformer(dim=embed_dim)
#                         ),
#                     ),
#
#                     ReshapeBCFT("b c t f", nn.RMSNorm(embed_dim)),
#
#                     # Split along channel dimension and apply mask layer for each band
#                     SplitNTensor(
#                         replace(shape, c=num_splits, f=embed_dim),
#                         fns=[self.masking] * num_splits,
#                         split_points=list(range(1, num_splits)),  # [1, 2, 3, ..., num_splits-1]
#                         dim=1,  # Split on channel dim
#                         concat_dim=2  # Concat on frequency dim
#                     ),
#                     self.visualize("Mask")
#                 ),
#             ),
#             Rearrange("b (n c) t -> b n c t", n=num_instruments)
#         )
#
#     def process(self, x):
#         return self.model(x)
