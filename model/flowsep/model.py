import math

import torch
import torch.nn as nn
from einops import repeat

from loss import LossFactory, LossType
from model.base_model import BaseModel, SeparationMode
from model.magsep.rwkv import BiRWKVLayer
from modules.adaln import GatedResidual, AdaLN, TimeEmbedding
from modules.functional import ReshapeBCFT, WithShape, CFTShape, Residual, Repeat, ComplexMask, Mask
from modules.self_attention import SelfAttention
from modules.seq import Seq
from modules.unet import UNet


class FlowSepModel(BaseModel):
    def __init__(self,
                 config,
                 ):
        super().__init__()
        dropout = 0.1

        self.num_instruments = len(config.training.target_sources)
        self.n_fft = config.model.n_fft
        self.hop_length = config.model.hop_length

        self.loss_factory = LossFactory.create(LossType.MSE)

        # Shared time embedding - computed once per forward pass
        time_emb_dim = self.n_fft // 2
        self.time_embedding = TimeEmbedding(dim=time_emb_dim)

        self.dit = lambda dim, reshape: Seq(
            ReshapeBCFT(
                reshape,
                GatedResidual(
                    AdaLN(dim, time_emb_dim=time_emb_dim, return_gate=True),
                    SelfAttention(
                        embed_dim=dim,
                        num_heads=8,
                        dropout=dropout,
                        use_rope=True
                    ),
                ),
                GatedResidual(
                    AdaLN(dim, time_emb_dim=time_emb_dim, return_gate=True),
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout),
                ),
            ),
        )

        self.rwkv = lambda dim, reshape: Seq(
            ReshapeBCFT(
                reshape,
                GatedResidual(
                    AdaLN(dim, time_emb_dim=time_emb_dim, return_gate=True),
                    BiRWKVLayer(dim),
                ),
                GatedResidual(
                    AdaLN(dim, time_emb_dim=time_emb_dim, return_gate=True),
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout),
                ),
            ),
        )

        self.rwkv_no_ada = lambda dim, reshape: Seq(
            ReshapeBCFT(
                reshape,
                Residual(
                    nn.RMSNorm(dim),
                    BiRWKVLayer(dim),
                ),
                Residual(
                    nn.RMSNorm(dim),
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
            stride=(8, 1),
            output_channels=4,
            post_downsample_fn=lambda shape: Seq(
                # Repeat(
                #     1,
                #     self.rwkv_no_ada(shape.c, "(b t) f c"),
                # ),
            ),
            bottleneck_fn=lambda shape: Seq(
                Repeat(
                    1,  # slightly deeper bottleneck
                    self.rwkv_no_ada(shape.f, "(b c) t f")
                ),
            ),
            post_upsample_fn=lambda shape: Seq(
                # Repeat(
                #     1,
                #     self.rwkv_no_ada(shape.c, "(b t) f c"),
                # ),
            ),
            # This is the critical one for stability:
            post_skip_fn=lambda shape: Seq(
                Repeat(
                    1,  # slightly deeper bottleneck
                    self.rwkv_no_ada(shape.f, "(b c) t f")
                ),
            )
        )

        embed_dim = 256
        self.model = ComplexMask(
            WithShape(
                shape=CFTShape(c=9, f=self.n_fft // 2, t=87),
                fn=lambda shape: Seq(
                    self.unet(shape=shape),
                    nn.Tanh(),
                    # Rearrange("b c f t -> b 1 (f c) t"),
                    # Bandsplit(
                    #     shape=replace(shape, c=1, f=shape.f * (shape.c)),
                    #     num_splits=64,
                    #     fn=lambda shape: Seq(
                    #         Rearrange("b c f t -> b c t f"),
                    #         nn.Linear(shape.f, embed_dim),
                    #         nn.LayerNorm(embed_dim),
                    #         # AdaLN(embed_dim),
                    #         Rearrange("b c t f -> b c f t"),
                    #
                    #         Repeat(
                    #             1,
                    #             self.rwkv_no_ada(embed_dim, "b (c t) f"),
                    #             # self.rwkv(embed_dim, "(b t) c f"),
                    #         ),
                    #
                    #         Rearrange("b c f t -> b c t f"),
                    #         # AdaLN(embed_dim),
                    #         nn.LayerNorm(embed_dim),
                    #         nn.Linear(embed_dim, shape.f * 2),
                    #         nn.GLU(dim=-1),
                    #         Rearrange("b c t f -> b c f t", c=shape.c),
                    #
                    #     )
                    # ),
                    # Rearrange("b 1 (f c) t -> b c f t", c=9),
                ),
            ),
        )

        self.model2 = Mask(
            WithShape(
                shape=CFTShape(c=9, f=self.n_fft // 2, t=87),
                fn=lambda shape: Seq(
                    self.unet(shape=shape),
                    nn.Sigmoid(),
                ),
            ),
        )

        self.complex_mask = ComplexMask()

    def get_mode(self):
        return SeparationMode.FLOW_MATCHING

    def process(self, x, mixture, t):
        original = x

        t_emb = self.time_embedding(t)
        t_emb_expanded = t_emb[:, None, :, None].expand(-1, -1, -1, x.shape[3])

        # Concatenate: [B, 4+4+1, c, t] = [B, 9, c, t]
        x1 = torch.cat([x, mixture, t_emb_expanded], dim=1)
        x1 = self.model(x1, t_emb=t_emb, mixture=mixture)

        x = x1

        x = x - original
        return x
