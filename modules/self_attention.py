import math

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding


def _get_next_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, qkv_expand_dim=None,
                 cross_attention=False, kv_embed_dim=None, use_rope=False, rope_base=10000):
        super(SelfAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.cross_attention = cross_attention
        self.use_rope = use_rope

        self.kv_embed_dim = kv_embed_dim if kv_embed_dim is not None else embed_dim

        if qkv_expand_dim is None:
            self.qkv_dim = _get_next_multiple(embed_dim, num_heads)
        else:
            self.qkv_dim = qkv_expand_dim

        assert self.qkv_dim % num_heads == 0, f"qkv_dim ({self.qkv_dim}) must be divisible by num_heads ({num_heads})"
        assert self.qkv_dim >= embed_dim, f"qkv_dim ({self.qkv_dim}) must be >= embed_dim ({embed_dim})"

        self.head_dim = self.qkv_dim // num_heads

        # Ensure head_dim is even for RoPE by padding qkv_dim if necessary
        if self.use_rope and self.head_dim % 2 != 0:
            self.qkv_dim += num_heads  # Add 1 to each head dimension
            self.head_dim = self.qkv_dim // num_heads

        self.scale = math.sqrt(self.head_dim)

        self.norm = nn.RMSNorm(embed_dim)

        if self.use_rope:
            self.rotary_embed = RotaryEmbedding(
                dim=self.head_dim,
                theta=rope_base
            )

        self.query = nn.Linear(embed_dim, self.qkv_dim, bias=False)
        self.key = nn.Linear(self.kv_embed_dim, self.qkv_dim, bias=False)
        self.value = nn.Linear(self.kv_embed_dim, self.qkv_dim, bias=False)

        self.to_gates = nn.Linear(self.embed_dim, self.num_heads)

        self.out_proj = nn.Linear(self.qkv_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.dropout_p = dropout

    def forward(self, x, context=None, attn_mask=None, is_causal=False):
        x_norm = self.norm(x)
        if context is not None:
            kv_source = context
            # Normalize context as well if doing cross-attention
            kv_source_norm = self.norm(kv_source)
        else:
            kv_source_norm = x_norm

        Q = self.query(x_norm)
        K = self.key(kv_source_norm)
        V = self.value(kv_source_norm)

        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        if self.use_rope:
            Q = self.rotary_embed.rotate_queries_or_keys(Q)
            K = self.rotary_embed.rotate_queries_or_keys(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal and context is None,
            scale=1.0 / self.scale
        )

        # Apply gating using the original (non-normalized) input
        gates = self.to_gates(x)
        attn_output = attn_output * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        output = rearrange(attn_output, 'b h n d -> b n (h d)')

        output = self.out_proj(output)

        output = self.dropout(output)

        return output
