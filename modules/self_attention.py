import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


def _get_next_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

        self._cos_cache = None
        self._sin_cache = None
        self._cache_seq_len = 0

    def _update_cache(self, seq_len, device, dtype):
        if seq_len > self._cache_seq_len or self._cos_cache is None:
            seq_len_cache = max(seq_len, self.max_seq_len)
            t = torch.arange(seq_len_cache, device=device, dtype=dtype)

            freqs = torch.outer(t, self.inv_freq)

            cos = freqs.cos()
            sin = freqs.sin()

            self._cos_cache = cos
            self._sin_cache = sin
            self._cache_seq_len = seq_len_cache

    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.shape[2]

        self._update_cache(seq_len, q.device, q.dtype)

        cos = self._cos_cache[:seq_len]
        sin = self._sin_cache[:seq_len]

        q_rot = self._apply_rope(q, cos, sin)
        k_rot = self._apply_rope(k, cos, sin)

        return q_rot, k_rot

    def _apply_rope(self, x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x2 * cos + x1 * sin

        return torch.cat([x_rot1, x_rot2], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, max_seq_len=2048, qkv_expand_dim=None,
                 cross_attention=False, kv_embed_dim=None, use_rope=False, rope_base=10000, ):
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
        self.scale = math.sqrt(self.head_dim)

        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError(f"Head dimension ({self.head_dim}) must be even for RoPE")
            self.rope = RoPE(
                head_dim=self.head_dim,
                max_seq_len=max_seq_len,
                base=rope_base
            )

        self.query = nn.Linear(embed_dim, self.qkv_dim, bias=False)
        self.key = nn.Linear(self.kv_embed_dim, self.qkv_dim, bias=False)
        self.value = nn.Linear(self.kv_embed_dim, self.qkv_dim, bias=False)

        self.to_gates = nn.Linear(self.embed_dim, self.num_heads)

        self.out_proj = nn.Linear(self.qkv_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.dropout_p = dropout

    def forward(self, x, context=None, attn_mask=None, is_causal=False):
        batch_size, seq_len, embed_dim = x.shape

        if context is not None:
            kv_source = context
        else:
            kv_source = x

        Q = self.query(x)
        K = self.key(kv_source)
        V = self.value(kv_source)

        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        if self.use_rope:
            if context is not None:
                Q_seq_len = Q.shape[2]
                K_seq_len = K.shape[2]

                Q, _ = self.rope(Q, Q, seq_len=Q_seq_len)
                _, K = self.rope(torch.zeros_like(K), K, seq_len=K_seq_len)
            else:
                Q, K = self.rope(Q, K, seq_len=seq_len)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal and context is None,
            scale=1.0 / self.scale
        )

        gates = self.to_gates(x)
        attn_output = attn_output * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        output = rearrange(attn_output, 'b h n d -> b n (h d)')

        output = self.out_proj(output)

        output = self.dropout(output)

        return output
