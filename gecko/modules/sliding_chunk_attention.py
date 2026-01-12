from typing import Optional
import torch
from torch import Tensor, nn

from .rotary_positional_embedding import apply_rotary_emb
from .fused_ops import (
    sliding_chunk_attention,
    swift_efficient_attention,
)


def get_efficient_attention_function(attn_name: str):
    return {
        "swift": sliding_chunk_attention,
        "fused": sliding_chunk_attention
    }[attn_name]


class SlidingChunkAttention(nn.Module):
    """
    Sliding Chunk attention
    """

    def __init__(
        self,
        z_head_dim: int,
        v_head_dim: int,
        n_heads: int,
        n_rope_heads: int,
        chunk_size: int,
        dropout: float,
        efficient_attn: Optional[str],
    ):
        super().__init__()
        self.z_head_dim = z_head_dim
        self.v_head_dim = v_head_dim
        self.n_heads = n_heads
        self.n_rope_heads = n_rope_heads
        self.chunk_size = chunk_size
        self.dropout = dropout
        self.efficient_attn = efficient_attn

        # efficient attention
        self.attn_fn = get_efficient_attention_function(self.efficient_attn)
        assert self.attn_fn is not None

    def forward(
        self,
        xq: Tensor,
        xk: Tensor,
        xv: Tensor,
        freqs_cis: Optional[Tensor],
        prev_k: Optional[Tensor] = None,
        prev_v: Optional[Tensor] = None,
        segment_idx: Optional[Tensor] = None,
        prev_segment_idx: Optional[Tensor] = None,
    ):
        bs, slen, _ = xq.shape
        xq = xq.view(bs, slen, self.n_heads, self.z_head_dim)
        xk = xk.view(bs, slen, self.n_heads, self.z_head_dim)
        xv = xv.view(bs, slen, self.n_heads, self.v_head_dim)

        if self.n_rope_heads == self.n_heads:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, backward=False)
        elif self.n_rope_heads > 0:
            xq_rope, xq_nope = torch.split(xq, [self.n_rope_heads, self.n_heads - self.n_rope_heads], dim=2)
            xk_rope, xk_nope = torch.split(xk, [self.n_rope_heads, self.n_heads - self.n_rope_heads], dim=2)
            xq_rope, xk_rope = apply_rotary_emb(xq_rope, xk_rope, freqs_cis=freqs_cis, backward=False)
            xq = torch.cat([xq_rope, xq_nope], dim=2)
            xk = torch.cat([xk_rope, xk_nope], dim=2)

        if slen >= self.chunk_size:
            assert slen % self.chunk_size == 0
            output = self.attn_fn(xq, xk, xv, self.chunk_size, prev_k, prev_v, segment_idx, prev_segment_idx, 1.0, self.dropout, self.training)
            prev_k = xk[:, (slen - self.chunk_size):]
            prev_v = xv[:, (slen - self.chunk_size):]
        else:
            # tail or step-by-step generation mode
            xk = torch.cat([prev_k, xk], dim=1) if prev_k is not None else xk
            xv = torch.cat([prev_v, xv], dim=1) if prev_v is not None else xv
            assert xk.shape[1] <= self.chunk_size * 2
            if segment_idx is not None:
                q_segment_idx = segment_idx
                k_segment_idx = torch.cat([prev_segment_idx, segment_idx], dim=1) if prev_k is not None else segment_idx
            else:
                q_segment_idx = None
                k_segment_idx = None

            use_causal_mask = slen > 1
            output = swift_efficient_attention(xq, xk, xv, q_segment_idx, k_segment_idx, 1.0, self.dropout, use_causal_mask, self.training)
            prev_k = xk
            prev_v = xv

        output = output.view(bs, slen, -1)
        return output, prev_k, prev_v

    def extra_repr(self) -> str:
        return 'heads={} ({}), z_head_dim={}, v_head_dim={}, chunk={}, attn={}'.format(
            self.n_heads, self.n_rope_heads, self.z_head_dim, self.v_head_dim, self.chunk_size, self.efficient_attn
        )
