from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from gecko_extension.ops import (
    attention_fwd,
    attention_bwd,
    multiseg_attention_fwd
)


class SlidingChunkAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        chunk_size: int,
        prev_key_chunk: Optional[Tensor] = None,
        prev_value_chunk: Optional[Tensor] = None,
        segment_idx: Optional[Tensor] = None,
        prev_segment_idx: Optional[Tensor] = None,
        scale: float = 1.0,
        dropout: float = 0.0,
        training: bool = True
    ) -> Tensor:
        p = dropout if training else 0.0
        y, w = _efficient_sliding_chunk_attention_fwd(
            q, k, v, chunk_size, prev_key_chunk, prev_value_chunk, segment_idx, prev_segment_idx, scale, p, training
        )
        ctx.save_for_backward(q, k, v, w, prev_key_chunk, prev_value_chunk)
        # chunk_size and scale are not a torch.Tensor
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, None,
               Optional[Tensor], Optional[Tensor],
               None, None, None, None, None]:
        saved_tensors = ctx.saved_tensors
        q, k, v, w, prev_key_chunk, prev_value_chunk = saved_tensors
        chunk_size = ctx.chunk_size
        scale = ctx.scale
        q_grad, k_grad, v_grad, prev_k_grad, prev_v_grad = _efficient_sliding_chunk_attention_bwd(
            y_grad, q, k, v, w, chunk_size, prev_key_chunk, prev_value_chunk, scale
        )
        return q_grad, k_grad, v_grad, None, prev_k_grad, prev_v_grad, None, None, None, None, None


def _efficient_sliding_chunk_attention_fwd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    chunk_size: int,
    prev_key_chunk: Optional[Tensor] = None,
    prev_value_chunk: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    prev_segment_idx: Optional[Tensor] = None,
    scale: float = 1.0,
    dropout: float = 0.0,
    save_attention_matrix: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:
    bsz, seq_len, n_heads, _ = query.shape
    assert seq_len == key.shape[1] and seq_len == value.shape[1]
    assert segment_idx is None or segment_idx.shape[1] == seq_len
    assert prev_segment_idx is None or prev_segment_idx.shape[1] == chunk_size
    assert seq_len % chunk_size == 0

    nc = seq_len // chunk_size
    out = torch.empty_like(value)
    if save_attention_matrix:
        offset = 0 if prev_key_chunk is not None else 1
        attn_w = torch.empty(bsz, n_heads, chunk_size, (2 * nc - offset) * chunk_size, dtype=value.dtype, device=value.device)
    else:
        offset = None
        attn_w = None
    for c in range(nc):
        qbos = c * chunk_size
        kvbos = c * chunk_size if c == 0 else (c - 1) * chunk_size
        eos = (c + 1) * chunk_size
        q = query[:, qbos:eos]
        k = key[:, kvbos:eos]
        v = value[:, kvbos:eos]
        q_idx = None if segment_idx is None else segment_idx[:, qbos:eos]
        k_idx = None if segment_idx is None else segment_idx[:, kvbos:eos]
        if c == 0 and prev_key_chunk is not None:
            k = torch.cat([prev_key_chunk, k], dim=1)
            v = torch.cat([prev_value_chunk, v], dim=1)
            if k_idx is not None:
                k_idx = torch.cat([prev_segment_idx, k_idx], dim=1)
        # y: [bsz, chunk_size, nheads, vdim]
        # w: [bsz, nheads, chunk_size, 2*chunk_size]
        if q_idx is None:
            y, w = attention_fwd(q, k, v, scale, dropout, True)
        else:
            y, w = multiseg_attention_fwd(q, k, v, q_idx, k_idx, scale, dropout, True)

        out[:, qbos:eos] = y
        if save_attention_matrix:
            wbos = (0 if c == 0 else 2 * c - offset) * chunk_size
            weos = (2 * c + 2 - offset) * chunk_size
            attn_w[:, :, :, wbos:weos] = w

    return out, attn_w


def _efficient_sliding_chunk_attention_bwd(
    grad: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    w: Tensor,
    chunk_size: int,
    prev_key_chunk: Optional[Tensor] = None,
    prev_value_chunk: Optional[Tensor] = None,
    scale: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    seq_len = grad.shape[1]
    assert seq_len == query.shape[1] and seq_len == key.shape[1] and seq_len == value.shape[1]
    assert seq_len % chunk_size == 0
    nc = seq_len // chunk_size
    query_grad = torch.empty_like(query)
    key_grad = torch.empty_like(key)
    value_grad = torch.empty_like(value)
    prev_key_grad = torch.empty_like(prev_key_chunk) if prev_key_chunk is not None else None
    prev_value_grad = torch.empty_like(prev_value_chunk) if prev_value_chunk is not None else None
    if prev_key_chunk is None:
        splits = [chunk_size, ] + [2 * chunk_size] * (nc - 1)
    else:
        splits = [2 * chunk_size] * nc
    ws = torch.split(w, splits, dim=-1)
    for c in range(nc):
        qbos = c * chunk_size
        kvbos = c * chunk_size if c == 0 else (c - 1) * chunk_size
        eos = (c + 1) * chunk_size
        y_grad = grad[:, qbos:eos]
        q = query[:, qbos:eos]
        k = key[:, kvbos:eos]
        v = value[:, kvbos:eos]
        if c == 0 and prev_key_chunk is not None:
            k = torch.cat([prev_key_chunk, k], dim=1)
            v = torch.cat([prev_value_chunk, v], dim=1)
        w = ws[c]

        q_grad, k_grad, v_grad = attention_bwd(y_grad, q, k, v, w, scale, True)
        query_grad[:, qbos:eos] = q_grad
        curr_k_grad = k_grad[:, -chunk_size:]
        curr_v_grad = v_grad[:, -chunk_size:]
        key_grad[:, qbos:eos] = curr_k_grad
        value_grad[:, qbos:eos] = curr_v_grad

        prev_k_grad = None if k_grad.shape[1] == chunk_size else k_grad[:, :chunk_size]
        prev_v_grad = None if v_grad.shape[1] == chunk_size else v_grad[:, :chunk_size]
        if c == 0:
            prev_key_grad = prev_k_grad
            prev_value_grad = prev_v_grad
        else:
            key_grad[:, kvbos:qbos] += prev_k_grad
            value_grad[:, kvbos:qbos] += prev_v_grad

    return query_grad, key_grad, value_grad, prev_key_grad, prev_value_grad


sliding_chunk_attention = SlidingChunkAttentionFunc.apply


def sliding_chunk_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    prev_segment_idx: Optional[Tensor] = None,
    scale: float = 1.0,
    dropout: float = 0.0,
    training: bool = True,
    requires_grad: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:
    assert training or not requires_grad
    p = dropout if training else 0.0
    y, w = _efficient_sliding_chunk_attention_fwd(
        q, k, v, chunk_size, prev_k, prev_v, segment_idx, prev_segment_idx, scale, p, requires_grad
    )
    return y, w


def sliding_chunk_attention_bwd(
    grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    chunk_size: int,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    scale: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    return _efficient_sliding_chunk_attention_bwd(grad, q, k, v, w, chunk_size, prev_k, prev_v, scale)
