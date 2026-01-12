from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx
import torch.nn.functional as F
from .utils import logsumexp_backward


class AdaptiveWorkingMemoryFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        chunk_size: int,
        memory: Optional[Tensor],
        log_norm_term: Optional[Tensor],
        prev_k: Optional[Tensor],
        prev_v: Optional[Tensor],
        segment_idx: Optional[Tensor] = None,
        prev_segment_idx: Optional[Tensor] = None,
        prev_segment_count: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        y, mem, lnt = _adaptive_working_memory_fwd(
            q, k, v, chunk_size, memory, log_norm_term, prev_k, prev_v,
            segment_idx, prev_segment_idx, prev_segment_count
        )
        ctx.save_for_backward(
            q, k, v, memory, log_norm_term, prev_k, prev_v, segment_idx, prev_segment_idx, prev_segment_count
        )
        ctx.chunk_size = chunk_size
        return y, mem, lnt

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: Tensor,
        mem_grad: Tensor,
        lnt_grad: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, None, Tensor, Tensor,
               Optional[Tensor], Optional[Tensor], None, None, None]:
        q, k, v, memory, log_norm_term, prev_k, prev_v, segment_idx, prev_segment_idx, prev_segment_count = ctx.saved_tensors
        chunk_size = ctx.chunk_size

        q_grad, k_grad, v_grad, mem_grad, lnt_grad, prev_k_grad, prev_v_grad = _adaptive_working_memory_bwd(
            y_grad, mem_grad, lnt_grad, q, k, v, chunk_size,
            memory, log_norm_term, prev_k, prev_v,
            segment_idx, prev_segment_idx, prev_segment_count
        )

        return q_grad, k_grad, v_grad, None, mem_grad, lnt_grad, prev_k_grad, prev_v_grad, None, None, None


def _adaptive_working_memory_fwd(
    query: Tensor,  # B x H x L x S/H
    key: Tensor,  # B x H x L x S/H
    value: Tensor,  # B x H x L x V/H
    chunk_size: int,
    memory: Optional[Tensor],  # B x H x S/H x V/H
    log_norm_term: Optional[Tensor],  # B x H x S/H
    prev_key: Optional[Tensor],  # B x H x C x S/H
    prev_value: Optional[Tensor],  # B x H x C x V/H
    segment_idx: Optional[Tensor] = None,  # B x L
    prev_segment_idx: Optional[Tensor] = None,  # B x C
    prev_segment_count: Optional[Tensor] = None,  # B
):
    bsz, n_heads, seq_len, qk_head_dim = query.shape
    v_head_dim = value.shape[3]
    assert seq_len % chunk_size == 0
    nc = seq_len // chunk_size
    if memory is None:
        assert log_norm_term is None
        assert prev_key is None and prev_value is None
        assert prev_segment_idx is None
        assert prev_segment_count is None

    if segment_idx is not None:
        segment_idx = segment_idx.view(bsz, nc, chunk_size)
        # B x 1 x 1 x L
        key_mask = torch.ne(segment_idx, segment_idx[:, :, -1:]).view(bsz, 1, 1, seq_len)
        prev_k_mask = None
        # B x N
        memory_mask = torch.zeros(bsz, nc, dtype=torch.bool, device=query.device)
        torch.ne(segment_idx[:, :-2, -1], segment_idx[:, 1:-1, -1], out=memory_mask[:, 2:])
        # B x N x C
        out_mask = torch.zeros(bsz, nc, chunk_size, dtype=torch.bool, device=query.device)
        torch.ne(segment_idx[:, 2:], segment_idx[:, :-2, -1:], out=out_mask[:, 2:])
        if prev_segment_idx is not None:
            prev_k_mask = torch.ne(prev_segment_idx, prev_segment_idx[:, -1:]).view(bsz, 1, 1, chunk_size)
            torch.ne(
                torch.stack([prev_segment_count, prev_segment_idx[:, -1]], dim=1),
                torch.stack([prev_segment_idx[:, -1], segment_idx[:, 0, -1]], dim=1),
                out=memory_mask[:, :2]
            )
            torch.ne(
                segment_idx[:, :2],
                torch.stack([prev_segment_count.view(bsz, 1), prev_segment_idx[:, -1:]], dim=1),
                out=out_mask[:, :2]
            )
        out_mask = out_mask.view(bsz, seq_len)
    else:
        key_mask = None
        prev_k_mask = None
        memory_mask = None
        out_mask = None

    # B x H x L x S/H
    query = F.softmax(query, dim=-1, dtype=torch.float32).to(query)
    k_fp32 = key.float()
    qkey = F.softmax(k_fp32, dim=-1).to(key)
    # B x H x L x S/H -> B x H x S/H x L
    k_fp32 = k_fp32.transpose(2, 3).contiguous()
    if key_mask is not None:
        k_fp32.masked_fill_(key_mask, value=float("-inf"))
    # B x H x S/H x N x C
    k_fp32 = k_fp32.view(bsz, n_heads, qk_head_dim, nc, chunk_size)
    # B x H x S/H x L
    kkey = F.softmax(k_fp32, dim=-1).to(key).view(bsz, n_heads, qk_head_dim, seq_len)
    # B x H x S/H x N
    curr_log_norm_term = torch.empty(bsz, n_heads, qk_head_dim, nc, dtype=torch.float32, device=key.device)
    torch.logsumexp(k_fp32[:, :, :, :-1], dim=-1, out=curr_log_norm_term[:, :, :, 1:])

    if prev_key is not None:
        prev_k_fp32 = prev_key.float()
        prev_qkey = F.softmax(prev_k_fp32, dim=-1).to(key)
        # B x H x S/H x C
        prev_k_fp32 = prev_k_fp32.transpose(2, 3).contiguous()
        if prev_k_mask is not None:
            prev_k_fp32.masked_fill_(prev_k_mask, value=float("-inf"))
        prev_kkey = F.softmax(prev_k_fp32, dim=-1).to(key)
        torch.logsumexp(prev_k_fp32, dim=-1, out=curr_log_norm_term[:, :, :, 0])
    else:
        prev_qkey = None
        prev_kkey = None
        curr_log_norm_term[:, :, :, 0] = float("-inf")

    # B x H x L x V/H
    out = torch.empty_like(value)
    prev_qk = prev_qkey
    prev_kk = prev_kkey
    prev_v = prev_value
    for c in range(nc):
        bos = c * chunk_size
        eos = (c + 1) * chunk_size
        # B x H x C x S/H
        q = query[:, :, bos:eos]
        if memory is None:
            out[:, :, bos:eos] = 0
            memory = torch.zeros(bsz, n_heads, qk_head_dim, v_head_dim, dtype=q.dtype, device=q.device)
            log_norm_term = torch.full((bsz, n_heads, qk_head_dim), float("-inf"), device=q.device)
        else:
            # (B x H x C x S/H) x (B x H x S/H, V/H) -> B x H x C x V/H
            torch.matmul(q, memory, out=out[:, :, bos:eos])
            if memory_mask is not None:
                mmask = memory_mask[:, c]
                memory = memory.masked_fill(mmask.view(bsz, 1, 1, 1), value=0.)
                log_norm_term = log_norm_term.masked_fill(mmask.view(bsz, 1, 1), value=float("-inf"))

            # B x H x S/H
            log_norm_term_curr = curr_log_norm_term[:, :, :, c]
            log_norm_term = torch.logsumexp(torch.stack([log_norm_term_curr, log_norm_term], dim=0), dim=0)
            # B x H x S/H x 1
            ratio = torch.exp(log_norm_term_curr - log_norm_term).to(memory).unsqueeze(3)
            # B x H x C x V/H
            vv = prev_v - torch.matmul(prev_qk, memory)
            # (B x H, S/H x C) x (B x H x C x V/H) -> B x H x S/H x V/H
            memory_update = torch.matmul(prev_kk, vv)
            memory = torch.addcmul(memory, memory_update - memory, ratio)

        prev_v = value[:, :, bos:eos]
        prev_qk = qkey[:, :, bos:eos]
        prev_kk = kkey[:, :, :, bos:eos]

    # B x H x L x V/H
    if out_mask is not None:
        out.masked_fill_(out_mask.view(bsz, 1, seq_len, 1), 0.)

    return out, memory, log_norm_term


def _adaptive_working_memory_bwd(
    out_grad: Tensor,  # B x H x L x S/H
    memory_grad: Tensor,  # B x H x S/H x V/H
    log_norm_term_grad: Tensor,  # B x H x S/H
    query: Tensor,  # B x H x L x S/H
    key: Tensor,  # B x H x L x S/H
    value: Tensor,  # B x H x L x V/H
    chunk_size: int,
    memory: Optional[Tensor],  # B x H x S/H x V/H
    log_norm_term: Optional[Tensor],  # B x H x S/H
    prev_key: Optional[Tensor],  # B x H x C x S/H
    prev_value: Optional[Tensor],  # B x H x C x V/H
    segment_idx: Optional[Tensor] = None,  # B x L
    prev_segment_idx: Optional[Tensor] = None,  # B x C
    prev_segment_count: Optional[Tensor] = None,  # B
):
    bsz, n_heads, seq_len, qk_head_dim = query.shape
    v_head_dim = value.shape[3]
    assert seq_len % chunk_size == 0
    nc = seq_len // chunk_size

    if segment_idx is not None:
        segment_idx = segment_idx.view(bsz, nc, chunk_size)
        # B x 1 x L
        key_mask = torch.ne(segment_idx, segment_idx[:, :, -1:]).view(bsz, 1, seq_len)
        prev_k_mask = None
        # B x N
        memory_mask = torch.zeros(bsz, nc, dtype=torch.bool, device=query.device)
        torch.ne(segment_idx[:, :-2, -1], segment_idx[:, 1:-1, -1], out=memory_mask[:, 2:])
        # B x N x C
        out_mask = torch.zeros(bsz, nc, chunk_size, dtype=torch.bool, device=query.device)
        torch.ne(segment_idx[:, 2:], segment_idx[:, :-2, -1:], out=out_mask[:, 2:])
        if prev_segment_idx is not None:
            prev_k_mask = torch.ne(prev_segment_idx, prev_segment_idx[:, -1:]).view(bsz, 1, chunk_size)
            torch.ne(
                torch.stack([prev_segment_count, prev_segment_idx[:, -1]], dim=1),
                torch.stack([prev_segment_idx[:, -1], segment_idx[:, 0, -1]], dim=1),
                out=memory_mask[:, :2]
            )
            torch.ne(
                segment_idx[:, :2],
                torch.stack([prev_segment_count.view(bsz, 1), prev_segment_idx[:, -1:]], dim=1),
                out=out_mask[:, :2]
            )
        out_mask = out_mask.view(bsz, seq_len)
    else:
        key_mask = None
        prev_k_mask = None
        memory_mask = None
        out_mask = None

    # B x H x L x S/H
    q_fp32 = F.softmax(query, dim=-1, dtype=torch.float32)
    query = q_fp32.to(query)
    # multi-views of keys
    k_fp32 = key.float()
    qk_fp32 = F.softmax(k_fp32, dim=-1)
    qkey = qk_fp32.to(key)
    # B x H x L x S/H -> B x H x S/H x L
    k_fp32 = k_fp32.transpose(2, 3).contiguous()
    if key_mask is not None:
        k_fp32.masked_fill_(key_mask.unsqueeze(2), value=float("-inf"))
    # B x H x S/H x N x C
    k_fp32 = k_fp32.view(bsz, n_heads, qk_head_dim, nc, chunk_size)
    kk_fp32 = F.softmax(k_fp32, dim=-1)
    # B x H x S/H x L
    kkey = kk_fp32.to(key).view(bsz, n_heads, qk_head_dim, seq_len)
    # B x H x S/H x N
    curr_log_norm_term = torch.empty(bsz, n_heads, qk_head_dim, nc, dtype=torch.float32, device=key.device)
    torch.logsumexp(k_fp32[:, :, :, :-1], dim=4, out=curr_log_norm_term[:, :, :, 1:])
    accum_log_norm_term = torch.empty_like(curr_log_norm_term)
    ratio = torch.empty(nc, bsz, n_heads, qk_head_dim, 1, dtype=key.dtype, device=key.device)

    if prev_key is not None:
        prev_k_fp32 = prev_key.float()
        prev_qk_fp32 = F.softmax(prev_k_fp32, dim=-1)
        prev_qkey = prev_qk_fp32.to(key)
        # B x H x S/H x C
        prev_k_fp32 = prev_k_fp32.transpose(2, 3).contiguous()
        if prev_k_mask is not None:
            prev_k_fp32.masked_fill_(prev_k_mask.unsqueeze(2), value=float("-inf"))
        prev_kk_fp32 = F.softmax(prev_k_fp32, dim=-1)
        prev_kkey = prev_kk_fp32.to(key)
        torch.logsumexp(prev_k_fp32, dim=3, out=curr_log_norm_term[:, :, :, 0])
    else:
        prev_qkey = None
        prev_kkey = None
        prev_k_fp32 = None
        prev_qk_fp32 = None
        prev_kk_fp32 = None
        curr_log_norm_term[:, :, :, 0] = float("-inf")

    prev_qk = prev_qkey
    prev_kk = prev_kkey
    prev_v = prev_value
    vvalue = torch.zeros_like(value)
    accum_memory = torch.zeros(nc, bsz, n_heads, qk_head_dim, v_head_dim, dtype=key.dtype, device=key.device)
    memory_residual = torch.zeros_like(accum_memory)
    for c in range(nc):
        bos = c * chunk_size
        eos = (c + 1) * chunk_size
        # B x H x C x S/H
        q = query[:, :, bos:eos]
        if memory is None:
            memory = torch.zeros(bsz, n_heads, qk_head_dim, v_head_dim, dtype=q.dtype, device=q.device)
            log_norm_term = torch.full((bsz, n_heads, qk_head_dim), float("-inf"), device=q.device)
            accum_log_norm_term[:, :, :, c] = log_norm_term
        else:
            accum_memory[c] = memory
            accum_log_norm_term[:, :, :, c] = log_norm_term
            if memory_mask is not None:
                mmask = memory_mask[:, c]
                memory = memory.masked_fill(mmask.view(bsz, 1, 1, 1), value=0.)
                log_norm_term = log_norm_term.masked_fill(mmask.view(bsz, 1, 1), value=float("-inf"))

            # B x H x S/H
            log_norm_term_curr = curr_log_norm_term[:, :, :, c]
            log_norm_term = torch.logsumexp(torch.stack([log_norm_term_curr, log_norm_term], dim=0), dim=0)
            # B x H x S/H x 1
            ratio[c] = torch.exp(log_norm_term_curr - log_norm_term).to(memory).unsqueeze(3)
            # B x H x C x V/H
            vvalue[:, :, bos:eos] = prev_v - torch.matmul(prev_qk, memory)
            # (B x H, S/H x C) x (B x H x C x V/H) -> B x H x S/H x V/H
            memory_update = torch.matmul(prev_kk, vvalue[:, :, bos:eos])
            memory_residual[c] = memory_update - memory
            memory = torch.addcmul(memory, memory_residual[c], ratio[c])

        prev_v = value[:, :, bos:eos]
        prev_qk = qkey[:, :, bos:eos]
        prev_kk = kkey[:, :, :, bos:eos]

    # B x H x L x V/H
    query_grad = torch.empty_like(query)
    key_grad = torch.zeros_like(key)
    value_grad = torch.zeros_like(value)
    prev_key_grad = None
    prev_value_grad = None
    curr_log_norm_term_grad = torch.empty_like(curr_log_norm_term)

    # B x H x L x V/H
    if out_mask is not None:
        out_grad = out_grad.masked_fill(out_mask.view(bsz, 1, seq_len, 1), 0.)

    for c in reversed(range(nc)):
        pbos = (c - 1) * chunk_size
        bos = c * chunk_size
        eos = (c + 1) * chunk_size
        # B x H x C x S/H
        prev_qk = qkey[:, :, pbos:bos] if c > 0 else prev_qkey
        # B x H x S/H x C
        prev_kk = kkey[:, :, :, pbos:bos] if c > 0 else prev_kkey
        # B x H x C x V/H
        prev_v = value[:, :, pbos:bos] if c > 0 else prev_value
        vv = vvalue[:, :, bos:eos]
        q = query[:, :, bos:eos]
        og = out_grad[:, :, bos:eos]

        if prev_v is None:
            query_grad[:, :, bos:eos] = 0
            memory_grad = None
            log_norm_term_grad = None
        elif memory_grad is None:
            assert log_norm_term_grad is None
            log_norm_term = accum_log_norm_term[:, :, :, c]
            curr_log_norm_term_grad[:, :, :, c] = 0.
            log_norm_term_grad = torch.zeros_like(log_norm_term)
            memory = accum_memory[c]
            # (B x H x C x V/H) x (B x H x V/H x S/H) -> B x H x C x S/H
            q_grad = torch.matmul(og, memory.transpose(2, 3))
            query_grad[:, :, bos:eos] = torch.ops.aten._softmax_backward_data(
                q_grad.float(), q_fp32[:, :, bos:eos], -1, torch.float32
            ).to(q)
            # (B x H x S/H x C) x (B x H x C x V/H) -> B x H x S/H x V/H
            memory_grad = torch.matmul(q.transpose(2, 3), og)
        else:
            log_norm_term_prev = accum_log_norm_term[:, :, :, c]
            curr_memory = accum_memory[c]
            if memory_mask is not None:
                mmask = memory_mask[:, c]
                log_norm_term_prev = log_norm_term_prev.masked_fill(mmask.view(bsz, 1, 1), value=float("-inf"))
                curr_memory = curr_memory.masked_fill(mmask.view(bsz, 1, 1, 1), value=0.)
            # B x H x S/H
            ratio_grad = (memory_grad * memory_residual[c]).sum(dim=3).float()
            log_norm_term_curr = curr_log_norm_term[:, :, :, c]
            lnt_curr_grad = ratio_grad * ratio[c].squeeze(3)
            log_norm_term_grad = log_norm_term_grad - ratio_grad * ratio[c].squeeze(3)
            log_norm_term_grad = logsumexp_backward(
                log_norm_term_grad, torch.stack([log_norm_term_curr, log_norm_term_prev], dim=0), log_norm_term, dim=0
            )
            curr_log_norm_term_grad[:, :, :, c] = log_norm_term_grad[0] + lnt_curr_grad
            log_norm_term_grad = log_norm_term_grad[1]
            # B x H x S/H x V/H
            memory_update_grad = ratio[c] * memory_grad
            memory_grad = (1.0 - ratio[c]) * memory_grad
            # (B x H x C x S/H) x (B x H x S/H x V/H) -> B x H x C x V/H
            vv_grad = torch.matmul(prev_kk.transpose(2, 3), memory_update_grad)
            # (B x H x S/H x V/H) x (B x H x V/H x C) -> B x H x S/H x C
            prev_kk_grad = torch.matmul(memory_update_grad, vv.transpose(2, 3)).float()
            # (B x H x C x V/H) x (B x H x V/H x S/H) -> B x H x C x S/H
            prev_qk_grad = torch.matmul(vv_grad, curr_memory.transpose(2, 3)).float()
            # (B x H x S/H x C) x (B x H x C x V/H) -> B x H x S/H x V/H
            memory_grad -= torch.matmul(prev_qk.transpose(2, 3), vv_grad)
            if memory_mask is not None:
                mmask = memory_mask[:, c]
                log_norm_term_grad.masked_fill_(mmask.view(bsz, 1, 1), value=0.)
                memory_grad.masked_fill_(mmask.view(bsz, 1, 1, 1), value=0.)
            log_norm_term = accum_log_norm_term[:, :, :, c]
            memory = accum_memory[c]
            # (B x H x C x V/H) x (B x H x V/H x S/H) -> B x H x C x S/H
            q_grad = torch.matmul(og, memory.transpose(2, 3))
            query_grad[:, :, bos:eos] = torch.ops.aten._softmax_backward_data(
                q_grad.float(), q_fp32[:, :, bos:eos], -1, torch.float32
            ).to(q)
            # (B x H x S/H x C) x (B x H x C x V/H) -> B x H x S/H x V/H
            memory_grad += torch.matmul(q.transpose(2, 3), og)

            if c == 0:
                prev_value_grad = vv_grad
                prev_key_grad = (torch.ops.aten._softmax_backward_data(
                    prev_kk_grad, prev_kk_fp32, -1, torch.float32
                ).transpose(2, 3) - torch.ops.aten._softmax_backward_data(
                    prev_qk_grad, prev_qk_fp32, -1, torch.float32
                )).to(key)
            else:
                value_grad[:, :, pbos:bos] = vv_grad
                key_grad[:, :, pbos:bos] = (torch.ops.aten._softmax_backward_data(
                    prev_kk_grad, kk_fp32[:, :, :, c - 1], -1, torch.float32
                ).transpose(2, 3) - torch.ops.aten._softmax_backward_data(
                    prev_qk_grad, qk_fp32[:, :, pbos:bos], -1, torch.float32
                )).to(key)

    # B x H x S/H x (N-1) x C -> B x H x S/H x (N-1)*C -> B x H x (N-1)*C x S/H
    key_grad_log_term = logsumexp_backward(
        curr_log_norm_term_grad[:, :, :, 1:], k_fp32[:, :, :, :-1], curr_log_norm_term[:, :, :, 1:], dim=4
    ).view(bsz, n_heads, qk_head_dim, -1).transpose(2, 3).to(key)

    key_grad[:, :, :seq_len - chunk_size] += key_grad_log_term
    if key_mask is not None:
        key_grad.masked_fill_(key_mask.unsqueeze(3), value=0.)

    if prev_key is not None:
        # B x H x S/H x C -> B x H x C x S/H
        prev_key_grad += logsumexp_backward(
            curr_log_norm_term_grad[:, :, :, 0], prev_k_fp32, curr_log_norm_term[:, :, :, 0], dim=3
        ).transpose(2, 3).to(key)

        if prev_k_mask is not None:
            prev_key_grad.masked_fill_(prev_k_mask.unsqueeze(3), value=0.)

    return query_grad, key_grad, value_grad, memory_grad, log_norm_term_grad, prev_key_grad, prev_value_grad


adaptive_working_memory = AdaptiveWorkingMemoryFunc.apply


def adaptive_working_memory_fwd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    chunk_size: int,
    memory: Optional[Tensor],
    log_norm_term: Optional[Tensor],
    prev_key: Optional[Tensor],
    prev_value: Optional[Tensor],
    segment_idx: Optional[Tensor] = None,
    prev_segment_idx: Optional[Tensor] = None,
    prev_segment_count: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    return _adaptive_working_memory_fwd(
        query, key, value, chunk_size, memory, log_norm_term, prev_key, prev_value,
        segment_idx, prev_segment_idx, prev_segment_count
    )


def adaptive_working_memory_bwd(
    out_grad: Tensor,
    memory_grad: Tensor,
    log_norm_term_grad: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    chunk_size: int,
    memory: Optional[Tensor],
    log_norm_term: Optional[Tensor],
    prev_key: Optional[Tensor],
    prev_value: Optional[Tensor],
    segment_idx: Optional[Tensor] = None,
    prev_segment_idx: Optional[Tensor] = None,
    prev_segment_count: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor,
           Optional[Tensor], Optional[Tensor],
           Optional[Tensor], Optional[Tensor]]:
    return _adaptive_working_memory_bwd(
        out_grad, memory_grad, log_norm_term_grad, query, key, value, chunk_size,
        memory, log_norm_term, prev_key, prev_value, segment_idx, prev_segment_idx, prev_segment_count
    )
