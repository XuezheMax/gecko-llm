from typing import Optional
from functools import partial

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .fused_ops import adaptive_working_memory


class AdaptiveWorkingMemory(nn.Module):
    """
    Adaptive Working Memory
    """
    def __init__(
        self,
        qk_head_dim: int,
        v_head_dim: int,
        n_heads: int,
        chunk_size: int,
        query_activation: str
    ):
        super().__init__()
        self.z_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.query_activation = query_activation
        assert query_activation in ['softmax', 'silu']

    @property
    def query_act_fn(self):
        if self.query_activation == 'softmax':
            return partial(F.softmax, dim=-1, dtype=torch.float32)
        elif self.query_activation == 'silu':
            return F.silu
        else:
            raise ValueError(f"Unsupported query activation: {self.query_activation}")

    def forward(
        self,
        xq: Tensor,  # B x L x S
        xk: Tensor,  # B x L x S
        xv: Tensor,  # B x L x V
        memory: Optional[Tensor],  # B x H x S/H x V/H
        log_norm_term: Optional[Tensor],  # B x H x S/H
        prev_k: Optional[Tensor],  # B x H x C x S/H
        prev_v: Optional[Tensor],  # B x H x C x V/H
        segment_idx: Optional[Tensor] = None,  # B x L
        prev_segment_idx: Optional[Tensor] = None,  # B x C
        prev_segment_count: Optional[Tensor] = None,  # B
    ):
        bsz, seq_len, _ = xq.shape

        # B x L x H x S/H -> B x H x L x S/H
        xq = xq.view(bsz, seq_len, self.n_heads, self.z_head_dim).transpose(1, 2)
        xk = xk.view(bsz, seq_len, self.n_heads, self.z_head_dim).transpose(1, 2)
        xv = xv.view(bsz, seq_len, self.n_heads, self.v_head_dim).transpose(1, 2)

        if seq_len >= self.chunk_size:
            out, memory, log_norm_term = adaptive_working_memory(
                xq, xk, xv, self.chunk_size, memory, log_norm_term, prev_k, prev_v,
                segment_idx, prev_segment_idx, prev_segment_count
            )
            prev_k = xk[:, :, seq_len - self.chunk_size:]
            prev_v = xv[:, :, seq_len - self.chunk_size:]
        else:
            # tail or step-by-step generation mode
            xk = torch.cat([prev_k, xk], dim=2) if prev_k is not None else xk
            xv = torch.cat([prev_v, xv], dim=2) if prev_v is not None else xv
            if memory is None:
                assert xk.shape[2] <= self.chunk_size
                out = torch.zeros(bsz, self.n_heads, seq_len, self.v_head_dim, dtype=xq.dtype, device=xq.device)
            else:
                assert self.chunk_size < xk.shape[2] <= self.chunk_size * 2
                xq = F.softmax(xq, dim=-1, dtype=torch.float32).to(xq)
                # (B x H x L x S/H) x (B x H x S/H, V/H) -> B x H x L x V/H
                out = torch.matmul(xq, memory)
                if segment_idx is not None:
                    out_mask = torch.ne(segment_idx, prev_segment_count.view(bsz, 1))
                    out = torch.masked_fill(out, out_mask.view(bsz, 1, seq_len, 1), value=0.)
            # update memory
            if xk.shape[2] % self.chunk_size == 0:
                memory, log_norm_term = self.update_memory(
                    xk, xv, memory, log_norm_term, segment_idx, prev_segment_idx, prev_segment_count
                )
            prev_k = xk
            prev_v = xv

        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return out, memory, log_norm_term, prev_k, prev_v

    def update_memory(self, key, value, memory, log_norm_term, segment_idx, prev_segment_idx, prev_segment_count):
        bsz = key.shape[0]
        if memory is None:
            assert key.shape[2] == self.chunk_size
            memory = torch.zeros(bsz, self.n_heads, self.z_head_dim, self.v_head_dim, dtype=key.dtype, device=key.device)
            log_norm_term = torch.full((bsz, self.n_heads, self.z_head_dim), float("-inf"), device=key.device)
        else:
            assert key.shape[2] == self.chunk_size * 2
            prev_v = value[:, :, :self.chunk_size]
            prev_k = key[:, :, :self.chunk_size]
            prev_k_fp32 = prev_k.float()
            prev_qk = F.softmax(prev_k_fp32, dim=-1).to(key)
            # B x H x S/H x C
            prev_k_fp32 = prev_k_fp32.transpose(2, 3).contiguous()
            if segment_idx is not None:
                prev_segment_idx = torch.cat([prev_segment_idx, segment_idx], dim=1)[:, :self.chunk_size]
                prev_k_mask = torch.ne(prev_segment_idx, prev_segment_idx[:, -1:]).view(bsz, 1, 1, self.chunk_size)
                prev_k_fp32.masked_fill_(prev_k_mask, value=float("-inf"))
                memory_mask = torch.ne(prev_segment_idx[:, -1], prev_segment_count)
                memory = memory.masked_fill(memory_mask.view(bsz, 1, 1, 1), value=0.)
                log_norm_term = log_norm_term.masked_fill(memory_mask.view(bsz, 1, 1), value=float("-inf"))
            prev_kk = F.softmax(prev_k_fp32, dim=-1).to(key)
            curr_log_norm_term = torch.logsumexp(prev_k_fp32, dim=-1)
            log_norm_term = torch.logsumexp(torch.stack([curr_log_norm_term, log_norm_term], dim=0), dim=0)
            ratio = torch.exp(curr_log_norm_term - log_norm_term).to(memory).unsqueeze(3)
            vv = prev_v - torch.matmul(prev_qk, memory)
            memory_update = torch.matmul(prev_kk, vv)
            memory = torch.addcmul(memory, memory_update - memory, ratio)

        return memory, log_norm_term

    def extra_repr(self) -> str:
        return 'heads={}, qk_head_dim={}, v_head_dim={}, chunk={}'.format(
            self.n_heads, self.z_head_dim, self.v_head_dim, self.chunk_size
        )
