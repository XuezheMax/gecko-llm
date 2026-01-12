#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from .rms_norm import GroupRMSNorm
from .timestep_decay_norm import TimestepDecayNorm
from .complex_exponential_moving_average import MultiHeadComplexEMA
from .linear import Linear
from .sliding_chunk_attention import SlidingChunkAttention
from .adaptive_working_memory import AdaptiveWorkingMemory
from .fused_ops import memory_efficient_dropout
from gecko.utils import get_init_fn

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


class MovingAverageGatedAttention(nn.Module):
    """Exponential Moving Average Gated Attention.
    See "" for more details.
    """

    def __init__(
        self,
        mdim: int,
        zdim: int,
        hdim: int,
        ndim: int,
        num_heads: int,
        num_rope_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        chunk_size: int = 2048,
        efficient_attn: Optional[str] = None,
        timenorm_num_groups: Optional[int] = None,
        timenorm_beta1: float = 0.999,
        timenorm_beta2: float = 0.9999,
        rmsnorm_num_groups: int = 1,
        norm_affine: bool = True,
        norm_eps: float = 1e-5,
        memory_efficient_norm: bool = False,
        init_mode: str = 'he',
    ):
        super().__init__()

        self.mdim = mdim
        self.hdim = hdim
        self.zdim = zdim
        self.ndim = ndim
        self.num_heads = num_heads
        self.num_rope_heads = num_rope_heads
        assert 0 <= self.num_rope_heads <= self.num_heads

        self.init_mode = init_mode
        assert zdim % num_heads == 0 and hdim % num_heads == 0
        self.z_head_dim = zdim // num_heads
        self.v_head_dim = hdim // num_heads

        self.chunk_size = chunk_size
        self.efficient_attn = efficient_attn
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        self.timenorm = TimestepDecayNorm(
            mdim, timenorm_num_groups, timenorm_beta1, timenorm_beta2, eps=norm_eps, memory_efficient=memory_efficient_norm
        )
        self.cema = MultiHeadComplexEMA(mdim, ndim)
        self.rmsnorm = GroupRMSNorm(
            mdim,
            num_groups=rmsnorm_num_groups,
            elementwise_affine=norm_affine,
            eps=norm_eps,
            memory_efficient=False,
        )
        self.znorm = GroupRMSNorm(
            self.zdim,
            num_groups=self.num_heads,
            elementwise_affine=False,
            eps=norm_eps,
            memory_efficient=True,  # for znorm, always use efficient memory
        )

        init_fn = get_init_fn(init_mode)
        self.wv = Linear(
            mdim,
            hdim,
            bias=True,
            init_method=init_fn
        )
        self.wz = Linear(
            mdim,
            zdim,
            bias=True,
            init_method=init_fn
        )
        self.wr = Linear(
            mdim,
            hdim,
            bias=True,
            init_method=init_fn
        )
        self.wh1 = Linear(
            mdim,
            mdim,
            bias=True,
            init_method=init_fn
        )
        self.wh2 = Linear(
            hdim,
            mdim,
            bias=False,
            init_method=init_fn
        )
        self.sliding_chunk_attention = SlidingChunkAttention(
            self.z_head_dim,
            self.v_head_dim,
            self.num_heads,
            self.num_rope_heads,
            self.chunk_size,
            self.attention_dropout,
            efficient_attn
        )
        self.adaptive_working_memory = AdaptiveWorkingMemory(
            self.z_head_dim,
            self.v_head_dim,
            self.num_heads,
            self.chunk_size,
            'softmax'
        )
        self.gamma = Parameter(torch.zeros(4, self.z_head_dim * self.num_heads))
        self.beta = Parameter(torch.zeros(4, self.z_head_dim * self.num_heads))

    def forward(
        self,
        x: Tensor,
        freqs_cis: Optional[Tensor],
        bos_mask: Optional[Tensor] = None,
        segment_idx: Optional[Tensor] = None,
        prev_segment_idx: Optional[Tensor] = None,
        prev_segment_count: Optional[Tensor] = None,
        cache: Optional[Tuple[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, Tensor],
                              Tuple[Tensor, Tensor, Tensor], Tensor]] = None,
    ):
        bsz, seq_len, _ = x.size()
        residual = x

        if cache is not None:
            cache_sca, cache_awk, cache_norm, hx = cache
            prev_count, prev_mean, prev_var = cache_norm
            prev_sca_key, prev_value, kv_count = cache_sca
            prev_awk_key, memory, log_norm_term = cache_awk
        else:
            prev_count, prev_mean, prev_var = None, None, None
            prev_sca_key, prev_value, kv_count = None, None, None
            prev_awk_key, memory, log_norm_term = None, None, None
            cache_sca, cache_awk, cache_norm, hx = None, None, None, None

        # B x L x D
        out_tsn, prev_count, prev_mean, prev_var = self.timenorm(x, bos_mask, prev_count, prev_mean, prev_var)
        # B x D x L
        out_cema, hx = self.cema(out_tsn.transpose(1, 2), hx, bos_mask)

        # B x D x L -> B x L x D
        mx = self.rmsnorm(out_cema.transpose(1, 2))
        mx = memory_efficient_dropout(mx, self.hidden_dropout, self.training)

        # B x L x S
        z = self.wz(mx)
        # B x L x H x S/H -> B x L x S
        z = self.znorm(z)
        # B x L x S -> B x L x 1 x S -> B x L x 2 x S
        gamma = (self.gamma + 1.0) / math.sqrt(self.z_head_dim)
        z = z.unsqueeze(2) * gamma + self.beta
        # B x L x 4 x S -> B x L x S
        q, k, aq, ak = torch.unbind(z, dim=2)

        # B x L x E
        v = F.silu(self.wv(out_tsn))
        r = F.silu(self.wr(mx))

        prev_sca_value = prev_value
        prev_awk_value = prev_value.transpose(1, 2) if prev_value is not None else None
        # B x L x E
        sca, prev_sca_key, prev_value = self.sliding_chunk_attention(
            q, k, v, freqs_cis, prev_sca_key, prev_sca_value, segment_idx, prev_segment_idx
        )
        awk, memory, log_norm_term, prev_awk_key, _ = self.adaptive_working_memory(
            aq, ak, v, memory, log_norm_term, prev_awk_key, prev_awk_value,
            segment_idx, prev_segment_idx, prev_segment_count
        )
        attn = sca * r + awk
        attn = memory_efficient_dropout(attn, self.hidden_dropout, self.training)

        if cache is not None:
            cache_norm = (prev_count.detach(), prev_mean.detach(), prev_var.detach())
            hx = None if hx is None else hx.detach()
            prev_sca_key = prev_sca_key.detach()
            prev_value = prev_value.detach()
            prev_awk_key = prev_awk_key.detach()
            memory = memory.detach() if memory is not None else memory
            log_norm_term = log_norm_term.detach() if log_norm_term is not None else log_norm_term
            kv_count = kv_count + seq_len
            if kv_count % self.chunk_size == 0:
                prev_sca_key = prev_sca_key[:, -self.chunk_size:]
                prev_value = prev_value[:, -self.chunk_size:]
                prev_awk_key = prev_awk_key[:, :, -self.chunk_size:]
            cache_sca = (prev_sca_key, prev_value, kv_count)
            cache_awk = (prev_awk_key, memory, log_norm_term)

        # B x L x E -> B x L x D
        h = self.wh1(mx) + self.wh2(attn)
        h = memory_efficient_dropout(h, self.dropout, self.training)

        # B x L x D
        out = h + residual

        if cache is not None:
            cache = (cache_sca, cache_awk, cache_norm, hx)

        return out, cache

    def extra_repr(self) -> str:
        return 'edim={}, zdim={}, hdim={}, heads={} ({}), chunk={}, eff_attn={}, init={}'.format(
            self.mdim, self.zdim, self.hdim, self.num_heads, self.num_rope_heads, self.chunk_size, self.efficient_attn, self.init_mode
        )
