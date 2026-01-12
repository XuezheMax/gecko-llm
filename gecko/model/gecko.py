from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import torch
from torch import Tensor
from torch import nn

from tokenizer import Tokenizer
from gecko.modules import (
    Linear,
    MovingAverageGatedAttention,
    NormalizedFeedForwardNetwork,
    RotaryEmbedding,
    TimestepDecayNorm,
)
from gecko.modules.fused_ops import memory_efficient_dropout
from gecko.utils import get_init_fn


@dataclass
class ModelConf:
    num_layers: int = 32
    model_dim: int = 4096
    z_dim: int = 1024
    value_dim: int = 8192
    num_heads: int = 4
    ffn_hidden_dim: int = 8192
    cema_ndim: int = 16
    chunk_size: int = 2048
    efficient_attn: str = "fused"
    init_mode: str = 'he'
    # input & output
    vocab_size: int = -1  # defined later by tokenizer
    output_size: int = -1
    # normalization
    timenorm_num_groups: int = 64
    timenorm_beta1: float = 0.999
    timenorm_beta2: float = 0.9999
    layernorm_num_groups: int = 1
    memory_efficient_norm: bool = False
    norm_affine: bool = True
    norm_eps: float = 1e-5
    two_hop_residual: bool = False
    # rope base
    rope_base: float = 500000
    num_rope_heads: Optional[int] = None
    # dropout rates
    dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    #
    swiglu: bool = True
    rescale_nffn: bool = False
    scale_emb: bool = False
    share_emb: bool = False


class GeckoBlock(nn.Module):
    def __init__(self, cfg: ModelConf, layer_id: int):
        super().__init__()

        self.layer_id = layer_id
        self.two_hop_residual = cfg.two_hop_residual
        self.memory_efficient_norm = cfg.memory_efficient_norm

        self.mega = MovingAverageGatedAttention(
            mdim=cfg.model_dim,
            zdim=cfg.z_dim,
            hdim=cfg.value_dim,
            num_heads=cfg.num_heads,
            num_rope_heads=cfg.num_heads if cfg.num_rope_heads is None else cfg.num_rope_heads,
            ndim=cfg.cema_ndim,
            chunk_size=cfg.chunk_size,
            efficient_attn=cfg.efficient_attn,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            hidden_dropout=cfg.hidden_dropout,
            timenorm_num_groups=cfg.timenorm_num_groups,
            timenorm_beta1=cfg.timenorm_beta1,
            timenorm_beta2=cfg.timenorm_beta2,
            rmsnorm_num_groups=cfg.layernorm_num_groups,
            norm_affine=cfg.norm_affine,
            norm_eps=cfg.norm_eps,
            memory_efficient_norm=cfg.memory_efficient_norm,
            init_mode=cfg.init_mode
        )

        rescale = 0.1 * (0.5 ** layer_id) if cfg.rescale_nffn else None
        self.nffn = NormalizedFeedForwardNetwork(
            model_dim=cfg.model_dim,
            ffn_hidden_dim=cfg.ffn_hidden_dim,
            swiglu=cfg.swiglu,
            dropout=cfg.dropout,
            hidden_dropout=cfg.hidden_dropout,
            norm_num_groups=cfg.layernorm_num_groups,
            norm_affine=cfg.norm_affine,
            norm_eps=cfg.norm_eps,
            memory_efficient_norm=cfg.memory_efficient_norm,
            rescale=rescale,
            init_mode=cfg.init_mode
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Optional[Tensor],
        bos_mask: Optional[Tensor] = None,
        segment_idx: Optional[Tensor] = None,
        prev_segment_idx: Optional[Tensor] = None,
        prev_segment_count: Optional[Tensor] = None,
        cache: Optional[Tuple[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, Tensor], Tensor]] = None,
    ):
        y, cache = self.mega(x, freqs_cis, bos_mask, segment_idx, prev_segment_idx, prev_segment_count, cache)
        out = self.nffn(y, x if self.two_hop_residual else y)
        return out, cache


class GeckoOutputLayer(nn.Module):
    def __init__(self, cfg: ModelConf, embed_weight: nn.Parameter):
        super().__init__()

        self.model_dim = cfg.model_dim
        self.output_size = cfg.vocab_size if cfg.output_size == -1 else cfg.output_size
        self.memory_efficient_norm = cfg.memory_efficient_norm

        self.final_norm = TimestepDecayNorm(self.model_dim, cfg.timenorm_num_groups, cfg.timenorm_beta1, cfg.timenorm_beta2,
                                            eps=cfg.norm_eps, memory_efficient=cfg.memory_efficient_norm)

        init_fn = get_init_fn('gaussian', dim=self.model_dim)
        self.output = Linear(self.model_dim, self.output_size, bias=False, init_method=init_fn)
        self.share_emb = cfg.share_emb
        if self.share_emb:
            self.output.weight = embed_weight

    def forward(
        self,
        x: Tensor,
        bos_mask: Optional[Tensor] = None,
        cache: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ):
        if cache is not None:
            prev_count, prev_mean, prev_var = cache
        else:
            prev_count, prev_mean, prev_var = None, None, None

        x, prev_count, prev_mean, prev_var = self.final_norm(x, bos_mask, prev_count, prev_mean, prev_var)

        if cache is not None:
            cache = (prev_count.detach(), prev_mean.detach(), prev_var.detach())

        out = self.output(x)

        return out, cache


class Gecko(nn.Module):
    def __init__(self, cfg: ModelConf, tokenizer: Tokenizer):
        super().__init__()
        assert cfg.vocab_size > 0
        self.vocab_size = cfg.vocab_size
        self.num_layers = cfg.num_layers
        self.chunk_size = cfg.chunk_size
        self.dropout = cfg.dropout
        self.emb_scale = math.sqrt(cfg.model_dim) if cfg.scale_emb else None
        self.share_emb = cfg.share_emb
        self.tokenizer = tokenizer

        model_dim = cfg.model_dim
        init_fn = get_init_fn('gaussian', dim=model_dim)
        self.embed = nn.Embedding(self.vocab_size, model_dim)
        init_fn(self.embed.weight)

        self.z_head_dim = cfg.z_dim // cfg.num_heads
        num_rope_heads = cfg.num_heads if cfg.num_rope_heads is None else cfg.num_rope_heads
        if num_rope_heads > 0:
            self.rope = RotaryEmbedding(self.z_head_dim, cfg.chunk_size * 16, base=cfg.rope_base)
        else:
            self.rope = None

        self.layers = nn.ModuleList()
        for layer_id in range(self.num_layers):
            layer = GeckoBlock(cfg, layer_id)
            self.layers.append(layer)

        self.output = GeckoOutputLayer(cfg, self.embed.weight)

    @property
    def bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_id

    def forward(
        self,
        tokens: Tensor,
        multi_segments: bool,
        cache: Optional[Tuple[List[Tuple[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, Tensor],
                                         Tuple[Tensor, Tensor, Tensor], torch.Tensor]],
                              Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
    ):

        bsz, seq_len = tokens.shape

        if multi_segments:
            bos_mask = torch.eq(tokens, self.bos_id)
            segment_idx = torch.cumsum(bos_mask, dim=-1)
            prev_segment_idx = None
            prev_segment_count = None
        else:
            bos_mask = None
            segment_idx = None
            prev_segment_idx = None
            prev_segment_count = None

        if self.training:
            assert cache is None, "training model does not support kv cache."
            assert seq_len % self.chunk_size == 0
            # split tokens into chunks
            cache_layers, cache_output, cache_segment = None, None, None
            start = 0
            end = seq_len
        else:
            cache_layers, cache_output, cache_segment = cache
            start = 0 if cache_layers[0][0] is None else cache_layers[0][0][-1]
            end = start + seq_len
            if seq_len >= self.chunk_size:
                assert start % self.chunk_size == 0 and end % self.chunk_size == 0
            elif seq_len > 1:
                assert start % self.chunk_size == 0

            if bos_mask is not None:
                prev_segment_idx, prev_segment_count = cache_segment
                if prev_segment_idx is not None:
                    segment_idx = segment_idx + prev_segment_idx[:, -1:]

        # embeddings
        emb = self.embed(tokens)
        if self.emb_scale is not None:
            emb = emb * self.emb_scale

        x = memory_efficient_dropout(emb, self.dropout, self.training)
        # rope frequencies
        freq_cis = None if self.rope is None else self.rope.get_freqs_cis(start, end)

        for i, layer in enumerate(self.layers):
            layer_cache = cache_layers[i] if cache is not None else None
            x, layer_cache = layer(x, freq_cis, bos_mask, segment_idx, prev_segment_idx, prev_segment_count, layer_cache)

            if cache is not None:
                cache_layers[i] = layer_cache

        out, cache_output = self.output(x, bos_mask, cache_output)

        if cache is not None:
            if bos_mask is not None:
                if prev_segment_idx is None:
                    prev_segment_idx = segment_idx
                    prev_segment_count = segment_idx[:, 0]
                else:
                    prev_segment_idx = torch.cat([prev_segment_idx, segment_idx], dim=1)

                if end % self.chunk_size == 0:
                    length = prev_segment_idx.shape[1]
                    prev_segment_count = prev_segment_idx[:, max(length - self.chunk_size - 1, 0)]
                    prev_segment_idx = prev_segment_idx[:, length - self.chunk_size:]

                cache_segment = (prev_segment_idx, prev_segment_count)
            cache = (cache_layers, cache_output, cache_segment)

        return out.float(), cache


# fmt: on
