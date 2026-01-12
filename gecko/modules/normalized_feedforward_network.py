import torch
from torch import nn
import torch.nn.functional as F

from gecko.modules.linear import Linear
from gecko.modules.fused_ops import memory_efficient_dropout
from gecko.modules.layer_norm import GroupLayerNorm
from gecko.utils import get_init_fn


class NormalizedFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        model_dim,
        ffn_hidden_dim,
        dropout=0.0,
        hidden_dropout=0.0,
        swiglu=False,
        norm_num_groups=1,
        norm_affine=True,
        norm_eps=1e-5,
        memory_efficient_norm=False,
        rescale=None,
        init_mode='bert',
    ):
        super().__init__()

        self.model_dim = model_dim
        self.hidden_dim = ffn_hidden_dim
        self.dropout = dropout
        self.hidden_dropout = hidden_dropout
        self.swiglu = swiglu
        self.rescale_init = rescale
        self.init_mode = init_mode

        self.norm = GroupLayerNorm(
            model_dim,
            num_groups=norm_num_groups,
            elementwise_affine=norm_affine,
            eps=norm_eps,
            memory_efficient=memory_efficient_norm
        )

        # layers
        self.fc1 = Linear(
            model_dim,
            ffn_hidden_dim,
            bias=False,
            init_method=get_init_fn(init_mode),
        )
        self.fc2 = Linear(
            ffn_hidden_dim,
            model_dim,
            bias=False,
            init_method=get_init_fn(init_mode),
        )
        self.fc3 = Linear(
            model_dim,
            ffn_hidden_dim,
            bias=False,
            init_method=get_init_fn(init_mode),
        ) if self.swiglu else None

        if rescale is None:
            self.register_parameter('alpha', None)
        else:
            assert rescale > 0., 'Layer scale init value should be positive.'
            self.alpha = nn.Parameter(torch.full((model_dim,), rescale))

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.alpha is None else (self.alpha * x)

    def forward(self, x, residual):
        # B x L x D
        x = self.norm(x)
        # fc1 & fc3
        if self.swiglu:
            hidden = F.silu(self.fc1(x)) * self.fc3(x)
            hidden = memory_efficient_dropout(hidden, self.hidden_dropout, self.training)
        else:
            hidden = F.silu(self.fc1(x))
            hidden = memory_efficient_dropout(hidden, self.hidden_dropout, self.training)

        # fc2
        y = self.fc2(hidden)
        y = memory_efficient_dropout(y, self.dropout, self.training)
        # residual
        out = self.rescale(y) + residual

        return out

    def extra_repr(self) -> str:
        return 'dim={}, hdim={}, swiglu={}, init={}, rescale={}'.format(
            self.model_dim, self.hidden_dim, self.swiglu, self.init_mode, self.rescale_init
        )
