from typing import Tuple

import torch
from torch.autograd.function import FunctionCtx

from gecko_extension.ops import attention_softmax_fwd, attention_softmax_bwd


class AttentionSoftmaxFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        training: bool = True
    ) -> torch.Tensor:
        y = attention_softmax_fwd(x, dropout if training else 0.0, use_causal_mask)
        ctx.save_for_backward(y)
        # use_causal_mask is not a torch.Tensor.
        ctx.use_causal_mask = use_causal_mask
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        y, = ctx.saved_tensors
        use_causal_mask = ctx.use_causal_mask
        x_grad = attention_softmax_bwd(y_grad, y, use_causal_mask)
        return x_grad, None, None, None


attention_softmax = AttentionSoftmaxFunc.apply
