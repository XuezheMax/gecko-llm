from typing import Optional, Tuple
import torch
from torch.autograd.function import FunctionCtx

from gecko_extension.ops import (
    cema_blelloch_scan_fwd,
    cema_blelloch_scan_bwd,
    cema_cub_scan_fwd,
    cema_cub_scan_bwd,
)

cema_scan_fwd = cema_cub_scan_fwd
cema_scan_bwd = cema_cub_scan_bwd


class CEMABlellochScanFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x:torch.Tensor,
        hx: Optional[torch.Tensor],
        p: torch.Tensor,
        q: torch.Tensor,
        gamma: torch.Tensor,
        bos_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y, h, chunk_decay, chunk_gain = cema_scan_fwd(x, p, q, gamma, bos_mask, hx)
        ctx.save_for_backward(x, hx, p, q, gamma, chunk_decay, chunk_gain, bos_mask)
        return y, h

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor,
        h_grad: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               torch.Tensor, torch.Tensor, torch.Tensor, None]:
        x, hx, p, q, gamma, chunk_decay, chunk_gain, bos_mask = ctx.saved_tensors
        x_grad, p_grad, q_grad, gamma_grad, hx_grad = cema_scan_bwd(
            y_grad, h_grad, chunk_decay, chunk_gain, x, p, q, gamma, bos_mask
        )
        hx_grad = None if hx is None else hx_grad

        return x_grad, hx_grad, p_grad, q_grad, gamma_grad, None


cema_blelloch_scan = CEMABlellochScanFunc.apply
