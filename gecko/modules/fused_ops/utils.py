import torch
from torch import Tensor


def logsumexp_backward(
    grad: Tensor,
    inp: Tensor,
    out: Tensor,
    dim: int
) -> Tensor:
    if dim < 0:
        dim = inp.dim() + dim
    grad = grad.unsqueeze(dim)
    out = out.unsqueeze(dim)
    return grad * torch.exp(inp - out)
