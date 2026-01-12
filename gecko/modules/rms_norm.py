from typing import Optional

import torch
import torch.nn as nn
from torch.autograd.function import FunctionCtx

from gecko_extension.ops import (
    group_rms_norm_fwd,
    group_rms_norm_bwd,
    group_rms_norm_fwd_affine,
    group_rms_norm_bwd_affine,
)


class GroupRMSNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        num_groups: int,
        eps: float = 1e-5,
        memory_efficient: bool = False,
    ):
        num_features = x.shape[-1]
        if weight is not None:
            y, rstd = group_rms_norm_fwd_affine(x, num_features, num_groups, weight, eps)
        else:
            y, rstd = group_rms_norm_fwd(x, num_features, num_groups, eps)

        ctx.save_for_backward(y if memory_efficient else x, weight, rstd)
        ctx.num_features = num_features
        ctx.num_groups = num_groups
        ctx.memory_efficient = memory_efficient

        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor,
    ):
        x_or_y, weight, rstd = ctx.saved_tensors
        num_features = ctx.num_features
        num_groups = ctx.num_groups
        memory_efficient = ctx.memory_efficient

        if weight is not None:
            x_grad, weight_grad = group_rms_norm_bwd_affine(
                y_grad, x_or_y, num_features, num_groups, rstd, weight, memory_efficient
            )
        else:
            x_grad = group_rms_norm_bwd(
                y_grad, x_or_y, num_features, num_groups, rstd, memory_efficient
            )
            weight_grad = None

        return x_grad, weight_grad, None, None, None


group_rms_norm = GroupRMSNormFunc.apply


class GroupRMSNorm(nn.Module):
    def __init__(self, num_features, num_groups=1, eps=1e-5, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        assert num_features % num_groups == 0
        self.num_features = num_features
        self.num_groups = num_groups
        self.features_per_group = num_features // num_groups

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.memory_efficient = memory_efficient
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        weight = None if self.weight is None else self.weight + 1.0
        return group_rms_norm(x, weight, self.num_groups, self.eps, self.memory_efficient)

    def extra_repr(self):
        return "num_features={num_features} ({features_per_group}), " \
               "num_groups={num_groups}, " \
               "eps={eps}, affine={elementwise_affine}, " \
               "memory_efficient={memory_efficient}".format(**self.__dict__)
