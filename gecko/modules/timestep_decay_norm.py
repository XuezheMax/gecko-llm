from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch.autograd.function import FunctionCtx
from torch.nn.parameter import Parameter

from gecko_extension.ops import (
    group_timestep_decay_norm_fwd,
    group_timestep_decay_norm_bwd
)


class TimestepDecayNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        bos_mask: Optional[torch.Tensor],
        prev_count: torch.Tensor,
        prev_mean: torch.Tensor,
        prev_var: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        num_groups: int,
        padding_mask: Optional[torch.Tensor] = None,
        beta1: float = 0.999,
        beta2: float = 0.9999,
        eps: float = 1e-5,
        memory_efficient: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y, count, mean, var, cummean, cumrstd = group_timestep_decay_norm_fwd(
            x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, padding_mask, num_groups, beta1, beta2, eps,
        )
        ctx.save_for_backward(x if not memory_efficient else y, bos_mask, prev_count, cummean, cumrstd, gamma, beta, padding_mask)
        ctx.num_groups = num_groups  # num_groups is not a torch.Tensor
        ctx.beta1 = beta1  # beta1 is not a torch.Tensor
        ctx.beta2 = beta2  # beta2 is not a torch.Tensor
        ctx.eps = eps  # eps is not a torch.Tensor
        ctx.memory_efficient = memory_efficient
        return y, count, mean, var

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor,
        _,
        mean_grad: torch.Tensor,
        var_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, None, None, None, None, None, None]:
        x_or_y, bos_mask, prev_count, cummean, cumrstd, gamma, beta, padding_mask = ctx.saved_tensors
        num_groups = ctx.num_groups
        beta1 = ctx.beta1
        beta2 = ctx.beta2
        eps = ctx.eps
        memory_efficient = ctx.memory_efficient
        x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad = group_timestep_decay_norm_bwd(
            y_grad, mean_grad, var_grad, x_or_y, prev_count, bos_mask, cummean, cumrstd, gamma, beta, padding_mask, num_groups, beta1, beta2, eps, memory_efficient
        )
        return x_grad, None, None, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad, None, None, None, None, None, None


timestep_decay_norm = TimestepDecayNormFunc.apply


class TimestepDecayNorm(nn.Module):

    def __init__(
        self,
        num_features: int,
        num_groups: int,
        beta1: float = 0.999,
        beta2: float = 0.9999,
        eps: float = 1e-5,
        memory_efficient: bool = False,
    ) -> None:

        super().__init__()

        self.num_features = num_features
        self.num_groups = num_groups
        assert num_groups < num_features and num_features % num_groups == 0

        self.register_buffer("prior_count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("prior_mean", torch.zeros(self.num_groups))
        self.register_buffer("prior_var", torch.zeros(self.num_groups))

        self.register_parameter("weight", Parameter(torch.zeros(self.num_features)))
        self.register_parameter("bias", Parameter(torch.zeros(self.num_features)))

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.memory_efficient = memory_efficient

    def forward(
        self,
        x: torch.Tensor,
        bos_mask: Optional[torch.Tensor] = None,
        prev_count: Optional[torch.Tensor] = None,
        prev_mean: Optional[torch.Tensor] = None,
        prev_var: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = x.size(0)
        if prev_count is None:
            prev_count = self.prior_count.expand(batch_size).contiguous()
        if prev_mean is None:
            prev_mean = self.prior_mean.type_as(x).expand(batch_size, -1).contiguous()
        if prev_var is None:
            prev_var = self.prior_var.type_as(x).expand(batch_size, -1).contiguous()

        output = timestep_decay_norm(x, bos_mask, prev_count, prev_mean, prev_var, self.weight + 1.0, self.bias,
                                     self.num_groups, padding_mask, self.beta1, self.beta2,
                                     self.eps, self.memory_efficient)
        return output

    def extra_repr(self) -> str:
        return 'num_features={num_features}, ' \
               'num_groups={num_groups}, ' \
               'betas=({beta1}, {beta2}), eps={eps}, ' \
               'memory_efficient={memory_efficient}'.format(**self.__dict__)
