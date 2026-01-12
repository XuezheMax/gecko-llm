from typing import Callable

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class Linear(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
    ) -> None:
        super(Linear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        init_method(self.weight)
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = F.linear(input_, self.weight, self.bias)
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
