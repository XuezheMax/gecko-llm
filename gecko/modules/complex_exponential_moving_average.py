# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
import math
import torch
from torch import nn

from .fused_ops import cema_blelloch_scan

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


def _reset_parameters(alpha, delta, theta, gamma, embed_dim, ndim):
    # delta & alpha
    nn.init.normal_(alpha, mean=0.0, std=0.2)
    nn.init.normal_(delta, mean=0.0, std=0.2)
    # sync global permuted index
    idx = torch.randperm(embed_dim)
    # theta
    freqs = math.log(embed_dim) / embed_dim
    freqs = torch.exp(torch.arange(1, embed_dim + 1, requires_grad=False) * -freqs)
    freqs = freqs[idx]
    freqs = freqs.to(theta)
    freqs = torch.log(freqs / (1.0 - freqs))
    with torch.no_grad():
        theta.copy_(freqs)
    # gamma
    nn.init.normal_(gamma, mean=0.0, std=math.sqrt(1.0 / ndim))
    with torch.no_grad():
        gamma[:, :, 1] = 0.0


class MultiHeadComplexEMA(nn.Module):
    """Complex Exponential Moving Average Layer.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        ndim=16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim

        self.alpha = nn.Parameter(torch.Tensor(embed_dim, ndim))
        self.delta = nn.Parameter(torch.Tensor(embed_dim, ndim))
        self.theta = nn.Parameter(torch.Tensor(embed_dim))
        self.gamma = nn.Parameter(torch.Tensor(embed_dim, ndim, 2))
        self._coeffs = None
        # init parameters
        self._init_parameters()

    def _init_parameters(self):
        _reset_parameters(self.alpha, self.delta, self.theta, self.gamma, self.embed_dim, self.ndim)

    def _calc_coeffs(self):
        self._coeffs = None
        # D
        theta = torch.sigmoid(self.theta.float()) * (2 * math.pi / self.ndim)
        # N
        wavelets = torch.arange(1, self.ndim + 1, dtype=theta.dtype, device=theta.device)
        # D x N
        theta = wavelets * theta.unsqueeze(1)

        # D x N
        alpha = torch.sigmoid(self.alpha.float())
        delta = torch.sigmoid(self.delta.float())
        # coeffs
        p = alpha
        q = torch.polar(1.0 - alpha * delta, theta)
        # D x N
        gamma = _r2c(self.gamma.float())
        return p, q, gamma

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor],
        bos_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # calc coeffs
        p, q, gamma = self.coeffs()

        # B x D x L
        residual = x
        output, h = cema_blelloch_scan(x, hx, p, q, gamma, bos_mask)
        # residual
        output = output + residual
        return output, h

    def extra_repr(self) -> str:
        return 'edim={}, ndim={}'.format(self.embed_dim, self.ndim)
