from typing import Union, Tuple, List, Callable, Optional, Dict, Any
from functools import partial
import math

import torch
import torch.nn as nn


def get_init_fn(mode, dim=None) -> Callable[[torch.Tensor], torch.Tensor]:
    if mode == 'none':
        return lambda x: x
    elif mode == 'bert':
        std = 0.02
        init_fn = partial(nn.init.normal_, mean=0.0, std=std)
    elif mode == 'he':
        a = math.sqrt(5.0)
        init_fn = partial(nn.init.kaiming_normal_, a=a)
    elif mode == 'gaussian':
        std = 1.0 if dim is None else 1.0 / math.sqrt(dim)
        a = -3 * std
        b = 3 * std
        init_fn = partial(nn.init.trunc_normal_, mean=0.0, std=std, a=a, b=b)
    elif mode == 'xavier':
        init_fn = partial(nn.init.xavier_uniform_)
    else:
        raise ValueError('Unknown init mode: {}'.format(mode))

    return init_fn
