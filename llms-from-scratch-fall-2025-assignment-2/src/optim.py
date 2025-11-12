import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable, Optional
import math
from typing import Iterable, Optional, Callable
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    pass


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    pass


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    pass