import numpy as np
import torch
from typing import Tuple, BinaryIO, IO
import os
from torch import nn, optim



def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    pass


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:
    pass
    