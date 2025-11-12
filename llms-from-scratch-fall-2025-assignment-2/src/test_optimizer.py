from __future__ import annotations
import numpy
import torch
from typing import Any
from collections.abc import Iterable
from .optim import *
from torch.nn.utils.clip_grad import clip_grad_norm_


def get_adamw_cls() -> Any:
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return get_lr_cosine_schedule(
        it,
        max_learning_rate,
        min_learning_rate,
        warmup_iters,
        cosine_cycle_iters,
    )


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    return gradient_clipping(parameters,max_l2_norm)


def _optimize(opt_class) -> torch.Tensor:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=10,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    for _ in range(1000):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()


def test_adamw(numpy_snapshot):
    pytorch_weights = _optimize(torch.optim.AdamW)
    actual_weights = _optimize(get_adamw_cls())

    matches_pytorch = torch.allclose(actual_weights, pytorch_weights, atol=3e-3)
    if matches_pytorch:
        return
    else:
        diff = (actual_weights - pytorch_weights).abs().max()
        assert matches_pytorch, (
            f"AdamW implementation does not match PyTorch reference within tolerance.\n"
            f"Max abs diff: {diff.item():.6f}\n"
            f"PyTorch weights:\n{pytorch_weights}\n"
            f"Actual weights:\n{actual_weights}"
        )


def test_get_lr_cosine_schedule():
    max_learning_rate = 1
    min_learning_rate = 1 * 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21

    expected_lrs = [
        0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857143,
        0.8571428571428571,
        1.0,
        0.9887175604818206,
        0.9554359905560885,
        0.9018241671106134,
        0.8305704108364301,
        0.7452476826029011,
        0.6501344202803414,
        0.55,
        0.44986557971965857,
        0.3547523173970989,
        0.26942958916356996,
        0.19817583288938662,
        0.14456400944391146,
        0.11128243951817937,
        0.1,
        0.1,
        0.1,
        0.1,
    ]
    actual_lrs = [
        run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for it in range(25)
    ]
    numpy.testing.assert_allclose(numpy.array(actual_lrs), numpy.array(expected_lrs))




def test_gradient_clipping():
    tensors = [torch.randn((5, 5)) for _ in range(6)]
    max_norm = 1e-2

    t1 = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    t1[-1].requires_grad_(False)

    loss = torch.cat(t1).sum()
    loss.backward()
    clip_grad_norm_(t1, max_norm)
    t1_grads = [torch.clone(t.grad) for t in t1 if t.grad is not None]

    t1_c = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    t1_c[-1].requires_grad_(False)
    loss_c = torch.cat(t1_c).sum()
    loss_c.backward()
    run_gradient_clipping(t1_c, max_norm)
    t1_c_grads = [torch.clone(t.grad) for t in t1_c if t.grad is not None]

    assert len(t1_grads) == len(t1_c_grads)

    for t1_grad, t1_c_grad in zip(t1_grads, t1_c_grads):
        numpy.testing.assert_allclose(
            t1_grad.detach().numpy(),
            t1_c_grad.detach().numpy(),
            atol=1e-6,
        )