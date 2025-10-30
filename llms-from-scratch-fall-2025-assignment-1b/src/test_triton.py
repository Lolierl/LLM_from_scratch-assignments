import torch
from .triton_challenges import *
from .torch_sba import *
from .triton_sba import *
import pytest
import signal
import numpy as np
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel


def check_same_tensor(a, b, eps=1e-5):
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert a.device == b.device, f"Device mismatch: {a.device} vs {b.device}"
    assert a.dtype == b.dtype, f"Dtype mismatch: {a.dtype} vs {b.dtype}"
    
    with torch.no_grad():
        diff = (a - b).abs().max().item()
    assert diff < eps, f"Wrong Answer! Max difference: {diff:.2e} > {eps:.2e}\nA: {a}\nB: {b}"
    print(f"✓ Correctness test passed! Max difference: {diff:.2e}")

def run_first_call_very_carefully(func, *args, time_limit=30, **kwargs):
    tensor_args = [x for x in args if isinstance(x, torch.Tensor)]
    tensor_args_copy = [x.clone() for x in tensor_args]
    
    def handler(signum, frame):
        raise TimeoutError("Time limit exceeded: Your first call is too slow.")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_limit)

    try:
        ret = func(*args, **kwargs)
        for x, x_copy in zip(tensor_args, tensor_args_copy):
            assert x.dtype == x_copy.dtype, f"Input tensor's dtype changed!"
            assert x.device == x_copy.device, f"Input tensor's device changed!"
            assert x.shape == x_copy.shape, f"Input tensor's shape changed!"
            assert torch.all(x == x_copy), f"Input tensor changed!"
        return ret
    finally:
        signal.alarm(0)

@torch.no_grad()
def do_bench(func):
    return triton.testing.do_bench(func, rep=500)

def check_time_limit(ms, torch_ms, ratio=1.05):
    assert ms < torch_ms * ratio, f"Time limit exceeded: Your Triton implementation is too slow. Yours: {ms:.6f} ms, PyTorch: {torch_ms:.6f} ms. Ratio: {ms / torch_ms:.2f} > {ratio}"
    print(f"✓ Speed test passed! Yours: {ms:.6f} ms, PyTorch: {torch_ms:.6f} ms. Ratio: {ms / torch_ms:.2f} < {ratio}")
    

@torch.compile
def outer_product_with_relu_torch(x: torch.Tensor, y: torch.Tensor):
    return torch.relu(torch.outer(x, y))

@pytest.mark.parametrize("m, n, test_speed", [
    (1, 1, False),
    (1, 100, False),
    (100, 1, False),
    (30, 1000, True),
    (1000, 30, True),
    (900, 1100, True),
    (1100, 900, True),
    (10000, 45, True),
    (45, 10000, True),
])
@torch.no_grad()
def test_outer_product_with_relu(m, n, test_speed, capsys):
    """
    Test the outer_product_with_relu function.
    Your implementation must be at most 5% slower than the PyTorch version.
    """

    with capsys.disabled():
        x = torch.randn(m, device='cuda', dtype=torch.float32)
        y = torch.randn(n, device='cuda', dtype=torch.float32)

        print(f"\n>>> Testing outer_product_with_relu with m={m}, n={n}")

        out = run_first_call_very_carefully(outer_product_with_relu, x, y)
        ans = outer_product_with_relu_torch(x, y)
        
        check_same_tensor(ans, out)
        
        if test_speed:
            ms = do_bench(lambda: outer_product_with_relu(x, y))
            torch_ms = do_bench(lambda: outer_product_with_relu_torch(x, y))
            check_time_limit(ms, torch_ms)


@torch.compile
def l2_norm_torch(x: torch.Tensor):
    return torch.sqrt((x * x).sum(dim=1))

@pytest.mark.parametrize("d, n, test_speed", [
    (1, 10000, True),
    (2, 10000, True),
    (32, 10000, True),
    (128, 10000, True),
    (32, 10240, True),
    (2, 20000, True),
    (1, 30000, True),
    (1, 50000, True),
    (7, 50000, True),
    (1, 70000, True),
    (16, 70000, True),
    (16, 100000, True),
    (7, 100000, True),
    (1, 100000, True),
    (128, 100000, True),
])
@torch.no_grad()
def test_l2_norm(d, n, test_speed, capsys):
    """
    Test the l2_norm function.
    Your implementation must be at most 5% slower than the PyTorch version.
    """

    with capsys.disabled():
        x = torch.randn(d, n, device='cuda', dtype=torch.float32)
        print(f"\n>>> Testing l2_norm with d={d}, n={n}")
        out = run_first_call_very_carefully(l2_norm, x)
        ans = l2_norm_torch(x)
        check_same_tensor(ans, out, eps=1e-8 * n)
        
        if test_speed:
            ms = do_bench(lambda: l2_norm(x))
            torch_ms = do_bench(lambda: l2_norm_torch(x))
            check_time_limit(ms, torch_ms)

@torch.compile
def naive_sigmoid_attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    factor = q.shape[1] ** -0.5
    return (torch.sigmoid((q @ k.T) * factor).to(torch.float16) @ v) / q.shape[0]

@pytest.mark.parametrize("n, time_limit_ratio", [
    (1, 0),
    (2, 0),
    (3, 0),
    (50, 0),
    (100, 0),
    (128, 0),
    (500, 0.5),
    (1000, 0.3),
    (2000, 0.2),
    (4000, 0.1),
    (7000, 0.1),
    (8000, 0.1),
    (10000, 0.07),
])
@torch.no_grad()
def test_sigmoid_attention(n, time_limit_ratio, capsys):
    """
    Test the sigmoid_attention function.
    Your implementation should be "flash"!
    """

    with capsys.disabled():
        d = 128
        q = torch.randn((n, d), device='cuda', dtype=torch.float16)
        k = torch.randn((n, d), device='cuda', dtype=torch.float16)
        v = torch.randn((n, d), device='cuda', dtype=torch.float16)
        print(f"\n>>> Testing sigmoid_attention with n={n}")
        
        out = run_first_call_very_carefully(sigmoid_attention, q, k, v)
        ans = naive_sigmoid_attention_torch(q, k, v)
        check_same_tensor(ans, out, eps=1e-3)
        
        if time_limit_ratio > 0:
            ms = do_bench(lambda: sigmoid_attention(q, k, v))
            torch_ms = do_bench(lambda: naive_sigmoid_attention_torch(q, k, v))
            check_time_limit(ms, torch_ms, ratio=time_limit_ratio)

def naive_sigmoid_attention_torch_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    factor = q.shape[1] ** -0.5
    return (torch.sigmoid((q @ k.T) * factor).to(torch.float16) @ v) / q.shape[0]

@pytest.mark.parametrize("n, time_limit_ratio", [
    (1, 0),
    (2, 0),
    (3, 0),
    (50, 0),
    (100, 0),
    (128, 0),
    (8000, 0.9),
    (10000, 0.8),
    (20000, 0.7),
    (40000, 0.6),
])
@torch.no_grad()
def test_sigmoid_attention_v2(n, time_limit_ratio, capsys):
    """
    Test the sigmoid_attention function.
    Your implementation should be "flash"!
    """

    with capsys.disabled():
        d = 128
        q = torch.randn((n, d), device='cuda', dtype=torch.float16)
        k = torch.randn((n, d), device='cuda', dtype=torch.float16)
        v = torch.randn((n, d), device='cuda', dtype=torch.float16)
        print(f"\n>>> Testing sigmoid_attention with n={n}")
        
        out = run_first_call_very_carefully(sigmoid_attention, q, k, v)
        ans = naive_sigmoid_attention_torch(q, k, v)
        check_same_tensor(ans, out, eps=1e-3)
        
        if time_limit_ratio > 0:
            ms = do_bench(lambda: sigmoid_attention(q, k, v))
            torch_ms = do_bench(lambda: naive_sigmoid_attention_torch_v2(q, k, v))
            check_time_limit(ms, torch_ms, ratio=time_limit_ratio)

@torch.compile
def log_softmax_torch(x: torch.Tensor, T: float):
    return torch.log_softmax(x / T, dim=1)

@pytest.mark.parametrize("B, m, test_speed", [
    (32, 128, False),
    (64, 512, False),
    (128, 1024, False),
    (60, 3000, True),
    (128, 5000, True),
    (50, 7000, True),
    (88, 10000, True),
    (128, 10000, True),
])
@torch.no_grad()
def test_log_softmax(B, m, test_speed, capsys):
    """
    Test the log_softmax function.
    Your implementation must be at least 10% faster than the PyTorch version.
    """

    with capsys.disabled():
        print(f"\n>>> Testing log_softmax with B={B}, m={m}")
        for t in range(5):
            T = (0.1 + torch.rand((1,), dtype=torch.float32) * 10).item()
            x = torch.randn((B, m), device='cuda', dtype=torch.float32) + torch.randn((B, 1), device='cuda', dtype=torch.float32) * 100

            if t == 0:
                out = run_first_call_very_carefully(log_softmax, x, T)
            else:
                out = log_softmax(x, T)
            ans = log_softmax_torch(x, T)
            check_same_tensor(ans, out, eps=5e-7 * m)

        if test_speed:
            ms = do_bench(lambda: log_softmax(x, T))
            torch_ms = do_bench(lambda: log_softmax_torch(x, T))
            check_time_limit(ms, torch_ms, ratio=0.9)


def rand_attn_qkv(bs, h, l, d):
    k = torch.randn((bs, h, l, d), device='cuda', dtype=torch.float16)
    
    q1 = torch.gather(k, dim=2, index=torch.randint(l, size=(bs, h, l, 1), device='cuda', dtype=torch.int64))
    q2 = torch.gather(k, dim=2, index=torch.randint(l, size=(bs, h, l, 1), device='cuda', dtype=torch.int64))
    p1 = torch.rand((bs, h, l, 1), device='cuda', dtype=torch.float16) * 10
    p2 = torch.rand((bs, h, l, 1), device='cuda', dtype=torch.float16) * 10
    q = p1 * q1 + p2 * q2 + torch.randn((bs, h, l, d), device='cuda', dtype=torch.float16)

    v = torch.randn((bs, h, l, d), device='cuda', dtype=torch.float16)

    return q, k, v
    
    
@torch.compile
def flash_attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # # Only enable flash attention backend
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return scaled_dot_product_attention(query=q, key=k, value=v)

@pytest.mark.parametrize("bs, h, l, time_limit_ratio", [
    (1, 1, 1, 0),
    (1, 1, 10, 0),
    (1, 1, 128, 0),
    (1, 4, 128, 0),
    (4, 8, 128, 0),
    (4, 8, 129, 0),
    (4, 8, 174, 0),
    (4, 8, 1024, 1.7),
    (6, 7, 2000, 1.7),
    (3, 8, 2048, 1.7),
    (2, 3, 4096, 1.7),
    (8, 5, 8000, 1.7),
    (8, 8, 8192, 1.7),
    (4, 8, 12800, 1.7),
    (4, 4, 16384, 1.7),
])
@torch.no_grad()
def test_flash_attention(bs, h, l, time_limit_ratio, capsys):
    """
    Test the flash attention function.
    Your implementation must be at most 70% slower than PyTorch's flash attention.
    """

    with capsys.disabled():
        print(f"\n>>> Testing flash_attention with bs={bs} h={h} l={l}")
        d = 128
        q, k, v = rand_attn_qkv(bs, h, l, d)
        
        out = run_first_call_very_carefully(flash_attention, q, k, v)
        ans = flash_attention_torch(q, k, v)
        check_same_tensor(ans, out, eps=1e-2)
        
        if time_limit_ratio > 0:
            ms = do_bench(lambda: flash_attention(q, k, v))
            torch_ms = do_bench(lambda: flash_attention_torch(q, k, v))
            check_time_limit(ms, torch_ms, ratio=time_limit_ratio)


def do_bench_with_grad(func, dO, warmup=50, rep=200, grad_to_none=None):
    """
    Performs a benchmark on a given function, timing both forward and backward passes.
    """
    di = torch._dynamo.device_interface.get_interface_for_device("cuda")
    func().backward(dO, retain_graph=False)
    di.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        func().backward(dO, retain_graph=False)
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    fwd_start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    fwd_end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    bwd_start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    bwd_end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        func().backward(dO, retain_graph=False)
    # Benchmark
    for i in range(n_repeat):
        # clear the gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None

        # we clear the L2 cache before each run
        cache.zero_()

        # record forward time
        fwd_start_event[i].record()
        output = func() 
        fwd_end_event[i].record()

        # we clear the L2 cache between forward and backward
        cache.zero_()

        # record backward time
        bwd_start_event[i].record()
        output.backward(dO, retain_graph=False)
        bwd_end_event[i].record()

    # Record clocks
    di.synchronize()
    fwd_times = [s.elapsed_time(e) for s, e in zip(fwd_start_event, fwd_end_event)]
    bwd_times = [s.elapsed_time(e) for s, e in zip(bwd_start_event, bwd_end_event)]
    return np.mean(fwd_times), np.mean(bwd_times)

@torch.compile
def torch_stick_breaking_attention(q, k, v):
    return StickBreakingAttention.apply(q, k, v)

@pytest.mark.parametrize("bs, h, l", [
    (1, 1, 1),
    (1, 1, 10),
    (1, 1, 128),
    (1, 1, 256),
    (2, 2, 257),
    (2, 4, 512),
    (2, 4, 1024),
    (2, 4, 2048),
    (2, 4, 4097),
    (2, 4, 4096),
    (2, 4, 8192),
    (2, 4, 12800),
    (2, 4, 16384),
])
def test_stick_breaking_attention_correctness(bs, h, l, capsys):
    """
    Correctness test for the Triton Stick-Breaking Attention implementation
    """

    with capsys.disabled():
        print(f"\n>>> Testing Stick-Breaking Attention with B={bs}, H={h}, L={l}")
        d = 128
        q, k, v = rand_attn_qkv(bs, h, l, d)
        dO = torch.randn_like(v)

        baseline_fn = torch_stick_breaking_attention
        custom_fn = TritonStickBreakingAttention.apply

        q1, k1, v1 = q.clone(), k.clone(), v.clone()
        q1.requires_grad_()
        k1.requires_grad_()
        v1.requires_grad_()
        out_baseline = baseline_fn(q1, k1, v1)
        out_baseline.backward(gradient=dO)
        
        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        q2.requires_grad_()
        k2.requires_grad_()
        v2.requires_grad_()
        out_triton = run_first_call_very_carefully(custom_fn, q2, k2, v2)
        run_first_call_very_carefully(out_triton.backward, gradient=dO)

        check_same_tensor(out_triton, out_baseline, eps=5e-2)
        check_same_tensor(q2.grad, q1.grad, eps=5e-2)
        check_same_tensor(k2.grad, k1.grad, eps=5e-2)
        check_same_tensor(v2.grad, v1.grad, eps=5e-2)

@pytest.mark.parametrize("bs, h, l", [
    (2, 4, 4096),
    (2, 4, 16384),
])
def test_stick_breaking_attention_speedup(bs, h, l, capsys):
    """
    Speedup test for the Triton Stick-Breaking Attention implementation
    """

    with capsys.disabled():
        print(f"\n>>> Testing Stick-Breaking Attention with B={bs}, H={h}, L={l}")
        d = 128
        q, k, v = rand_attn_qkv(bs, h, l, d)
        dO = torch.randn_like(v)

        baseline_fn = torch_stick_breaking_attention
        custom_fn = TritonStickBreakingAttention.apply

        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        baseline_bench_fn = lambda: baseline_fn(q, k, v)
        triton_bench_fn = lambda: custom_fn(q, k, v)

        fwd_torch_ms, bwd_torch_ms = do_bench_with_grad(baseline_bench_fn, dO, warmup=1000, rep=10000, grad_to_none=[q, k, v])
        fwd_ms, bwd_ms = do_bench_with_grad(triton_bench_fn, dO, warmup=1000, rep=10000, grad_to_none=[q, k, v])

        print(f"Torch fwd pass: {fwd_torch_ms:.3f}; bwd pass: {bwd_torch_ms:.3f}")
        print(f"Triton fwd pass: {fwd_ms:.3f}; bwd pass: {bwd_ms:.3f}")
        torch_ms = fwd_torch_ms + bwd_torch_ms
        ms = fwd_ms + bwd_ms

        speed_up = torch_ms / ms
        print(f"Your triton implementation's speedup is {speed_up:.3f}")
