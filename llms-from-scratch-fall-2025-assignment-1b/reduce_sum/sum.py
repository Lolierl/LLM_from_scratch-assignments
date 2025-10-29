import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'S0': s0, 'S1': s1, 'B1': b1}, num_warps=w)
        for s0 in [1, 2]
        for s1 in [256, 512, 1024]
        for b1 in [1, 4, 16, 64, 256]
        for w in [1, 2, 4, 8]
    ],
    key=['m', 'n'],
    restore_value=['output_ptr']
)
@triton.jit
def reduce_sum_kernel(
    x_ptr, x_stride0, x_stride1, output_ptr, m, n,
    S0: tl.constexpr, S1: tl.constexpr, B1):
    
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    
    i = pid0 * S0
    j = pid1 * S1

    offset_i = i + tl.arange(0, S0)
    mask_i = offset_i < m
    
    acc = tl.zeros((S0, S1), dtype=tl.float32)
    for k in range(0, tl.cdiv(n, S1 * B1)):
        offset_j = j + k * S1 * B1 + tl.arange(0, S1)
        mask_j = offset_j < n
        
        offset = offset_i[:, None] * x_stride0 + offset_j[None, :] * x_stride1
        mask = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + offset, mask=mask, other=0)

        acc += x
    
    sum_x = tl.sum(acc, axis=1)

    if B1 > 1:
        tl.atomic_add(output_ptr + offset_i, sum_x, mask=mask_i)
    else:
        tl.store(output_ptr + offset_i, sum_x, mask=mask_i)

def reduce_sum(x: torch.Tensor):
    m = x.shape[0]
    n = x.shape[1]
    output = torch.zeros(m, device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(m, meta['S0']), meta['B1'])
    reduce_sum_kernel[grid](x, x.stride(0), x.stride(1), output, m, n)
    return output


def check_same_tensor(a, b, eps=1e-5):
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert a.device == b.device, f"Device mismatch: {a.device} vs {b.device}"
    assert a.dtype == b.dtype, f"Dtype mismatch: {a.dtype} vs {b.dtype}"
    
    with torch.no_grad():
        diff = (a - b).abs().max().item()
    assert diff < eps, f"Wrong Answer! Max difference: {diff:.2e} > {eps:.2e}\nA: {a}\nB: {b}"
    print(f"✓ Correctness test passed! Max difference: {diff:.2e}")


def test_reduce_sum(m, n):
    print(f"Testing size: {m}x{n}")

    x = torch.rand(size=(m, n), device='cuda', dtype=torch.float32)
    output_torch = torch.sum(x, dim=1)
    output_triton = reduce_sum(x)
    check_same_tensor(output_torch, output_triton, eps=1e-6 * n)

    ms = triton.testing.do_bench(lambda: reduce_sum(x), rep=500)
    torch_ms = triton.testing.do_bench(lambda: torch.sum(x, dim=1), rep=500)
    print(f"Triton: {ms:.6f} ms, Torch: {torch_ms:.6f} ms. Ratio: {ms / torch_ms:.2f}")

    return ms, torch_ms


if __name__ == "__main__":
    test_reduce_sum(20, 100000)
