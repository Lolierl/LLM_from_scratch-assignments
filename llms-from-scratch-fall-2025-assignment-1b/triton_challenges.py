import torch
import triton
import triton.language as tl

# WRITE YOUR CODE HERE!
@triton.autotune(
    configs=[
        triton.Config({'S0': s0, 'S1': s1}, num_warps = w)
        for s0 in [8, 16, 32, 64, 128]
        for s1 in [8, 16, 32, 64, 128]
        for w in [2, 4, 8, 16]
    ],
    key = ['m', 'n'],
    restore_value= ['output_ptr']
)

@triton.jit
def outer_product_with_relu_kernel(x_ptr, y_ptr, output_ptr, m, n, S0: tl.constexpr, S1: tl.constexpr):
    pidx = tl.program_id(axis = 0)
    pidy = tl.program_id(axis = 1)
    segment_start_x = pidx * S0
    segment_start_y = pidy * S1
    offsets_x = segment_start_x + tl.arange(0, S0)
    offsets_y = segment_start_y + tl.arange(0, S1)
    mask_x = offsets_x < m
    mask_y = offsets_y < n
    x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0)
    y = tl.load(y_ptr + offsets_y, mask=mask_y, other=0.0)
    output = x[:, None] * y[None, :]
    offsets = offsets_x[:, None] * n + offsets_y[None, :]
    mask = mask_x[:, None] & mask_y[None, :]
    output = tl.maximum(output, 0)
    tl.store(output_ptr + offsets, output, mask=mask)

def outer_product_with_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = x.shape[0]
    n = y.shape[0]
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(m, meta['S0']), triton.cdiv(n, meta['S1']))
    outer_product_with_relu_kernel[grid](x, y, output, m, n)
    return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}, num_warps=w)
        for bs in [256, 512, 1024, 2048]
        for w in [4, 8, 16]
    ],
    key=['n'],
)
@triton.jit
def l2_norm_kernel(x_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = x_ptr + pid * n
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for start in range(0, n, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    
    result = tl.sum(acc, axis=0)
    tl.store(output_ptr + pid, tl.sqrt(result))

def l2_norm(x: torch.Tensor) -> torch.Tensor:
    d, n = x.shape
    output = torch.empty((d,), device=x.device, dtype=x.dtype)
    grid = lambda meta: (d, )
    l2_norm_kernel[grid](x, output, n)
    return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=16,  num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=16,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128}, num_warps=4,  num_stages=2),
    ],
    key=['n'],
    restore_value=['o_ptr']
)
@triton.jit
def sigmoid_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    n, d,
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    scale, inv_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):

    pid_m = tl.program_id(0)
    m_id = pid_m * BLOCK_M

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(n, d),
        strides=(stride_qn, stride_qd),
        offsets=(m_id, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(0, 1),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr,
        shape=(n, d),
        strides=(stride_on, stride_od),
        offsets=(m_id, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(0, 1),
    )

    q_tile = tl.load(q_block_ptr, boundary_check=(0, ), padding_option="zero")
    q_tile = q_tile * scale.to(tl.float16)

    o_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    num_tiles = tl.cdiv(n, BLOCK_N)
    for n0 in range(0, num_tiles):
        offs_n = n0 * BLOCK_N

        k_block_ptr = tl.make_block_ptr(
            base=k_ptr, shape=(n, d), strides=(stride_kn, stride_kd),
            offsets=(offs_n, 0), block_shape=(BLOCK_N, BLOCK_D), order=(0, 1),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_ptr, shape=(n, d), strides=(stride_vn, stride_vd),
            offsets=(offs_n, 0), block_shape=(BLOCK_N, BLOCK_D), order=(0, 1),
        )

        k_tile = tl.load(k_block_ptr, boundary_check=(0, ), padding_option="zero")
        v_tile = tl.load(v_block_ptr, boundary_check=(0, ), padding_option="zero")

        scores = tl.dot(q_tile, tl.trans(k_tile))      
        att = (1.0 / (1.0 + tl.exp(-scores))).to(tl.float16)          
        o_acc = tl.dot(att, v_tile, o_acc)

    o_acc = (o_acc * inv_n).to(tl.float16)
    tl.store(o_block_ptr, o_acc, boundary_check=(0, ))

def sigmoid_attention(q: torch.Tensor,
                      k: torch.Tensor, 
                      v: torch.Tensor) -> torch.Tensor:
    n, d = q.shape
    o = torch.empty_like(q, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_M']),)
    sigmoid_attention_kernel[grid](
        q, k, v, o,
        n, d,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        (d ** -0.5), float(1.0 / n),
        BLOCK_D=d,
    )

    return o

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}, num_warps=w)
        for bs in [512, 1024, 2048, 4096, 8192]
        for w in [8, 16]
    ],
    key=['m'],
    restore_value=['output_ptr']
)
@triton.jit
def log_softmax_kernel(x_ptr, output_ptr, m, inv_T, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = x_ptr + pid * m
    output_row_ptr = output_ptr + pid * m
    x_max = tl.full((1,), -float('inf'), dtype=tl.float32)
    sum = tl.zeros((1,), dtype=tl.float32)
    for start in range(0, m, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < m
        x = tl.load(row_ptr + offs, mask=mask, other=-float('inf')).to(tl.float32)
        x = x * inv_T
        x_max_new = tl.maximum(x_max, tl.max(x, axis=0))
        sum = sum * tl.exp(x_max - x_max_new) + tl.sum(tl.exp(x - x_max_new), axis=0)
        x_max = x_max_new
    logfac = tl.log(sum) + x_max
    for start in range(0, m, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < m
        x = tl.load(row_ptr + offs, mask=mask, other=-float('inf')).to(tl.float32)
        x = x * inv_T
        res = x - logfac
        tl.store(output_row_ptr + offs, res, mask=mask)


def log_softmax(x: torch.Tensor, T: float) -> torch.Tensor:
    d, m = x.shape
    output = torch.empty((d, m), device=x.device, dtype=x.dtype)
    grid = lambda meta: (d, )
    log_softmax_kernel[grid](x, output, m, 1.0 / T)
    return output


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': m, 'BLOCK_N': n},  num_warps=w,  num_stages=s)
        for m in [64, 128]
        for n in [64]
        for w in [4, 8, 16]
        for s in [2, 3, 4]
    ],
    key=['n'],
    restore_value=['o_ptr'],
)
@triton.jit
def flash_attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, n,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)  
    pid_m  = tl.program_id(1)  
    m0 = pid_m * BLOCK_M

    q_bh = q_ptr + pid_bh * stride_qbh
    k_bh = k_ptr + pid_bh * stride_kbh
    v_bh = v_ptr + pid_bh * stride_vbh
    o_bh = o_ptr + pid_bh * stride_obh

    q_block_ptr = tl.make_block_ptr(
        base=q_bh, shape=(n, BLOCK_D), strides=(stride_qn, stride_qd),
        offsets=(m0, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0),
    )
    
    o_block_ptr = tl.make_block_ptr(
        base=o_bh, shape=(n, BLOCK_D), strides=(stride_on, stride_od),
        offsets=(m0, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0),
    )
    
    q = tl.load(q_block_ptr, boundary_check=(0,), padding_option="zero")

    x_max = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)  
    sum = tl.zeros((BLOCK_M,), dtype=tl.float32)                
    res = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)        
    
    LOG2_E = 1.442695 

    for n0 in range(0, tl.cdiv(n, BLOCK_N)):
        offs_n = n0 * BLOCK_N

        k_block_ptr = tl.make_block_ptr(
            base = k_bh, shape = (n, BLOCK_D), strides = (stride_kn, stride_kd),
            offsets = (offs_n, 0), block_shape = (BLOCK_N, BLOCK_D), order = (1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base = v_bh, shape = (n, BLOCK_D), strides = (stride_vn, stride_vd),
            offsets = (offs_n, 0), block_shape = (BLOCK_N, BLOCK_D), order = (1, 0),
        )
        k = tl.load(k_block_ptr, boundary_check=(0,), padding_option="zero")
        v = tl.load(v_block_ptr, boundary_check=(0,), padding_option="zero")

        scores = tl.dot(q, k.T) * scale * LOG2_E 
        mask_n = (offs_n + tl.arange(0, BLOCK_N)) < n
        scores = tl.where(mask_n[None, :], scores, -float('inf'))

        current_max = tl.max(scores, axis=1)                      
        p = tl.exp2(scores - current_max[:, None]).to(tl.float16)

        x_max_new = tl.maximum(x_max, current_max)
        alpha = tl.exp2(x_max - x_max_new)
        beta = tl.exp2(current_max - x_max_new)
        res = res * alpha[:, None] + tl.dot(p, v) * beta[:, None]
        sum = sum * alpha + tl.sum(p, axis=1) * beta
        x_max = x_max_new

    o = res / sum[:, None]
    tl.store(o_block_ptr, o.to(tl.float16), boundary_check=(0,))

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, H, N, D = q.shape
    q = q.view(-1, N, D)
    k = k.view(-1, N, D)
    v = v.view(-1, N, D)
    o = torch.empty_like(q)
    grid = lambda meta: (B * H, triton.cdiv(N, meta['BLOCK_M']))
    flash_attention_fwd_kernel[grid](
        q, k, v, o, N,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        D ** -0.5,
        BLOCK_D=D,
    )
    o = o.view(B, H, N, D)
    return o
