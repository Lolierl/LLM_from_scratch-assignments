#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>

constexpr int WARP_SIZE = 32;
constexpr int MAX_NUM_THREADS = 512;
constexpr int ELEMENT_SIZE_BYTES = sizeof(float);

inline int64_t div_up(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// returns floor(log2(n))
inline int last_pow2(int n) {
    n |= (n >>  1);
    n |= (n >>  2);
    n |= (n >>  4);
    n |= (n >>  8);
    n |= (n >> 16);
    return std::max(1, n - (n >> 1));
}

void cpu_reduce_sum(float *x, int m, int n, float *output) {
    for (int k = 0; k < m; k++) {
        float *data = x + k * n;
        float s = data[0];
        for (int i = 1; i < n; i++) {
            s += data[i];
        }
        output[k] = s;
    }
}

// aligned vector generates vectorized load/store on CUDA
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <int vec_size, typename scalar_t>
__device__ aligned_vector<scalar_t, vec_size> load_vector(const scalar_t *base_ptr, uint32_t offset) {
  using vec_t = aligned_vector<scalar_t, vec_size>;
  auto *from = reinterpret_cast<const vec_t *>(base_ptr);
  return from[offset];
}

struct ReduceSumConfig {
    static constexpr int BLOCK_X = 0;
    static constexpr int BLOCK_Y = 1;
    static constexpr int CTA = 2;

    static constexpr int input_vec_size = 4;

    ReduceSumConfig(int num_outputs, int num_inputs)
        : num_inputs(num_inputs), num_outputs(num_outputs) {}
    int num_inputs;
    int num_outputs;
    int step_input = 1;
    int step_output = 1;
    int ctas_per_output = 1;
    int input_mult[3] = {0, 0, 0};
    int output_mult[2] = {0, 0};

    int block_width;
    int block_height;
    int num_threads;

    bool vectorize_input = false;

    void set_block_dimension(int dim0, int dim1) {
        int dim0_pow2 = dim0 < MAX_NUM_THREADS ? last_pow2(dim0) : MAX_NUM_THREADS;
        int dim1_pow2 = dim1 < MAX_NUM_THREADS ? last_pow2(dim1) : MAX_NUM_THREADS;
        block_width = std::min(dim0_pow2, WARP_SIZE);
        block_height = std::min(dim1_pow2, MAX_NUM_THREADS / block_width);
        block_width = std::min(dim0_pow2, MAX_NUM_THREADS / block_height);
        num_threads = block_width * block_height;
    }

    int split_input(int parallelism) {
        int step = step_input;
        step_input *= parallelism;
        return step;
    }

    int split_output(int parallelism) {
        int step = step_output;
        step_output *= parallelism;
        return step;
    }

    dim3 block() const {
        return dim3(block_width, block_height);
    }

    dim3 grid() const {
        return dim3(div_up(num_outputs, step_output), ctas_per_output);
    }

    __host__ __device__ bool should_block_x_reduce() const {
        return input_mult[BLOCK_X] != 0;
    }

    __host__ __device__ bool should_block_y_reduce() const {
        return input_mult[BLOCK_Y] != 0;
    }

    __host__ __device__ bool should_global_reduce() const {
        return input_mult[CTA] != 0;
    }

    __device__ bool should_store(int output_idx) const {
        return output_idx < num_outputs &&
            (!should_block_x_reduce() || threadIdx.x == 0) &&
            (!should_block_y_reduce() || threadIdx.y == 0);
    }

    __device__ bool should_reduce_tail() const {
        return (!should_block_y_reduce() || threadIdx.y == 0) &&
            (!should_global_reduce() || blockIdx.y == 0);
    }

    __host__ __device__ int input_idx() const {
        int lane = threadIdx.x;
        int warp = threadIdx.y;
        int cta2 = blockIdx.y;
        return (lane * input_mult[BLOCK_X] +
                warp * input_mult[BLOCK_Y] +
                cta2 * input_mult[CTA]);
    }

    __host__ __device__ int output_idx() const {
        int lane = threadIdx.x;
        int warp = threadIdx.y;
        int cta1 = blockIdx.x;
        return lane * output_mult[BLOCK_X] + warp * output_mult[BLOCK_Y] + cta1 * step_output;
    }

    __device__ int shared_memory_offset(int offset) const {
        return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
    }

    __device__ int staging_memory_offset(int cta2) const {
        int offset = cta2 + blockIdx.x * gridDim.y;
        if (!should_block_x_reduce()) {
            offset = threadIdx.x + offset * blockDim.x;
        }
        return offset;
    }

    int shared_memory_size() const {
        if (!should_block_y_reduce() &&
            (!should_block_x_reduce() || block_width <= WARP_SIZE)) {
            return 0;
        }
        return ELEMENT_SIZE_BYTES * num_threads;
    }

    int64_t global_memory_size() const {
        if (!should_global_reduce()) {
            return 0;
        }
        auto size = (int64_t)ELEMENT_SIZE_BYTES * num_outputs * ctas_per_output;
        if (!should_block_x_reduce()) {
            size *= block().x;
        }
        return size;
    }

    int semaphore_size() const {
        if (!should_global_reduce()) {
            return 0;
        }
        return sizeof(int) * grid().x;
    }

    int values_per_thread() const {
        return div_up(num_inputs, step_input);
    }
};

template<int vt0=4>
struct ReduceSumOp {
    static constexpr int input_vec_size = ReduceSumConfig::input_vec_size;

    ReduceSumConfig config;
    const float *x;
    const int m, n;
    float *output;
    float *buffer;
    int *semaphores;

    ReduceSumOp(
        ReduceSumConfig config,
        const float *x,
        int m,
        int n,
        float *output,
        float *buffer,
        int *semaphores)
        : config(config),
          x(x),
          m(m),
          n(n),
          output(output),
          buffer(buffer),
          semaphores(semaphores) {
    }

    __device__ void run() const {
        extern __shared__ float shared[];

        int output_idx = config.output_idx();
        int input_idx = config.input_idx();

        float value;
        if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
            const float *input_slice = x + output_idx * n;
            value = thread_reduce(input_slice);
        }
        if (config.should_block_y_reduce()) {
            value = block_y_reduce(value, shared);
        }
        if (config.should_block_x_reduce()) {
            value = block_x_reduce(value, shared);
        }

        if (config.should_global_reduce()) {
            value = global_reduce(value, shared);
        } else if (config.should_store(output_idx)) {
            output[output_idx] = value;
        }
    }

    __device__ float thread_reduce(const float *data) const {
        if (config.vectorize_input) {
            return input_vectorized_thread_reduce_impl(data);
        } else {
            return thread_reduce_impl(data);
        }
    }

    __device__ float input_vectorized_thread_reduce_impl(const float *data) const {
        int end = config.num_inputs;

        // Handle the head of input slice where data is not aligned
        float value = 0;
        constexpr int align_bytes = alignof(aligned_vector<float, input_vec_size>);
        constexpr int align_elements = align_bytes / sizeof(float);
        int shift = ((uint64_t)data) % align_bytes / sizeof(float);
        if (shift > 0) {
            data -= shift;
            end += shift;
            if (threadIdx.x >= shift && threadIdx.x < align_elements && config.should_reduce_tail()) {
                value += data[threadIdx.x];
            }
            end -= align_elements;
            data += align_elements;
            shift = align_elements - shift;
        }

        // Do the vectorized reduction
        int idx = config.input_idx();
        const int stride = config.step_input;

        // Multiple accumulators to remove dependency between unrolled loops.
        float value_list[input_vec_size];
        value_list[0] = value;

        #pragma unroll
        for (int i = 1; i < input_vec_size; i++) {
            value_list[i] = 0;
        }

        while (idx * input_vec_size + input_vec_size - 1 < end) {
            const auto values_vec = load_vector<input_vec_size>(data, idx);
            #pragma unroll
            for (int i = 0; i < input_vec_size; i++) {
                value_list[i] += values_vec.val[i];
            }
            idx += stride;
        }

        // tail
        int tail_start = end - end % input_vec_size;
        if (config.should_reduce_tail()) {
            int idx = tail_start + threadIdx.x;
            if (idx < end) {
                value_list[0] += data[idx];
            }
        }

        // combine accumulators
        #pragma unroll
        for (int i = 1; i < input_vec_size; i++) {
            value_list[0] += value_list[i];
        }
        return value_list[0];
    }

    __device__ float thread_reduce_impl(const float *data) const {
        int idx = config.input_idx();
        const int end = config.num_inputs;
        const int stride = config.step_input;

        // Multiple accumulators to remove dependency between unrolled loops.
        float value_list[vt0];
        #pragma unroll
        for (int i = 0; i < vt0; i++) {
            value_list[i] = 0;
        }

        float values[vt0];
        while (idx + (vt0 - 1) * stride < end) {
            #pragma unroll
            for (int i = 0; i < vt0; i++) {
                values[i] = data[idx + i * stride];
            }
            #pragma unroll
            for (int i = 0; i < vt0; i++) {
                value_list[i] += values[i];
            }
            idx += stride * vt0;
        }

        // tail
        int idx_ = idx;
        #pragma unroll
        for (int i = 0; i < vt0; i++) {
            if (idx >= end) {
                break;
            }
            values[i] = data[idx];
            idx += stride;
        }
        idx = idx_;
        #pragma unroll
        for (int i = 0; i < vt0; i++) {
            if (idx >= end) {
                break;
            }
            value_list[i] += values[i];
            idx += stride;
        }

        // combine accumulators
        #pragma unroll
        for (int i = 1; i < vt0; i++) {
            value_list[0] += value_list[i];
        }
        return value_list[0];
    }

    __device__ float block_x_reduce(float value, float *shared) const {
        int dim_x = blockDim.x;
        if (dim_x > warpSize) {
            int address_base = threadIdx.x + threadIdx.y * blockDim.x;
            shared[address_base] = value;
            for (int offset = dim_x / 2; offset >= warpSize; offset >>= 1) {
                __syncthreads();
                if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
                    float other = shared[address_base + offset];
                    value += other;
                    shared[address_base] = value;
                }
            }
            dim_x = warpSize;
        }

        __syncthreads();

        for (int offset = 1; offset < dim_x; offset <<= 1) {
            float other = __shfl_down_sync(0xffffffff, value, offset);
            value += other;
        }
        return value;
    }

    __device__ float block_y_reduce(float value, float *shared) const {
        shared[config.shared_memory_offset(0)] = value;
        for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
            __syncthreads();
            if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
                float other = shared[config.shared_memory_offset(offset)];
                value += other;
                shared[config.shared_memory_offset(0)] = value;
            }
        }
        return value;
    }

    __device__ bool mark_block_finished() const {
        __shared__ bool is_last_block_done_shared;
    
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
            is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
        }
    
        __syncthreads();
    
        return is_last_block_done_shared;
    }

    __device__ float global_reduce(float value, float *shared) const {
        int output_idx = config.output_idx();
        
        bool should_store = config.should_store(output_idx);
        if (should_store) {
            int offset = config.staging_memory_offset(blockIdx.y);
            buffer[offset] = value;
        }

        __threadfence(); // make sure writes are globally visible
        __syncthreads(); // if multiple warps in this block wrote to staging, make sure they're all done
        bool is_last_block_done = mark_block_finished();

        if (is_last_block_done) {
            __threadfence(); // complete the acquire pattern after atomic
            value = 0;
            if (config.should_block_x_reduce()) {
                int input_offset = threadIdx.x + threadIdx.y * blockDim.x;
                int step = blockDim.x * blockDim.y;
                for (; input_offset < config.ctas_per_output; input_offset += step) {
                    int idx = config.staging_memory_offset(input_offset);
                    float next = buffer[idx];
                    value += next;
                }
            } else {
                int input_offset = threadIdx.y;
                int step = blockDim.y;
                for (; input_offset < config.ctas_per_output; input_offset += step) {
                    int idx = config.staging_memory_offset(input_offset);
                    float next = buffer[idx];
                    value += next;
                }
            }
            value = block_y_reduce(value, shared);
            if (config.should_block_x_reduce()) {
                value = block_x_reduce(value, shared);
            }
            if (should_store) {
                output[output_idx] = value;
            }
        }
        return value;
    }
};

template<typename R>
__global__ void reduce_kernel(R reduction) {
    reduction.run();
}

template<typename R>
static void launch_reduce_kernel(const ReduceSumConfig& config, const R& reduction, cudaStream_t stream) {
    dim3 block = config.block();
    dim3 grid = config.grid();
    int shared_memory = config.shared_memory_size();
    reduce_kernel<R><<<grid, block, shared_memory, stream>>>(reduction);
}

template<int vt0>
ReduceSumConfig setReduceSumConfig(int m, int n) {
    // Start by assuming that each thread handles a single output and all
    // the inputs for that output.
    ReduceSumConfig config(m, n);

    int dim0 = n;
    int dim1 = m;

    // We do vectorization to gain better memory access.
    // We are reducing along fastest moving dimesion. Threads with the same threadIdx.y works
    // on the same reduction cooperatively and will produce results for the same output.
    // In such case, values in each loaded vector always correspond to the same output.
    if (dim0 > 128 && vt0 >= ReduceSumConfig::input_vec_size) {
        // Note that if vt0 < input_vec_size, then this means the register pressure could be high, in such case,
        // we should avoid vectorization.
        config.vectorize_input = true;
        dim0 /= config.input_vec_size;
    }

    // Adjust block_width and block_height
    config.set_block_dimension(dim0, dim1);

    int block_width = config.block_width;
    int block_height = config.block_height;

    // Split the input across lanes if the input is contiguous in the reduced
    // dimension. This will require reduction between threads using warp
    // shuffle instructions and shared memory (if block_width > warpSize).
    config.input_mult[0] = config.split_input(block_width);

    constexpr int min_values_per_thread = 16;
    constexpr int max_values_per_thread = 256;

    if (config.values_per_thread() >= block_height * 16 || config.values_per_thread() >= max_values_per_thread) {
        // Divide the input across warps in a thread-block, if that leaves at least
        // 16 elements to be summed by each thread. This will require inter-warp
        // reduction using shared memory.
        config.input_mult[1] = config.split_input(block_height);
    } else {
        // Otherwise, each warp handles a separate output.
        config.output_mult[1] = config.split_output(block_height);
    }

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    const int num_mp = prop.multiProcessorCount;
    const int blocks_per_sm = prop.maxThreadsPerMultiProcessor / config.num_threads;
    const int target_grid_size = num_mp * blocks_per_sm;
    int grid = config.grid().x;
    if (config.input_mult[1] != 0 && config.values_per_thread() >= max_values_per_thread && grid <= target_grid_size) {
        // Divide the input across thread-blocks if the amount of work per-thread
        // is large enough and the size of the output is small enough. This will
        // require a reduction using global memory.
        // If we decide to split input across blocks, as long as we can get enough
        // number of blocks (`target_grid_size`) to balance SM, we should still
        // make the number of values per thread large for best performance.
        int ctas_per_output1 = div_up(target_grid_size, grid);
        int ctas_per_output2 = div_up(config.values_per_thread(), min_values_per_thread);
        int ctas_per_output3 = div_up(config.values_per_thread(), max_values_per_thread);
        // We want the minimum of ctas_per_output1 and ctas_per_output2, so that each thread can have
        // a large number of values to deal with. But we don't want values_per_thread to be larger than
        // max_values_per_thread
        config.ctas_per_output = std::max(std::min<int>(ctas_per_output1, ctas_per_output2), ctas_per_output3);
        if (config.ctas_per_output > 1) {
            config.input_mult[2] = config.split_input(config.ctas_per_output);
        }
    }
    return config;
}

template<int vt0=4>
inline void gpu_reduce_sum(float *x, int m, int n, float *output, cudaStream_t stream) {
    ReduceSumConfig config = setReduceSumConfig<vt0>(m, n);

    float *buffer = nullptr;
    int *semaphores = nullptr;
    if (config.should_global_reduce()) {
        cudaMalloc(&buffer, config.global_memory_size());
        cudaMalloc(&semaphores, config.semaphore_size());
        cudaMemsetAsync(semaphores, 0, config.semaphore_size(), stream);
    }

    auto reduce = ReduceSumOp<vt0>(
        config,
        x,
        m,
        n,
        output,
        buffer,
        semaphores);

    launch_reduce_kernel(config, reduce, stream);

    if (config.should_global_reduce()) {
        cudaFreeAsync(buffer, stream);
        cudaFreeAsync(semaphores, stream);
    }
}

template<typename FN>
void warmup(FN fn, int ms) {
    auto start = std::chrono::steady_clock::now();
    while (true) {
        fn();
        auto end = std::chrono::steady_clock::now();
        if (end - start > std::chrono::milliseconds(ms)) {
            break;
        }
    }
}

bool test_reduce_sum(int m, int n) {
    const int T = 100;

    float *x_cpu = new float[m * n];
    float *output_cpu = new float[m];

    // generate random floats
    for (int i = 0; i < m * n; i++) {
        x_cpu[i] = rand() / float(RAND_MAX) - 0.5;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *x, *output;
    cudaMalloc(&x, m * n * sizeof(float));
    cudaMalloc(&output, m * sizeof(float));
    cudaMemcpyAsync(x, x_cpu, m * n * sizeof(float), cudaMemcpyHostToDevice, stream);

    float *foo;
    const int L = 100'000'000;
    cudaMalloc(&foo, L * sizeof(float));

    warmup([&]() {
        cudaMemset(foo, 0, L * sizeof(float));
        gpu_reduce_sum(x, m, n, output, stream);
    }, 250);

    cudaEvent_t start[T], end[T];
    for (int t = 0; t < T; t++) {
        cudaEventCreate(&start[t]);
        cudaEventCreate(&end[t]);
    }
    for (int t = 0; t < T; t++) {
        cudaMemset(foo, 0, L * sizeof(float));
        cudaEventRecord(start[t], stream);
        gpu_reduce_sum(x, m, n, output, stream);
        cudaEventRecord(end[t], stream);
    }
    cudaStreamSynchronize(stream);

    float gpu_time = 0;
    for (int t = 0; t < T; t++) {
        float cur_time;
        cudaEventElapsedTime(&cur_time, start[t], end[t]);
        gpu_time += cur_time;
    }
    std::cout << "GPU Version: " << gpu_time / T << " ms" << std::endl;
    for (int t = 0; t < T; t++) {
        cudaEventDestroy(start[t]);
        cudaEventDestroy(end[t]);
    }

    cudaMemcpyAsync(output_cpu, output, m * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float *ans_cpu = new float[m];
    warmup([&]() {
        cpu_reduce_sum(x_cpu, m, n, ans_cpu);
    }, 250);
    clock_t start_cpu = clock();
    for (int _ = 0; _ < T; _++) {
        cpu_reduce_sum(x_cpu, m, n, ans_cpu);
    }
    clock_t end_cpu = clock();
    float cpu_time = (end_cpu - start_cpu) / (float)CLOCKS_PER_SEC * 1000 / T;
    std::cout << "CPU Version: " << cpu_time << " ms" << std::endl;

    bool success = true;
    for (int i = 0; i < m; i++) {
        if (std::fabs(output_cpu[i] - ans_cpu[i]) > 1e-6 * n) {
            std::cout << "Error: " << output_cpu[i] << " != " << ans_cpu[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Success" << std::endl;
    }

    cudaFreeAsync(foo, stream);
    cudaFreeAsync(x, stream);
    cudaFreeAsync(output, stream);
    cudaStreamDestroy(stream);
    delete[] x_cpu;
    delete[] output_cpu;
    delete[] ans_cpu;

    return success;
}

int main() {
    std::pair<int, int> sizes[] = {
        {1, 1},
        {1, 32},
        {1, 800},
        {1, 1024},
        {1, 2000},
        {1, 10000},
        {1, 30000},
        {1, 100000},
        {1, 150000},
        {1, 300000},
        {1, 1000000},
        {1, 3000000},
        {1, 10000000},
        {1, 100000000},
        {2, 1},
        {10, 3200},
        {32, 3000000},
        {1024, 32},
        {100000, 32},
        {1000000, 8},
        {10000, 3200},
    };

    for (auto [m, n] : sizes) {
        std::cout << "Testing size: " << m << "x" << n << std::endl;
        if (!test_reduce_sum(m, n)) {
            break;
        }
    }

    return 0;
}
