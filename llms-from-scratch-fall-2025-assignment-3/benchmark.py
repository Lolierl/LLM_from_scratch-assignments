import torch
import statistics
from typing import Callable, Tuple

def benchmark_cuda_fn(
    fn: Callable[[], None],
    n_warmup: int = 10,
    n_runs: int = 50,
) -> Tuple[float, float]:
    """
    Benchmark the execution time of a CUDA function.

    Args:
        fn: A callable that performs one full training step
            (forward + loss + backward + optimizer step).
            It must run on CUDA.
        n_warmup: Number of warmup runs.
        n_runs: Number of timed runs.

    Returns:
        mean_time_ms: Mean execution time in milliseconds.
        std_time_ms: Standard deviation of execution time in milliseconds.
    """

    assert torch.cuda.is_available(), "CUDA is required for this benchmark."

    # -----------------------
    # Warmup
    # -----------------------
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # -----------------------
    # Timed runs
    # -----------------------
    times_ms = []

    for _ in range(n_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))  # milliseconds

    # -----------------------
    # Statistics
    # -----------------------
    mean_time_ms = statistics.mean(times_ms)
    std_time_ms = statistics.stdev(times_ms) if n_runs > 1 else 0.0

    return mean_time_ms, std_time_ms
