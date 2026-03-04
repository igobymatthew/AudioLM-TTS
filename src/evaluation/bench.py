import time


def benchmark_tokens_per_sec(n_tokens: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return n_tokens / elapsed_s


def timed(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0
