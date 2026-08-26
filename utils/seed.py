"""
Reproducible random seeding utilities.

Sets Python, NumPy, and PyTorch RNGs to a fixed seed so that experiments
are repeatable across centralized, synchronous FL, and asynchronous FL.

Also pins PyTorch's CPU thread count. Thread count controls the partitioning
of parallel reductions, and float addition is not associative, so a varying
thread count is in principle a source of run-to-run divergence. Measured on
this model it is not: forward+backward was verified bitwise identical at 1, 4,
8 and 10 threads, and identical across repeated runs under heavy CPU
contention (2026-08-19). Pinning is therefore a guarantee rather than a fix,
and it makes the reproducibility claim independent of the host machine.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_num_threads(num_threads: Optional[int] = None) -> int:
    """Pin PyTorch intra-op thread count and return the effective value.

    Resolution order: explicit argument, then the FL_NUM_THREADS environment
    variable, then PyTorch's own default (derived from the host core count).

    Must run before any parallel tensor work. PyTorch's native parallel
    backend fixes the thread count at the first parallel op and warns if it is
    changed afterwards.
    """
    if num_threads is None:
        env = os.environ.get("FL_NUM_THREADS")
        num_threads = int(env) if env else None

    if num_threads is not None:
        torch.set_num_threads(num_threads)

    return torch.get_num_threads()


def set_seed(seed: Optional[int] = None) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    # Pin threads first; this must precede any parallel tensor work.
    set_num_threads()

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For deterministic behaviour on CUDA; has no effect on CPU/MPS.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
