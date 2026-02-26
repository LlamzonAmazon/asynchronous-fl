"""
Reproducible random seeding utilities.

Sets Python, NumPy, and PyTorch RNGs to a fixed seed so that experiments
are repeatable across centralized, synchronous FL, and asynchronous FL.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
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

