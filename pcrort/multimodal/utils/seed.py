"""Deterministic seeding for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set deterministic seeds across all libraries.

    Sets seeds for: Python random, NumPy, PyTorch (CPU + CUDA).
    Also enables deterministic CuDNN behavior.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch >= 1.8
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch 2.0+ deterministic mode
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch versions don't support warn_only
        pass


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the best available device.

    Args:
        use_cuda: Whether to prefer CUDA if available.

    Returns:
        torch.device for CPU or CUDA.
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def print_device_info(device: torch.device) -> None:
    """Print detailed device information."""
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("  Running on CPU (no CUDA available)")
