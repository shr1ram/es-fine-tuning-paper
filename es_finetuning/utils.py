"""Utilities for ES fine-tuning."""

import gc
import os

import torch


def force_memory_cleanup():
    """Force aggressive GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def setup_tensorboard(log_dir):
    """Create a TensorBoard SummaryWriter, creating the directory if needed.

    Args:
        log_dir: Path for TensorBoard log files.

    Returns:
        SummaryWriter instance.
    """
    from torch.utils.tensorboard import SummaryWriter

    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)
