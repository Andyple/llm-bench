"""
Memory sampling utilities.
Supports NVIDIA GPUs via GPUtil and falls back to psutil RAM
(useful for unified-memory systems like Strix Halo where VRAM = system RAM).
"""
from __future__ import annotations
from typing import Tuple
import psutil

try:
    import GPUtil
    _GPUTIL_AVAILABLE = True
except ImportError:
    _GPUTIL_AVAILABLE = False


def sample_memory(prefer_gpu: bool = True) -> Tuple[float, float]:
    """
    Returns (used_mb, total_mb).
    Tries GPU first (if available and prefer_gpu=True), falls back to system RAM.
    For Strix Halo with unified memory, system RAM is the right metric.
    """
    if prefer_gpu and _GPUTIL_AVAILABLE:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return gpu.memoryUsed, gpu.memoryTotal

    # Fallback: system RAM (unified memory / CPU inference)
    vm = psutil.virtual_memory()
    return vm.used / (1024 ** 2), vm.total / (1024 ** 2)
