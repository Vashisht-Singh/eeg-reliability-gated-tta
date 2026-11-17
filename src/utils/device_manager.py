"""
Device Manager for PyTorch Hardware Acceleration.
Detects available GPU devices (NVIDIA CUDA), manages device placement,
and provides VRAM diagnostics.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


def get_device(requested: str = "auto") -> torch.device:
    """
    Resolves the execution device.
    
    Args:
        requested (str): 'cuda', 'cpu', or 'auto'.
        
    Returns:
        torch.device: Resolved PyTorch device.
    """
    if requested == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    else:
        device_str = requested

    device = torch.device(device_str)
    
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(
            f"Active GPU: {props.name} | Total Memory: {props.total_memory / (1024**2):.1f} MB | "
            f"Compute Capability: {props.major}.{props.minor}"
        )
    else:
        logger.info("Executing on CPU.")

    return device


def get_gpu_memory_info() -> dict:
    """
    Returns allocated, reserved, and maximum VRAM in Megabytes.
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2),
    }


def empty_gpu_cache() -> None:
    """Clears unreferenced GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
