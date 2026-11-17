"""
Memory Manager for Resource-Constrained EEG Deep Learning Workflows.
Monitors host RAM and GPU VRAM, enforces garbage collection, and logs memory states.
"""

import os
import gc
import psutil
import torch
import logging

logger = logging.getLogger(__name__)


class MemoryTracker:
    """Tracks and logs system RAM and GPU memory usage during training and inference."""

    def __init__(self, warning_threshold_ram_pct: float = 85.0):
        self.process = psutil.Process(os.getpid())
        self.warning_threshold_ram_pct = warning_threshold_ram_pct

    def get_ram_usage(self) -> dict:
        """Returns process and system RAM metrics in Megabytes."""
        mem_info = self.process.memory_info()
        sys_mem = psutil.virtual_memory()
        return {
            "process_rss_mb": round(mem_info.rss / (1024 ** 2), 2),
            "process_vms_mb": round(mem_info.vms / (1024 ** 2), 2),
            "system_used_pct": sys_mem.percent,
            "system_available_mb": round(sys_mem.available / (1024 ** 2), 2),
        }

    def check_memory(self, context_str: str = "") -> dict:
        """
        Inspects memory, logs usage, and triggers collection if nearing threshold.
        """
        ram = self.get_ram_usage()
        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)

        if ram["system_used_pct"] > self.warning_threshold_ram_pct:
            logger.warning(
                f"[MEM WARNING] {context_str} - System RAM at {ram['system_used_pct']}% "
                f"(Avail: {ram['system_available_mb']} MB). Forcing garbage collection."
            )
            self.collect()

        return {**ram, "gpu_allocated_mb": vram_mb}

    def collect(self) -> None:
        """Executes full Python garbage collection and PyTorch cache cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global tracker instance
memory_tracker = MemoryTracker()
