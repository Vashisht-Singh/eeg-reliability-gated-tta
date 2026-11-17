"""
Tensor and Data Validation Utilities for EEG Pipeline Integrity.
Guarantees clean tensors, checks against NaNs/Infs, and validates montage dimensions.
"""

import numpy as np
import torch
from typing import Union


def validate_eeg_window(
    window: Union[np.ndarray, torch.Tensor],
    expected_channels: int = 18,
    expected_samples: int = 1024,
    allow_empty: bool = False
) -> None:
    """
    Validates that an EEG window has the expected shape and contains no NaNs or Infs.
    
    Raises:
        ValueError: If shape is incorrect, or if values contain NaNs/Infs.
    """
    if isinstance(window, np.ndarray):
        if window.ndim != 2:
            raise ValueError(f"Expected 2D array (channels, samples), got shape {window.shape}")
        c, s = window.shape
        if c != expected_channels or s != expected_samples:
            raise ValueError(f"Expected shape ({expected_channels}, {expected_samples}), got {window.shape}")
        if np.isnan(window).any():
            raise ValueError("EEG window contains NaN values.")
        if np.isinf(window).any():
            raise ValueError("EEG window contains infinite values.")
        if not allow_empty and np.all(window == 0):
            raise ValueError("EEG window contains only zeros (dead/flatline).")
            
    elif isinstance(window, torch.Tensor):
        if window.ndim == 2:
            c, s = window.shape
        elif window.ndim == 3:  # Batch dimension (B, C, S)
            _, c, s = window.shape
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {window.shape}")
            
        if c != expected_channels or s != expected_samples:
            raise ValueError(f"Expected ({expected_channels}, {expected_samples}) along channel/sample axes, got {window.shape}")
        if torch.isnan(window).any():
            raise ValueError("EEG tensor contains NaN values.")
        if torch.isinf(window).any():
            raise ValueError("EEG tensor contains infinite values.")
