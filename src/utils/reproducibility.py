"""
Reproducibility Utilities for EEG Research.
Ensures deterministic behavior across Python, NumPy, PyTorch, and CUDA.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across all libraries to enforce deterministic execution.
    
    Args:
        seed (int): The integer random seed (default: 42).
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Guarantees deterministic convolutional algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_generator(seed: int = 42) -> torch.Generator:
    """
    Returns a seeded PyTorch Generator for DataLoader reproducibility.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
