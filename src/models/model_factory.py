"""
Model Factory for Deep Learning Architectures.
Provides centralized instantiation of baseline and comparative EEG architectures.
"""

import torch.nn as nn
from src.models.eeg_cnn import EEG1DCNN
from src.models.eeg_tcn import EEGTCN


def create_model(model_name: str = "EEG1DCNN", num_channels: int = 18, input_samples: int = 1024, **kwargs) -> nn.Module:
    """
    Factory function to instantiate models by configuration name.
    """
    name_clean = model_name.strip().upper()
    if name_clean in ["EEG1DCNN", "1DCNN", "CNN"]:
        return EEG1DCNN(num_channels=num_channels, input_samples=input_samples, **kwargs)
    elif name_clean in ["EEGTCN", "TCN"]:
        return EEGTCN(num_channels=num_channels, input_samples=input_samples, **kwargs)
    else:
        raise ValueError(f"Unknown model architecture: '{model_name}'. Available: 'EEG1DCNN', 'EEGTCN'")
