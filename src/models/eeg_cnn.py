"""
Primary Deep Learning Architecture: 1D Temporal Convolutional Neural Network (EEG1DCNN).
Optimized for multi-channel scalp EEG seizure detection with dynamic channel/sample support
and explicit adaptable parameter grouping for test-time adaptation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class ConvBlock1D(nn.Module):
    """Temporal convolution block with batch normalization, activation, and pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        pool_size: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        padding = kernel_size // 2  # Same padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size) if pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class EEG1DCNN(nn.Module):
    """
    Lightweight, research-grade 1D CNN for multi-channel EEG classification.
    Input shape: (batch_size, num_channels, input_samples) e.g. (B, 18, 1024)
    Output shape: (batch_size, 1) raw logit for binary seizure detection
    """

    def __init__(
        self,
        num_channels: int = 18,
        input_samples: int = 1024,
        conv_channels: List[int] = [32, 64, 128, 256],
        kernel_sizes: List[int] = [15, 11, 7, 5],
        pool_sizes: List[int] = [2, 2, 2, 4],
        dropouts: List[float] = [0.2, 0.3, 0.3, 0.4],
        fc_dim: int = 128
    ):
        super().__init__()
        self.num_channels = num_channels
        self.input_samples = input_samples

        # Feature extractor blocks
        layers = []
        in_c = num_channels
        for out_c, k_size, p_size, drop in zip(conv_channels, kernel_sizes, pool_sizes, dropouts):
            layers.append(ConvBlock1D(in_c, out_c, kernel_size=k_size, pool_size=p_size, dropout=drop))
            in_c = out_c
        self.features = nn.Sequential(*layers)

        # Global temporal pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[-1], fc_dim, bias=True),
            nn.LayerNorm(fc_dim),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(fc_dim, 1)  # Binary logit output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (B, num_channels, input_samples)
        Returns:
            logits: (B, 1)
        """
        feat = self.features(x)
        pooled = self.global_pool(feat).squeeze(-1)  # (B, conv_channels[-1])
        logits = self.classifier(pooled)             # (B, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid calibrated probability of seizure."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def get_adaptable_parameters(self) -> List[nn.Parameter]:
        """
        Collects feature extractor BatchNorm affine parameters (weights and biases) for TTA.
        Standard protocol for TENT, EATA, and SAR.
        """
        params = []
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    params.append(m.weight)
                if m.bias is not None:
                    params.append(m.bias)
        return params

    def get_parameter_count(self) -> Dict[str, int]:
        """Return total, trainable, and adaptable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adaptable = sum(p.numel() for p in self.get_adaptable_parameters())
        return {
            'total_parameters': total,
            'trainable_parameters': trainable,
            'adaptable_parameters': adaptable
        }
