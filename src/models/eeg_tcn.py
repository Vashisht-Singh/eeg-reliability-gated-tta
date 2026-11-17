"""
Comparative Deep Learning Architecture: Lightweight Temporal Convolutional Network (EEG-TCN).
Implements dilated causal 1D convolutions with residual connections for sequential EEG processing.
"""

import torch
import torch.nn as nn
from typing import Dict, List


class ChausalConv1dBlock(nn.Module):
    """Causal dilated convolution with weight norm, activation, and residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               dilation=dilation, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               dilation=dilation, padding=self.padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv 1: remove future padding to enforce causality
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        # Conv 2: remove future padding
        out2 = self.conv2(out)
        if self.padding > 0:
            out2 = out2[:, :, :-self.padding]
        out2 = self.bn2(out2)
        out2 = self.act2(out2)
        out2 = self.drop2(out2)

        res = self.downsample(x)
        return nn.functional.gelu(out2 + res)


class EEGTCN(nn.Module):
    """
    Lightweight Causal TCN for multi-channel EEG classification.
    """

    def __init__(
        self,
        num_channels: int = 18,
        input_samples: int = 1024,
        num_channels_list: List[int] = [32, 64, 128, 256],
        kernel_size: int = 5,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_channels = num_channels
        self.input_samples = input_samples

        layers = []
        in_c = num_channels
        for i, out_c in enumerate(num_channels_list):
            dilation = 2 ** i
            layers.append(ChausalConv1dBlock(in_c, out_c, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            in_c = out_c

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels_list[-1], 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.network(x)
        pooled = self.global_pool(feat).squeeze(-1)
        logits = self.classifier(pooled)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def get_adaptable_parameters(self) -> List[nn.Parameter]:
        params = []
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    params.append(m.weight)
                if m.bias is not None:
                    params.append(m.bias)
        return params

    def get_parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adaptable = sum(p.numel() for p in self.get_adaptable_parameters())
        return {
            'total_parameters': total,
            'trainable_parameters': trainable,
            'adaptable_parameters': adaptable
        }
