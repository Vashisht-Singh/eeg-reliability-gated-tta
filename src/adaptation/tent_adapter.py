"""
TENT: Test-Time Adaptation by Entropy Minimization (Wang et al., ICLR 2021).
Optimizes affine parameters of normalization layers by minimizing prediction entropy.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

from src.adaptation.base_adapter import BaseAdapter


def binary_entropy(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute Shannon entropy for binary probabilities."""
    prob = torch.clamp(prob, eps, 1.0 - eps)
    return -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob))


class TentAdapter(BaseAdapter):
    """
    TENT baseline implementation strictly adhering to original methodology.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-4, device: str = "cuda"):
        super().__init__(model, lr, device)

        # 1. Freeze all parameters except normalization affine layers
        for p in self.model.parameters():
            p.requires_grad = False

        self.adaptable_params = self.model.get_adaptable_parameters()
        for p in self.adaptable_params:
            p.requires_grad = True

        # 2. Configure optimizer exclusively over adaptable parameters
        self.optimizer = torch.optim.Adam(self.adaptable_params, lr=self.lr)

    def adapt(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Adapt normalization layers using current window via entropy minimization.
        """
        self.model.train()  # Normalization layers in train mode for affine tracking
        x_tensor = x.to(self.device)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)

        self.optimizer.zero_grad()
        logits = self.model(x_tensor)
        prob = torch.sigmoid(logits)
        loss = binary_entropy(prob).mean()

        loss.backward()
        self.optimizer.step()

        return {
            'updated': True,
            'adaptation_loss': float(loss.item()),
            'entropy': float(binary_entropy(prob).mean().item()),
            'method': 'TENT'
        }
