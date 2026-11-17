"""
Loss Functions for Imbalanced EEG Seizure Detection.
Provides class-weighted BCE and focal loss for severe class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with extreme class imbalance:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view_as(logits).float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def get_loss_function(name: str = "BCEWithLogitsLoss", pos_weight: float = 1.0) -> nn.Module:
    """
    Instantiate configured loss function with class imbalance weights.
    """
    name_clean = name.strip().lower()
    if "focal" in name_clean:
        return FocalLoss(alpha=0.75, gamma=2.0)
    else:
        # Standard BCE with positive class weight
        pw_tensor = torch.tensor([pos_weight], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
