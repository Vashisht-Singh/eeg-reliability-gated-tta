"""
Abstract Base Class for Test-Time Adaptation (TTA).
Enforces the strict research protocol:
Predict First -> Update Second -> Never Peek Ahead.
"""

import abc
import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


class BaseAdapter(abc.ABC):
    """
    Abstract interface for chronological online test-time adaptation methods.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-4, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.source_model_state = copy.deepcopy(model.state_dict())
        self.lr = lr
        self.total_steps = 0
        self.total_updates = 0

    @abc.abstractmethod
    def adapt(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Perform an unsupervised parameter update step.
        Returns dictionary of update metrics (loss, entropy, updated_flag).
        """
        pass

    def predict(self, x: torch.Tensor) -> float:
        """
        Compute prediction using incumbent model state.
        MUST be called before any update is applied to the current window.
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = x.to(self.device)
            if x_tensor.ndim == 2:
                x_tensor = x_tensor.unsqueeze(0)  # (1, 18, 1024)
            prob = self.model.predict_proba(x_tensor).item()
        return float(prob)

    def step(self, x: torch.Tensor, timestamp: float) -> Tuple[float, Dict[str, Any]]:
        """
        Execute one chronological streaming step:
        1. Predict first (incumbent prediction).
        2. Adapt second (unsupervised parameter adjustment).
        """
        # Step 1: Predict first
        prob = self.predict(x)

        # Step 2: Adapt second
        adapt_info = self.adapt(x)
        self.total_steps += 1
        if adapt_info.get('updated', False):
            self.total_updates += 1

        adapt_info['incumbent_prediction'] = prob
        adapt_info['timestamp'] = timestamp
        return prob, adapt_info

    def reset_to_source(self):
        """Restore model weights to original source checkpoint."""
        self.model.load_state_dict(copy.deepcopy(self.source_model_state))
