"""
EATA: Efficient Test-Time Model Adaptation without Forgetting (Niu et al., ICML 2022).
Implements reliable sample selection via entropy margin, redundancy filtering,
and anti-forgetting parameter regularization.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Any, List

from src.adaptation.base_adapter import BaseAdapter
from src.adaptation.tent_adapter import binary_entropy


class EataAdapter(BaseAdapter):
    """
    EATA implementation adhering to the original ICML 2022 formulation.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        entropy_margin_e0: float = 0.40,
        fisher_weight: float = 200.0,
        redundancy_threshold: float = 0.05,
        device: str = "cuda"
    ):
        super().__init__(model, lr, device)
        self.e0 = entropy_margin_e0
        self.fisher_weight = fisher_weight
        self.redundancy_threshold = redundancy_threshold

        # Freeze non-adaptable layers
        for p in self.model.parameters():
            p.requires_grad = False

        self.adaptable_params = self.model.get_adaptable_parameters()
        for p in self.adaptable_params:
            p.requires_grad = True

        self.optimizer = torch.optim.Adam(self.adaptable_params, lr=self.lr)

        # Store initial parameters for anti-forgetting regularization
        self.initial_params = [p.clone().detach() for p in self.adaptable_params]

        # Moving average probability vector for redundancy check
        self.moving_avg_prob = 0.5
        self.num_samples_adapted = 0

    def adapt(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Active sample selection and anti-forgetting adaptation.
        """
        self.model.train()
        x_tensor = x.to(self.device)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)

        with torch.no_grad():
            logits_eval = self.model(x_tensor)
            prob_eval = float(torch.sigmoid(logits_eval).item())
            ent_eval = float(binary_entropy(torch.sigmoid(logits_eval)).item())

        # 1. Reliable Sample Selection: Entropy Margin Filter
        if ent_eval >= self.e0:
            return {
                'updated': False,
                'reason': f'entropy_too_high_{ent_eval:.3f}_ge_{self.e0}',
                'entropy': ent_eval,
                'method': 'EATA'
            }

        # 2. Redundancy Filtering (Cosine similarity with moving average probability)
        # For binary: absolute difference between current probability and running probability
        redundancy_dist = abs(prob_eval - self.moving_avg_prob)
        if redundancy_dist < self.redundancy_threshold and self.num_samples_adapted > 10:
            return {
                'updated': False,
                'reason': f'redundant_sample_dist_{redundancy_dist:.4f}',
                'entropy': ent_eval,
                'method': 'EATA'
            }

        # 3. Perform Update
        self.optimizer.zero_grad()
        logits = self.model(x_tensor)
        prob = torch.sigmoid(logits)
        ent_loss = binary_entropy(prob).mean()

        # 4. Anti-Forgetting Parameter Penalty: beta * ||theta - theta_0||^2
        fisher_loss = 0.0
        for p, p_init in zip(self.adaptable_params, self.initial_params):
            fisher_loss += torch.sum((p - p_init) ** 2)
        total_loss = ent_loss + self.fisher_weight * fisher_loss

        total_loss.backward()
        self.optimizer.step()

        # Update running probability average
        self.moving_avg_prob = 0.9 * self.moving_avg_prob + 0.1 * prob_eval
        self.num_samples_adapted += 1

        return {
            'updated': True,
            'adaptation_loss': float(total_loss.item()),
            'entropy': ent_eval,
            'entropy_loss': float(ent_loss.item()),
            'fisher_loss': float(fisher_loss.item()),
            'method': 'EATA'
        }
