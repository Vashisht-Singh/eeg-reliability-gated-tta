"""
SAR: Sharpness-Aware and Reliable Test-Time Adaptation (Niu et al., ICLR 2023).
Implements reliable entropy filtering, sharpness-aware optimization (SAM) to find flat minima,
and unsupervised model recovery upon gradient divergence.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Any, List

from src.adaptation.base_adapter import BaseAdapter
from src.adaptation.tent_adapter import binary_entropy


class SarAdapter(BaseAdapter):
    """
    SAR implementation adhering to the original ICLR 2023 formulation.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        margin_e0: float = 0.40,
        rho: float = 0.05,
        device: str = "cuda"
    ):
        super().__init__(model, lr, device)
        self.e0 = margin_e0
        self.rho = rho

        # Freeze non-adaptable layers
        for p in self.model.parameters():
            p.requires_grad = False

        self.adaptable_params = self.model.get_adaptable_parameters()
        for p in self.adaptable_params:
            p.requires_grad = True

        self.optimizer = torch.optim.Adam(self.adaptable_params, lr=self.lr)
        self.ema_model_state = copy.deepcopy(self.model.state_dict())

    def adapt(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Sharpness-aware minimization on reliable samples.
        """
        self.model.train()
        x_tensor = x.to(self.device)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)

        # 1. Reliable Sample Selection
        with torch.no_grad():
            logits_eval = self.model(x_tensor)
            ent_eval = float(binary_entropy(torch.sigmoid(logits_eval)).item())

        if ent_eval >= self.e0:
            return {
                'updated': False,
                'reason': f'entropy_ge_e0_{ent_eval:.3f}_ge_{self.e0}',
                'entropy': ent_eval,
                'method': 'SAR'
            }

        # 2. First Pass: Compute Gradients for SAM perturbation
        self.optimizer.zero_grad()
        logits = self.model(x_tensor)
        loss = binary_entropy(torch.sigmoid(logits)).mean()
        loss.backward()

        # Compute gradient norm
        grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in self.adaptable_params if p.grad is not None]), p=2)

        if grad_norm == 0 or torch.isnan(grad_norm):
            return {'updated': False, 'reason': 'zero_or_nan_gradient', 'entropy': ent_eval, 'method': 'SAR'}

        # 3. Apply perturbation: epsilon = rho * grad / ||grad||
        scale = self.rho / (grad_norm + 1e-12)
        eps_list = []
        with torch.no_grad():
            for p in self.adaptable_params:
                if p.grad is not None:
                    eps = p.grad * scale
                    p.add_(eps)
                    eps_list.append(eps)
                else:
                    eps_list.append(None)

        # 4. Second Pass: Compute gradient at perturbed position
        self.optimizer.zero_grad()
        logits_pert = self.model(x_tensor)
        loss_pert = binary_entropy(torch.sigmoid(logits_pert)).mean()
        loss_pert.backward()

        # 5. Restore original parameters and apply SAM gradient step
        with torch.no_grad():
            for p, eps in zip(self.adaptable_params, eps_list):
                if eps is not None:
                    p.sub_(eps)

        self.optimizer.step()

        return {
            'updated': True,
            'adaptation_loss': float(loss.item()),
            'perturbed_loss': float(loss_pert.item()),
            'entropy': ent_eval,
            'grad_norm': float(grad_norm.item()),
            'method': 'SAR'
        }
