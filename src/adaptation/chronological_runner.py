"""
Chronological Adaptation Runner and Proposed Method Controller.
Coordinates Candidate Generation, Reliability Gating, Rollback Management,
and Step-by-Step Logging for Test-Time Adaptation.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from src.adaptation.base_adapter import BaseAdapter
from src.adaptation.tent_adapter import binary_entropy
from src.adaptation.reliability_gate import ReliabilityGate
from src.adaptation.rollback_manager import RollbackManager


class ReliabilityGatedAdapter(BaseAdapter):
    """
    Proposed Method: Seizure-Aware Reliability-Gated Test-Time Adaptation.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        entropy_min: float = 0.05,
        entropy_max: float = 0.50,
        max_divergence: float = 0.40,
        max_parameter_delta: float = 0.20,
        device: str = "cuda"
    ):
        super().__init__(model, lr, device)

        # Candidate Model (separate copy to evaluate updates safely)
        self.candidate_model = copy.deepcopy(model).to(self.device)

        # Freeze non-adaptable layers in both models
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.candidate_model.parameters():
            p.requires_grad = False

        self.incumbent_adaptable = self.model.get_adaptable_parameters()
        self.candidate_adaptable = self.candidate_model.get_adaptable_parameters()
        for p in self.candidate_adaptable:
            p.requires_grad = True

        self.candidate_optimizer = torch.optim.Adam(self.candidate_adaptable, lr=self.lr)

        self.gate = ReliabilityGate(
            entropy_min=entropy_min,
            entropy_max=entropy_max,
            max_divergence=max_divergence,
            max_parameter_delta=max_parameter_delta
        )
        self.rollback_mgr = RollbackManager(self.model)

        self.stats = {
            'total_candidates': 0,
            'accepted_updates': 0,
            'rejected_updates': 0,
            'rollbacks': 0
        }

    def adapt(self, x: torch.Tensor, quality_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generates candidate update, assesses reliability, and executes gate decision.
        """
        self.stats['total_candidates'] += 1
        quality_metrics = quality_metrics or {'is_valid_quality': 1.0}

        # 1. Sync candidate model with current incumbent model
        self.candidate_model.load_state_dict(copy.deepcopy(self.model.state_dict()))

        # 2. Generate Candidate Update
        self.candidate_model.train()
        x_tensor = x.to(self.device)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)

        self.candidate_optimizer.zero_grad()
        cand_logits = self.candidate_model(x_tensor)
        cand_prob = torch.sigmoid(cand_logits)
        cand_loss = binary_entropy(cand_prob).mean()
        cand_loss.backward()
        self.candidate_optimizer.step()

        # 3. Assess Reliability via Multi-Signal Gate
        accepted, decision_reason, signals = self.gate.evaluate_candidate(
            x_window=x_tensor,
            incumbent_model=self.model,
            candidate_model=self.candidate_model,
            candidate_params=self.candidate_adaptable,
            incumbent_params=self.incumbent_adaptable,
            quality_metrics=quality_metrics
        )

        # 4. Execute Gate Decision
        if accepted:
            # Commit candidate weights to incumbent model
            self.model.load_state_dict(copy.deepcopy(self.candidate_model.state_dict()))
            self.stats['accepted_updates'] += 1
        else:
            self.stats['rejected_updates'] += 1

        # 5. Rollback Monitoring
        needs_rollback, rb_reason = self.rollback_mgr.register_step(
            model=self.model,
            was_accepted=accepted,
            signals=signals,
            timestamp=0.0
        )
        if needs_rollback:
            to_src = "to_source" in rb_reason
            self.rollback_mgr.execute_rollback(self.model, to_source=to_src)
            self.candidate_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
            self.stats['rollbacks'] += 1

        return {
            'updated': accepted,
            'decision': 'ACCEPT' if accepted else 'REJECT',
            'reason': decision_reason,
            'rollback_triggered': needs_rollback,
            'rollback_reason': rb_reason,
            'signals': signals,
            'method': 'PROPOSED_RELIABILITY_GATED'
        }
