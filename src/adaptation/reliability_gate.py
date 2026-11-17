"""
Proposed Method: Seizure-Aware Multi-Signal Reliability Gate.
Evaluates candidate model updates against the incumbent model using strictly label-free signals:
  1. EEG signal quality and artifact bounds
  2. Predictive uncertainty bounds (entropy interval [E_min, E_max])
  3. Candidate-vs-Incumbent divergence
  4. Parameter update magnitude ||Delta theta||_2
  5. Class-collapse protection
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.adaptation.tent_adapter import binary_entropy


class ReliabilityGate:
    """
    Unsupervised controller that accepts or rejects candidate model updates
    WITHOUT touching target domain labels.
    """

    def __init__(
        self,
        entropy_min: float = 0.05,       # Reject over-confident redundant predictions
        entropy_max: float = 0.50,       # Reject excessively noisy/ambiguous predictions
        max_divergence: float = 0.40,    # Max allowed |prob_cand - prob_incumbent|
        max_parameter_delta: float = 0.20, # Max L2 parameter step
        class_collapse_window: int = 50, # Window length to check for class collapse
        min_positive_rate: float = 0.005 # Minimum expected activity (avoids all-zero collapse)
    ):
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max
        self.max_divergence = max_divergence
        self.max_parameter_delta = max_parameter_delta
        self.class_collapse_window = class_collapse_window
        self.min_positive_rate = min_positive_rate

        self.recent_predictions = []

    def evaluate_candidate(
        self,
        x_window: torch.Tensor,
        incumbent_model: nn.Module,
        candidate_model: nn.Module,
        candidate_params: list,
        incumbent_params: list,
        quality_metrics: Dict[str, float]
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Evaluate candidate update and decide ACCEPT or REJECT.
        """
        rejection_reasons = []

        # Signal Quality Check
        if quality_metrics.get('is_valid_quality', 1.0) == 0.0:
            rejection_reasons.append("artifact_or_clipped_signal")

        # Compute predictions
        with torch.no_grad():
            x_dev = x_window.unsqueeze(0) if x_window.ndim == 2 else x_window
            prob_incumbent = float(incumbent_model.predict_proba(x_dev).item())
            prob_candidate = float(candidate_model.predict_proba(x_dev).item())

            ent_cand = float(binary_entropy(torch.tensor([prob_candidate])).item())

        # Uncertainty bounds
        if ent_cand > self.entropy_max:
            rejection_reasons.append(f"entropy_exceeds_max_{ent_cand:.3f}>{self.entropy_max}")
        elif ent_cand < self.entropy_min:
            rejection_reasons.append(f"entropy_below_min_redundant_{ent_cand:.3f}<{self.entropy_min}")

        # Candidate vs Incumbent Disagreement
        divergence = abs(prob_candidate - prob_incumbent)
        if divergence > self.max_divergence:
            rejection_reasons.append(f"divergence_exceeds_threshold_{divergence:.3f}>{self.max_divergence}")

        # Parameter Delta L2 Norm
        param_delta_l2 = 0.0
        for p_cand, p_inc in zip(candidate_params, incumbent_params):
            param_delta_l2 += torch.sum((p_cand - p_inc) ** 2).item()
        param_delta_l2 = np.sqrt(param_delta_l2)

        if param_delta_l2 > self.max_parameter_delta:
            rejection_reasons.append(f"parameter_delta_too_large_{param_delta_l2:.4f}>{self.max_parameter_delta}")

        # Class Collapse Protection
        self.recent_predictions.append(prob_candidate)
        if len(self.recent_predictions) > self.class_collapse_window:
            self.recent_predictions.pop(0)

        if len(self.recent_predictions) >= self.class_collapse_window:
            mean_pred = float(np.mean(self.recent_predictions))
            # Collapse check: all zero or all one
            if mean_pred < 1e-4:
                rejection_reasons.append("class_collapse_all_background")
            elif mean_pred > 0.99:
                rejection_reasons.append("class_collapse_all_seizure")

        signals = {
            'prob_incumbent': prob_incumbent,
            'prob_candidate': prob_candidate,
            'divergence': divergence,
            'candidate_entropy': ent_cand,
            'param_delta_l2': float(param_delta_l2),
            'quality_valid': quality_metrics.get('is_valid_quality', 1.0)
        }

        if len(rejection_reasons) == 0:
            return True, "accepted_all_criteria_met", signals
        else:
            return False, ";".join(rejection_reasons), signals
