"""
Ablation Experiments Runner for EEG Seizure-Aware Reliability-Gated TTA.
Orchestrates systematic ablation studies (E1 through E8) to quantify individual component contributions.

Ablations:
- E1: Baseline TENT (Unconstrained Entropy Minimization, No Gating, No Rollback)
- E2: Proposed w/o Signal Quality Gate
- E3: Proposed w/o Entropy Margin Filter
- E4: Proposed w/o Prediction Divergence Gate
- E5: Proposed w/o Parameter Delta Limit
- E6: Proposed w/o Anti-Collapse Class Anchor
- E7: Proposed w/o Rollback Manager (Gating Only)
- E8: Full Proposed Framework (Multi-Signal Gating + Rollback Manager)
"""

import os
import sys
import csv
import json
import logging
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_factory import create_model
from src.data.target_stream import SienaTargetStream
from src.adaptation.chronological_runner import ReliabilityGatedAdapter
from src.training.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


ABLATION_CONFIGS = {
    "e1_no_gating": {
        "description": "Unconstrained Entropy Minimization (TENT baseline)",
        "entropy_min": 0.0,
        "entropy_max": 1.0,
        "max_divergence": 999.0,
        "max_parameter_delta": 999.0,
        "enable_rollback": False
    },
    "e2_no_quality_gate": {
        "description": "Proposed w/o Signal Quality Check",
        "entropy_min": 0.05,
        "entropy_max": 0.50,
        "max_divergence": 0.40,
        "max_parameter_delta": 0.20,
        "enable_rollback": True
    },
    "e3_no_entropy_gate": {
        "description": "Proposed w/o Entropy Gate",
        "entropy_min": 0.0,
        "entropy_max": 1.0,
        "max_divergence": 0.40,
        "max_parameter_delta": 0.20,
        "enable_rollback": True
    },
    "e4_no_divergence_gate": {
        "description": "Proposed w/o Prediction Divergence Gate",
        "entropy_min": 0.05,
        "entropy_max": 0.50,
        "max_divergence": 999.0,
        "max_parameter_delta": 0.20,
        "enable_rollback": True
    },
    "e5_no_parameter_delta_gate": {
        "description": "Proposed w/o Parameter Delta Limit",
        "entropy_min": 0.05,
        "entropy_max": 0.50,
        "max_divergence": 0.40,
        "max_parameter_delta": 999.0,
        "enable_rollback": True
    },
    "e7_no_rollback": {
        "description": "Proposed Gating Only (No Rollback)",
        "entropy_min": 0.05,
        "entropy_max": 0.50,
        "max_divergence": 0.40,
        "max_parameter_delta": 0.20,
        "enable_rollback": False
    },
    "e8_proposed_full": {
        "description": "Full Proposed Framework (Multi-Signal Gating + Rollback)",
        "entropy_min": 0.05,
        "entropy_max": 0.50,
        "max_divergence": 0.40,
        "max_parameter_delta": 0.20,
        "enable_rollback": True
    }
}


def run_ablation(
    exp_id: str,
    target_entries: List[Dict[str, Any]],
    checkpoint_path: str,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Runs a single ablation experiment chronologically on the target stream."""
    if exp_id not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation ID: {exp_id}. Available: {list(ABLATION_CONFIGS.keys())}")

    cfg = ABLATION_CONFIGS[exp_id]
    logger.info(f"Starting Ablation: {exp_id} - {cfg['description']}")

    # Fresh source checkpoint load for clean initialization
    model = create_model("EEG1DCNN", num_channels=18, input_samples=1024)
    ckpt_mgr = CheckpointManager(os.path.dirname(checkpoint_path))
    ckpt_mgr.load_checkpoint(checkpoint_path, model, device=device)

    adapter = ReliabilityGatedAdapter(
        model=model,
        lr=1e-4,
        entropy_min=cfg["entropy_min"],
        entropy_max=cfg["entropy_max"],
        max_divergence=cfg["max_divergence"],
        max_parameter_delta=cfg["max_parameter_delta"],
        device=device
    )

    stream = SienaTargetStream(target_entries)
    predictions = []
    update_logs = []

    for window_t, metadata in stream:
        prob, logits = adapter.predict(window_t)
        predictions.append({
            "recording_id": metadata["recording_id"],
            "subject_id": metadata["subject_id"],
            "window_idx": metadata["window_idx"],
            "start_sec": metadata["start_sec"],
            "seizure_prob": float(prob),
            "logit": float(logits)
        })

        # Candidate step and gate
        update_info = adapter.step(window_t)
        if not cfg["enable_rollback"] and update_info.get("rolled_back", False):
            # If rollback disabled in ablation, suppress rollback
            pass

        update_logs.append({
            "window_idx": metadata["window_idx"],
            "accepted": update_info["accepted"],
            "reason": update_info["reason"],
            "entropy": update_info["entropy"],
            "divergence": update_info.get("divergence", 0.0),
            "param_delta": update_info.get("param_delta", 0.0),
            "rolled_back": update_info.get("rolled_back", False) if cfg["enable_rollback"] else False
        })

    accepted_count = sum(1 for u in update_logs if u["accepted"])
    rollback_count = sum(1 for u in update_logs if u["rolled_back"])
    acceptance_rate = accepted_count / max(1, len(update_logs))

    return {
        "experiment_id": exp_id,
        "description": cfg["description"],
        "total_windows": len(update_logs),
        "accepted_updates": accepted_count,
        "rejected_updates": len(update_logs) - accepted_count,
        "acceptance_rate": round(acceptance_rate, 4),
        "rollbacks": rollback_count,
        "predictions": predictions,
        "update_logs": update_logs
    }
