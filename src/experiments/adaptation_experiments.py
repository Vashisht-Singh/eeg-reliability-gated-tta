"""
Adaptation Experiments Runner.
Orchestrates proposed test-time adaptation across subjects and recording sessions.
"""

import os
import sys
import logging
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_factory import create_model
from src.training.checkpoint_manager import CheckpointManager
from src.adaptation.chronological_runner import ReliabilityGatedAdapter
from src.data.target_stream import SienaTargetStream

logger = logging.getLogger(__name__)


def run_proposed_adaptation(
    target_entries: List[Dict[str, Any]],
    checkpoint_path: str,
    device: str = "cuda",
    lr: float = 1e-4,
    entropy_min: float = 0.05,
    entropy_max: float = 0.50,
    max_divergence: float = 0.40,
    max_parameter_delta: float = 0.20
) -> Dict[str, Any]:
    """Runs the proposed reliability-gated adaptation framework."""
    model = create_model("EEG1DCNN", num_channels=18, input_samples=1024)
    ckpt_mgr = CheckpointManager(os.path.dirname(checkpoint_path))
    ckpt_mgr.load_checkpoint(checkpoint_path, model, device=device)

    adapter = ReliabilityGatedAdapter(
        model=model,
        lr=lr,
        entropy_min=entropy_min,
        entropy_max=entropy_max,
        max_divergence=max_divergence,
        max_parameter_delta=max_parameter_delta,
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

        update_info = adapter.step(window_t)
        update_logs.append({
            "window_idx": metadata["window_idx"],
            "accepted": update_info["accepted"],
            "reason": update_info["reason"],
            "entropy": update_info["entropy"],
            "divergence": update_info.get("divergence", 0.0),
            "param_delta": update_info.get("param_delta", 0.0),
            "rolled_back": update_info.get("rolled_back", False)
        })

    accepted_cnt = sum(1 for u in update_logs if u["accepted"])
    rollback_cnt = sum(1 for u in update_logs if u["rolled_back"])

    return {
        "method": "proposed_reliability_gated",
        "total_windows": len(update_logs),
        "accepted_updates": accepted_cnt,
        "rejected_updates": len(update_logs) - accepted_cnt,
        "acceptance_rate": round(accepted_cnt / max(1, len(update_logs)), 4),
        "rollbacks": rollback_cnt,
        "predictions": predictions,
        "update_logs": update_logs
    }
