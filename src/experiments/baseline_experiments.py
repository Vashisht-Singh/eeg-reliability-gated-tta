"""
Baseline Experiments Orchestrator for Cross-Dataset EEG Adaptation.
Runs Frozen model inference and baselines (TENT, EATA, SAR) systematically.
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
from src.adaptation.tent_adapter import TENTAdapter
from src.adaptation.eata_adapter import EATAAdapter
from src.adaptation.sar_adapter import SARAdapter
from src.data.target_stream import SienaTargetStream

logger = logging.getLogger(__name__)


def run_baseline_method(
    method_name: str,
    target_entries: List[Dict[str, Any]],
    checkpoint_path: str,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Runs a single baseline adaptation method across streaming windows."""
    model = create_model("EEG1DCNN", num_channels=18, input_samples=1024)
    ckpt_mgr = CheckpointManager(os.path.dirname(checkpoint_path))
    ckpt_mgr.load_checkpoint(checkpoint_path, model, device=device)

    method_name = method_name.lower()
    if method_name == "tent":
        adapter = TENTAdapter(model, lr=1e-4, device=device)
    elif method_name == "eata":
        adapter = EATAAdapter(model, lr=1e-4, e_margin=0.40, d_margin=0.05, device=device)
    elif method_name == "sar":
        adapter = SARAdapter(model, lr=1e-4, margin_e0=0.40, device=device)
    else:
        raise ValueError(f"Unknown baseline: {method_name}")

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
            "entropy": update_info["entropy"]
        })

    return {
        "method": method_name,
        "total_windows": len(update_logs),
        "accepted_updates": sum(1 for u in update_logs if u["accepted"]),
        "predictions": predictions,
        "update_logs": update_logs
    }
