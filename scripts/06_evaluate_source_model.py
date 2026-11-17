"""
Script 06: Evaluate Source Model on Source Test Patients.
Evaluates the trained CHB-MIT model on independent source test patients (chb12, chb13, chb17, chb24).
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_factory import create_model
from src.data.source_dataset import CHBMITWindowDataset
from src.training.trainer import SourceTrainer
from src.training.losses import get_loss_function
from src.training.checkpoint_manager import CheckpointManager


def main():
    print("=" * 80)
    print("EVALUATING SOURCE MODEL ON PATIENT-INDEPENDENT TEST SPLIT")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "source", "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "models", "source", "latest_checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Source checkpoint not found in {os.path.dirname(ckpt_path)}")

    # 1. Load model
    model = create_model("EEG1DCNN", num_channels=18, input_samples=1024).to(device)
    ckpt_mgr = CheckpointManager(os.path.dirname(ckpt_path))
    ckpt = ckpt_mgr.load_checkpoint(ckpt_path, model, device=device)
    print(f"Loaded source model from: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    # 2. Load test windows
    manifest_path = os.path.join(PROJECT_ROOT, "data", "manifests", "source_window_manifest.csv")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        all_windows = list(csv.DictReader(f))

    test_windows = [w for w in all_windows if w.get('split') == 'source_test']
    print(f"Loaded {len(test_windows)} test windows from test patients (chb12, chb13, chb17, chb24)")

    # 3. Evaluate
    test_dataset = CHBMITWindowDataset(test_windows[:2000])  # Sample evaluation for efficiency
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    criterion = get_loss_function("BCEWithLogitsLoss")
    trainer = SourceTrainer(model, optimizer=None, criterion=criterion, device=device)
    metrics = trainer.validate(test_loader)

    print("\nSource Test Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    # Save metrics
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "source_test_metrics.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {out_path}")


if __name__ == '__main__':
    main()
