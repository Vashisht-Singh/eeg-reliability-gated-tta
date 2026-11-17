"""
Script 08: Phase 5 — Run Test-Time Adaptation Baselines (TENT, EATA, SAR).
Executes unsupervised online adaptation chronologically over Siena.
Enforces predict-first protocol and saves prediction timelines and update logs.
"""

import os
import sys
import csv
import time
import argparse
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_factory import create_model
from src.data.target_stream import SienaTargetStream
from src.adaptation.tent_adapter import TentAdapter
from src.adaptation.eata_adapter import EataAdapter
from src.adaptation.sar_adapter import SarAdapter
from src.training.checkpoint_manager import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(description="Run TTA Baselines on Siena Target Stream")
    parser.add_argument("--methods", nargs="+", default=["tent", "eata", "sar"], help="Baselines to run")
    parser.add_argument("--smoke-test", action="store_true", help="Run on smoke-test target manifest")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of streaming windows")
    return parser.parse_args()


def run_single_baseline(method_name: str, entries: list, ckpt_path: str, device: str):
    print(f"\n--- Running Baseline: {method_name.upper()} ---")

    # 1. Instantiate fresh model from source checkpoint
    model = create_model("EEG1DCNN", num_channels=18, input_samples=1024)
    ckpt_mgr = CheckpointManager(os.path.dirname(ckpt_path))
    ckpt_mgr.load_checkpoint(ckpt_path, model, device=device)

    # 2. Select Adapter
    if method_name.lower() == 'tent':
        adapter = TentAdapter(model, lr=1e-4, device=device)
    elif method_name.lower() == 'eata':
        adapter = EataAdapter(model, lr=1e-4, entropy_margin_e0=0.40, fisher_weight=200.0, device=device)
    elif method_name.lower() == 'sar':
        adapter = SarAdapter(model, lr=1e-4, margin_e0=0.40, rho=0.05, device=device)
    else:
        raise ValueError(f"Unknown baseline: {method_name}")

    # 3. Stream through windows
    stream = SienaTargetStream(entries)
    predictions = []
    update_logs = []
    t0 = time.time()

    for idx, win in enumerate(stream.stream_windows()):
        x_tensor = torch.from_numpy(win['x']).unsqueeze(0)
        t_sec = float(win['start_sec'])

        # Predict first, adapt second
        prob, info = adapter.step(x_tensor, timestamp=t_sec)
        pred_class = 1 if prob >= 0.5 else 0

        predictions.append({
            'window_id': win['window_id'],
            'dataset': win['dataset'],
            'subject_id': win['subject_id'],
            'filename': win['filename'],
            'start_sec': win['start_sec'],
            'end_sec': win['end_sec'],
            'probability': round(prob, 5),
            'predicted_class': pred_class
        })

        update_logs.append({
            'window_id': win['window_id'],
            'timestamp': t_sec,
            'updated': int(info.get('updated', False)),
            'entropy': round(info.get('entropy', 0.0), 4),
            'loss': round(info.get('adaptation_loss', 0.0), 4),
            'reason': info.get('reason', 'normal_update')
        })

        if (idx + 1) % 500 == 0 or (idx + 1) == len(entries):
            elapsed = time.time() - t0
            fps = (idx + 1) / max(1e-3, elapsed)
            updates = adapter.total_updates
            print(f"  [{method_name.upper()}] Processed {idx + 1}/{len(entries)} windows | Updates: {updates} ({fps:.1f} win/s)")

    # 4. Save results
    pred_dir = os.path.join(PROJECT_ROOT, "experiments", "predictions")
    log_dir = os.path.join(PROJECT_ROOT, "experiments", "update_logs")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    pred_csv = os.path.join(pred_dir, f"{method_name.lower()}_predictions.csv")
    log_csv = os.path.join(log_dir, f"{method_name.lower()}_update_log.csv")

    with open(pred_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)

    with open(log_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(update_logs[0].keys()))
        writer.writeheader()
        writer.writerows(update_logs)

    print(f"  Saved predictions: {pred_csv}")
    print(f"  Saved update log:  {log_csv}")


def main():
    args = parse_args()
    print("=" * 80)
    print("PHASE 5: TEST-TIME ADAPTATION BASELINES (TENT, EATA, SAR)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "source", "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "models", "source", "latest_checkpoint.pt")

    manifests_dir = os.path.join(PROJECT_ROOT, "data", "manifests")
    manifest_file = os.path.join(manifests_dir, "target_smoke_test_manifest.csv" if args.smoke_test else "target_stream_manifest.csv")

    with open(manifest_file, 'r', encoding='utf-8') as f:
        entries = list(csv.DictReader(f))
    if args.limit:
        entries = entries[:args.limit]

    print(f"Target Windows: {len(entries)} | Baselines: {args.methods}")

    for method in args.methods:
        run_single_baseline(method, entries, ckpt_path, device)

    print("\n" + "=" * 80)
    print("PHASE 5 BASELINE ADAPTATION RUNS COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
