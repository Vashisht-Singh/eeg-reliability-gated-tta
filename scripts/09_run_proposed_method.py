"""
Script 09: Phase 6 — Run Proposed Seizure-Aware Reliability-Gated Adaptation.
Executes the proposed framework: Predict First -> Candidate Update -> Multi-Signal Gating -> Rollback.
Includes ablation study flags (E4 through E8).
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
from src.adaptation.chronological_runner import ReliabilityGatedAdapter
from src.training.checkpoint_manager import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(description="Run Proposed Reliability-Gated Adaptation")
    parser.add_argument("--experiment-id", type=str, default="e8_proposed_full", help="Experiment identifier (e.g. e4, e5, e7, e8)")
    parser.add_argument("--smoke-test", action="store_true", help="Run on smoke-test target manifest")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of streaming windows")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print(f"PHASE 6: PROPOSED SEIZURE-AWARE RELIABILITY-GATED ADAPTATION [{args.experiment_id.upper()}]")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "source", "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "models", "source", "latest_checkpoint.pt")

    # 1. Instantiate Model
    model = create_model("EEG1DCNN", num_channels=18, input_samples=1024)
    ckpt_mgr = CheckpointManager(os.path.dirname(ckpt_path))
    ckpt_mgr.load_checkpoint(ckpt_path, model, device=device)

    # 2. Instantiate Proposed Adapter
    adapter = ReliabilityGatedAdapter(
        model=model,
        lr=1e-4,
        entropy_min=0.05,
        entropy_max=0.50,
        max_divergence=0.40,
        max_parameter_delta=0.20,
        device=device
    )

    # 3. Load Target Streaming Manifest
    manifests_dir = os.path.join(PROJECT_ROOT, "data", "manifests")
    manifest_file = os.path.join(manifests_dir, "target_smoke_test_manifest.csv" if args.smoke_test else "target_stream_manifest.csv")

    with open(manifest_file, 'r', encoding='utf-8') as f:
        entries = list(csv.DictReader(f))
    if args.limit:
        entries = entries[:args.limit]

    print(f"Target Stream: {len(entries)} windows | Mode: Chronological Online Gating")

    # 4. Stream and Adapt
    stream = SienaTargetStream(entries)
    predictions = []
    update_logs = []
    t0 = time.time()

    for idx, win in enumerate(stream.stream_windows()):
        x_tensor = torch.from_numpy(win['x']).unsqueeze(0)
        t_sec = float(win['start_sec'])

        # Predict first (incumbent prediction)
        incumbent_prob = adapter.predict(x_tensor)
        pred_class = 1 if incumbent_prob >= 0.5 else 0

        # Adapt second (candidate generation + gating + rollback)
        info = adapter.adapt(x_tensor, quality_metrics=win['quality_metrics'])
        signals = info.get('signals', {})

        predictions.append({
            'window_id': win['window_id'],
            'dataset': win['dataset'],
            'subject_id': win['subject_id'],
            'filename': win['filename'],
            'start_sec': win['start_sec'],
            'end_sec': win['end_sec'],
            'probability': round(incumbent_prob, 5),
            'predicted_class': pred_class
        })

        update_logs.append({
            'window_id': win['window_id'],
            'timestamp': t_sec,
            'decision': info['decision'],
            'reason': info['reason'],
            'divergence': round(signals.get('divergence', 0.0), 4),
            'candidate_entropy': round(signals.get('candidate_entropy', 0.0), 4),
            'param_delta_l2': round(signals.get('param_delta_l2', 0.0), 4),
            'rollback_triggered': int(info['rollback_triggered'])
        })

        if (idx + 1) % 500 == 0 or (idx + 1) == len(entries):
            elapsed = time.time() - t0
            fps = (idx + 1) / max(1e-3, elapsed)
            acc = adapter.stats['accepted_updates']
            rej = adapter.stats['rejected_updates']
            rb = adapter.stats['rollbacks']
            acc_rate = (acc / max(1, acc + rej)) * 100.0
            print(f"  Processed {idx + 1}/{len(entries)} | Accepted: {acc} | Rejected: {rej} (Acc Rate: {acc_rate:.1f}%) | Rollbacks: {rb} ({fps:.1f} win/s)")

    # 5. Save Predictions and Gate Logs
    pred_dir = os.path.join(PROJECT_ROOT, "experiments", "predictions")
    log_dir = os.path.join(PROJECT_ROOT, "experiments", "update_logs")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    pred_csv = os.path.join(pred_dir, f"{args.experiment_id.lower()}_predictions.csv")
    log_csv = os.path.join(log_dir, f"{args.experiment_id.lower()}_update_log.csv")

    with open(pred_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)

    with open(log_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(update_logs[0].keys()))
        writer.writeheader()
        writer.writerows(update_logs)

    print("\n" + "=" * 80)
    print(f"PROPOSED METHOD EXECUTION COMPLETE!")
    print(f"  Accepted Updates: {adapter.stats['accepted_updates']} / {adapter.stats['total_candidates']}")
    print(f"  Rejected Updates: {adapter.stats['rejected_updates']} / {adapter.stats['total_candidates']}")
    print(f"  Rollback Events:  {adapter.stats['rollbacks']}")
    print(f"  Saved Predictions: {pred_csv}")
    print(f"  Saved Gate Logs:   {log_csv}")
    print("=" * 80)


if __name__ == '__main__':
    main()
