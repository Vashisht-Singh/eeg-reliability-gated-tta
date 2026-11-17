"""
Script 07: Phase 4 — Frozen Source Model Evaluation on Siena Target Stream.
Runs the unadapted source model chronologically over Siena without parameter updates.
Enforces strict label isolation: target annotations are completely inaccessible.
"""

import os
import sys
import csv
import argparse
import time
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_factory import create_model
from src.data.target_stream import SienaTargetStream
from src.training.checkpoint_manager import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(description="Run Frozen Source Model on Siena Stream")
    parser.add_argument("--smoke-test", action="store_true", help="Run on smoke-test target manifest")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of streaming windows")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("PHASE 4: FROZEN SOURCE MODEL ON CHRONOLOGICAL TARGET STREAM (SIENA)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "source", "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "models", "source", "latest_checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Source checkpoint not found in {os.path.dirname(ckpt_path)}")

    # 1. Load Model
    model = create_model("EEG1DCNN", num_channels=18, input_samples=1024).to(device)
    ckpt_mgr = CheckpointManager(os.path.dirname(ckpt_path))
    ckpt_mgr.load_checkpoint(ckpt_path, model, device=device)
    model.eval()
    print(f"Loaded source checkpoint: {ckpt_path}")

    # 2. Load Streaming Manifest (Unlabeled)
    manifests_dir = os.path.join(PROJECT_ROOT, "data", "manifests")
    if args.smoke_test:
        manifest_file = os.path.join(manifests_dir, "target_smoke_test_manifest.csv")
    else:
        manifest_file = os.path.join(manifests_dir, "target_stream_manifest.csv")

    with open(manifest_file, 'r', encoding='utf-8') as f:
        entries = list(csv.DictReader(f))

    if args.limit:
        entries = entries[:args.limit]
    print(f"Target Stream: {len(entries)} windows (Strictly unlabeled)")

    # 3. Stream through windows and record predictions
    stream = SienaTargetStream(entries)
    predictions = []

    print("\nStreaming through target recordings chronologically...")
    t0 = time.time()

    with torch.no_grad():
        for idx, win in enumerate(stream.stream_windows()):
            x_tensor = torch.from_numpy(win['x']).unsqueeze(0).to(device)
            prob = float(model.predict_proba(x_tensor).item())
            pred_class = 1 if prob >= 0.5 else 0

            predictions.append({
                'window_id': win['window_id'],
                'dataset': win['dataset'],
                'subject_id': win['subject_id'],
                'filename': win['filename'],
                'start_sec': win['start_sec'],
                'end_sec': win['end_sec'],
                'probability': round(prob, 5),
                'predicted_class': pred_class,
                'is_valid_quality': int(win['is_valid_quality'])
            })

            if (idx + 1) % 500 == 0 or (idx + 1) == len(entries):
                elapsed = time.time() - t0
                fps = (idx + 1) / max(1e-3, elapsed)
                print(f"  Processed {idx + 1}/{len(entries)} windows ({fps:.1f} windows/sec)")

    # 4. Save Predictions
    pred_dir = os.path.join(PROJECT_ROOT, "experiments", "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    out_csv = os.path.join(pred_dir, "frozen_siena_predictions.csv")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)

    print("\n" + "=" * 80)
    print(f"FROZEN EVALUATION COMPLETE! Saved {len(predictions)} predictions to:")
    print(f"  {out_csv}")
    print("=" * 80)


if __name__ == '__main__':
    main()
