"""
Script 05: Train Source Domain Model on CHB-MIT.
Executes patient-independent model training with early stopping, mixed precision,
and checkpointing. Supports --smoke-test mode for rapid pipeline verification.
"""

import os
import sys
import csv
import time
import argparse
import yaml
import psutil
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_factory import create_model
from src.data.source_dataset import CHBMITWindowDataset
from src.training.losses import get_loss_function
from src.training.trainer import SourceTrainer
from src.training.checkpoint_manager import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(description="Train Source Model on CHB-MIT")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on small subset")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("PHASE 3: SOURCE MODEL TRAINING (CHB-MIT)")
    print("=" * 80)

    # 1. Load Configurations
    with open(os.path.join(PROJECT_ROOT, "config", "training.yaml"), 'r') as f:
        train_cfg = yaml.safe_load(f)['training']
    with open(os.path.join(PROJECT_ROOT, "config", "model.yaml"), 'r') as f:
        model_cfg = yaml.safe_load(f)['model']

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    batch_size = args.batch_size or (32 if args.smoke_test else train_cfg.get('batch_size', 64))
    epochs = args.epochs or (2 if args.smoke_test else train_cfg.get('epochs', 25))

    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"Batch Size: {batch_size} | Epochs: {epochs} | Smoke Test: {args.smoke_test}")

    # 2. Load Manifest
    manifests_dir = os.path.join(PROJECT_ROOT, "data", "manifests")
    if args.smoke_test:
        manifest_file = os.path.join(manifests_dir, "source_smoke_test_manifest.csv")
    else:
        manifest_file = os.path.join(manifests_dir, "source_window_manifest.csv")

    with open(manifest_file, 'r', encoding='utf-8') as f:
        all_windows = list(csv.DictReader(f))

    if args.smoke_test:
        train_windows = all_windows[:int(len(all_windows) * 0.7)]
        val_windows = all_windows[int(len(all_windows) * 0.7):]
    else:
        train_windows = [w for w in all_windows if w.get('split') == 'train']
        val_windows = [w for w in all_windows if w.get('split') == 'val']

    print(f"Loaded: {len(train_windows)} training windows | {len(val_windows)} validation windows")

    # 3. Create Datasets and DataLoaders
    # For Windows multiprocessing stability, use num_workers=0
    train_dataset = CHBMITWindowDataset(train_windows, cache_in_memory=args.smoke_test)
    val_dataset = CHBMITWindowDataset(val_windows, cache_in_memory=args.smoke_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=(device == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == 'cuda'))

    # 4. Instantiate Model
    model = create_model(
        model_name=model_cfg.get('name', 'EEG1DCNN'),
        num_channels=model_cfg.get('num_channels', 18),
        input_samples=model_cfg.get('input_samples', 1024)
    )
    param_info = model.get_parameter_count()
    print(f"Model: {model_cfg.get('name', 'EEG1DCNN')}")
    print(f"Parameters: Total={param_info['total_parameters']:,} | Trainable={param_info['trainable_parameters']:,} | Adaptable (TTA)={param_info['adaptable_parameters']:,}")

    # 5. Loss, Optimizer, and Checkpoint Manager
    pos_weight = train_dataset.get_class_weights()
    criterion = get_loss_function(train_cfg.get('loss_function', 'BCEWithLogitsLoss'), pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg['optimizer']['lr']), weight_decay=float(train_cfg['optimizer']['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    ckpt_dir = os.path.join(PROJECT_ROOT, "models", "source")
    ckpt_mgr = CheckpointManager(ckpt_dir, monitor_metric="val_auprc", mode="max")
    trainer = SourceTrainer(model, optimizer, criterion, scheduler, device=device, checkpoint_manager=ckpt_mgr, use_amp=True)

    # 6. Training Loop
    print("\nStarting Training...")
    best_val_auprc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        epoch_time = time.time() - t0

        is_best = val_metrics['auprc'] > best_val_auprc
        if is_best:
            best_val_auprc = val_metrics['auprc']

        ckpt_mgr.save_checkpoint(epoch, model, optimizer, scheduler, val_metrics, is_best=is_best)

        # Resource monitoring
        ram_avail = psutil.virtual_memory().available / (1024**3)
        gpu_mem = torch.cuda.memory_allocated(0) / (1024**2) if device == 'cuda' else 0.0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] | "
            f"Train Loss: {train_metrics['loss']:.4f} AUPRC: {train_metrics['auprc']:.4f} AUROC: {train_metrics['auroc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} AUPRC: {val_metrics['auprc']:.4f} AUROC: {val_metrics['auroc']:.4f} "
            f"Sens: {val_metrics['sensitivity']:.3f} Spec: {val_metrics['specificity']:.3f} | "
            f"RAM Avail: {ram_avail:.1f}GB | GPU: {gpu_mem:.1f}MB {'*' if is_best else ''}"
        )

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE! Best Val AUPRC: {best_val_auprc:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
