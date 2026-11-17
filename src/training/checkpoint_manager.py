"""
Model Checkpoint Manager.
Handles atomic saving, loading, metric tracking, and resumption of model states.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class CheckpointManager:
    """
    Manages best and latest checkpoints for source model training.
    """

    def __init__(self, checkpoint_dir: str, monitor_metric: str = "val_auprc", mode: str = "max"):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.best_epoch = -1
        self.history = []

    def is_better(self, current_score: float) -> bool:
        if self.mode == 'max':
            return current_score > self.best_score
        else:
            return current_score < self.best_score

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> str:
        """Save latest and optionally best model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'monitor_metric': self.monitor_metric,
            'best_score': self.best_score
        }

        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(state, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(state, best_path)

        # Update history
        record = {'epoch': epoch, **metrics}
        self.history.append(record)
        hist_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(hist_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

        return latest_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Load checkpoint state into model and optimizer."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint
