"""
Source Model Trainer (CHB-MIT).
Executes patient-independent training with mixed-precision, class-weighted optimization,
metric evaluation (AUROC, AUPRC, F1, Balanced Acc), and early stopping.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, balanced_accuracy_score, confusion_matrix
from typing import Dict, Any, Optional, Tuple

from src.training.checkpoint_manager import CheckpointManager


class SourceTrainer:
    """
    Standardized trainer for 1D CNN baseline on CHB-MIT.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        checkpoint_manager: Optional[CheckpointManager] = None,
        use_amp: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        all_targets = []
        all_probs = []

        for batch_x, batch_y, _ in dataloader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True).view(-1, 1)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits = self.model(batch_x)
                    loss = self.criterion(logits, batch_y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            targets = batch_y.detach().cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(targets)

        if self.scheduler:
            self.scheduler.step()

        metrics = self._calculate_metrics(np.array(all_targets), np.array(all_probs))
        metrics['loss'] = total_loss / max(1, n_samples)
        return metrics

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation or test split."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch_x, batch_y, _ in dataloader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True).view(-1, 1)

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(batch_x)
                        loss = self.criterion(logits, batch_y)
                else:
                    logits = self.model(batch_x)
                    loss = self.criterion(logits, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                n_samples += batch_x.size(0)

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                targets = batch_y.cpu().numpy().flatten()
                all_probs.extend(probs)
                all_targets.extend(targets)

        metrics = self._calculate_metrics(np.array(all_targets), np.array(all_probs))
        metrics['loss'] = total_loss / max(1, n_samples)
        return metrics

    def _calculate_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Compute window-level classification metrics."""
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = {}

        # Handle edge cases where only one class exists in batch
        unique_classes = np.unique(y_true)
        if len(unique_classes) > 1:
            try:
                metrics['auroc'] = float(roc_auc_score(y_true, y_prob))
            except Exception:
                metrics['auroc'] = 0.5
            try:
                precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
                metrics['auprc'] = float(auc(recall_arr, precision_arr))
            except Exception:
                metrics['auprc'] = float(np.mean(y_true))
        else:
            metrics['auroc'] = 0.5
            metrics['auprc'] = float(np.mean(y_true))

        metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics['balanced_acc'] = float(balanced_accuracy_score(y_true, y_pred))

        # Confusion matrix: TN, FP, FN, TP
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        return metrics
