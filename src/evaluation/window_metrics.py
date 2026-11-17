"""
Window-Level Evaluation Metrics.
Calculates Accuracy, Sensitivity, Specificity, Precision, F1, AUROC, AUPRC, MCC, and Kappa.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    precision_recall_curve, auc, f1_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix
)
from typing import Dict, Any


def compute_window_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute comprehensive window-level classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred))
    }

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # Probability Curves
    if len(np.unique(y_true)) > 1:
        try:
            metrics['auroc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics['auroc'] = 0.5
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            metrics['auprc'] = float(auc(rec, prec))
        except Exception:
            metrics['auprc'] = float(np.mean(y_true))
    else:
        metrics['auroc'] = 0.5
        metrics['auprc'] = float(np.mean(y_true))

    return metrics
