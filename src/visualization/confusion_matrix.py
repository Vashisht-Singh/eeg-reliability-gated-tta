"""
Confusion Matrix Visualizer for Seizure Detection.
Generates publication-quality normalized and absolute confusion matrices.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = ["Background", "Seizure"],
    normalize: bool = True,
    title: str = "Normalized Confusion Matrix",
    save_path: Optional[str] = None
) -> None:
    """
    Plots a 2x2 confusion matrix with counts and percentages.
    
    Args:
        cm (np.ndarray): 2x2 confusion matrix [[TN, FP], [FN, TP]].
        class_names (list): Class label strings.
        normalize (bool): Whether to display percentage-normalized values.
        title (str): Plot title.
        save_path (str, optional): Destination file path.
    """
    if normalize:
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    fmt = '.2%' if normalize else 'd'
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm_norm[i, j]:{fmt}}\n(n={cm[i, j]:,})"
            ax.text(
                j, i, val_str,
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9, fontweight='bold'
            )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)
