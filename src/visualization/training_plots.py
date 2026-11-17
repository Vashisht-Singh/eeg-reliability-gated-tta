"""
Training Curves and Performance Visualization.
Plots Loss, AUROC, AUPRC, and Sensitivity-Specificity trajectories over training epochs.
"""

import os
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plots training and validation loss, AUROC, and AUPRC curves over epochs.
    
    Args:
        history (dict): Dictionary with keys 'train_loss', 'val_loss', 'val_auroc', 'val_auprc'.
        save_path (str, optional): Destination image path (e.g. .png, .pdf).
        show (bool): Whether to display the plot interactively.
    """
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    
    # 1. Loss Curve
    axes[0].plot(epochs, history.get('train_loss', []), 'b-o', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r--s', label='Val Loss', linewidth=2)
    axes[0].set_title('Cross-Entropy Loss Trajectory', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    # 2. AUROC Curve
    if 'val_auroc' in history:
        axes[1].plot(epochs, history['val_auroc'], 'g-^', label='Val AUROC', linewidth=2)
        axes[1].set_title('Area Under ROC Curve', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=10)
        axes[1].set_ylabel('AUROC', fontsize=10)
        axes[1].set_ylim([0.5, 1.0])
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend()

    # 3. AUPRC Curve
    if 'val_auprc' in history:
        axes[2].plot(epochs, history['val_auprc'], 'm-d', label='Val AUPRC', linewidth=2)
        axes[2].set_title('Precision-Recall Area (AUPRC)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Epoch', fontsize=10)
        axes[2].set_ylabel('AUPRC', fontsize=10)
        axes[2].set_ylim([0.0, 1.0])
        axes[2].grid(True, linestyle=':', alpha=0.6)
        axes[2].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)
