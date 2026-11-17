"""
Online Adaptation Dynamics Visualization.
Plots chronological streaming trajectories of model entropy, candidate updates,
acceptance/rejection decisions, and automated rollback interventions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional


def plot_adaptation_dynamics(
    update_logs: List[Dict[str, Any]],
    title: str = "Chronological Test-Time Adaptation Dynamics",
    save_path: Optional[str] = None
) -> None:
    """
    Plots entropy trajectory, update decisions, and rollback events across streaming windows.
    """
    window_indices = list(range(len(update_logs)))
    entropies = []
    accepted = []
    rolled_back = []
    divergences = []

    for idx, u in enumerate(update_logs):
        # Entropy
        ent = u.get('candidate_entropy', u.get('entropy', 0.0))
        entropies.append(float(ent))
        
        # Decision
        dec = str(u.get('decision', u.get('accepted', ''))).upper()
        is_acc = (dec in ['ACCEPT', 'TRUE', '1'])
        accepted.append(is_acc)

        # Rollback
        rb = str(u.get('rollback_triggered', u.get('rolled_back', '0')))
        is_rb = (rb in ['1', 'TRUE', 'True'])
        rolled_back.append(is_rb)

        # Divergence
        div = u.get('divergence', 0.0)
        divergences.append(float(div))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, dpi=300)

    # 1. Entropy Trajectory & Update Decisions
    ax1.plot(window_indices, entropies, color='gray', alpha=0.6, linewidth=1.0, label='Window Entropy $H(p)$')
    
    # Mark Accepted Updates
    acc_idx = [idx for idx, acc in zip(window_indices, accepted) if acc]
    acc_ent = [entropies[i] for i in acc_idx]
    if acc_idx:
        ax1.scatter(acc_idx, acc_ent, color='#2ca02c', s=30, label='Accepted Update', zorder=5)

    # Mark Rejected Updates
    rej_idx = [idx for idx, acc in zip(window_indices, accepted) if not acc]
    rej_ent = [entropies[i] for i in rej_idx]
    if rej_idx:
        ax1.scatter(rej_idx, rej_ent, color='#d62728', marker='x', s=16, alpha=0.5, label='Rejected Update', zorder=4)

    # Mark Rollbacks
    rb_idx = [idx for idx, rb in zip(window_indices, rolled_back) if rb]
    rb_ent = [entropies[i] for i in rb_idx]
    if rb_idx:
        ax1.scatter(rb_idx, rb_ent, color='purple', marker='v', s=80, label='Automated Rollback', zorder=6)

    ax1.set_ylabel('Shannon Entropy (nats)', fontsize=10)
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    # 2. Prediction Divergence Gate
    ax2.plot(window_indices, divergences, color='#1f77b4', linewidth=1.2, label='Candidate Prediction Divergence')
    ax2.axhline(0.40, color='darkorange', linestyle='--', linewidth=1.2, label='Divergence Gate Threshold (0.40)')
    ax2.set_xlabel('Streaming Window Index', fontsize=10)
    ax2.set_ylabel('Divergence $\\Delta_{\\text{prob}}$', fontsize=10)
    ax2.set_ylim([-0.05, 1.05])
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)
