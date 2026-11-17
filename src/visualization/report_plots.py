"""
Research Report and Publication Plots.
Generates comparative bar graphs, ROC/PR comparative curves, and ablation radar charts
for Scopus-indexed research paper publication and M.Tech thesis submission.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any


def plot_method_comparison_bar(
    comparison_df: pd.DataFrame,
    metrics: List[str] = ["auroc", "auprc", "f1_score", "balanced_accuracy"],
    save_path: Optional[str] = None
) -> None:
    """
    Plots a multi-bar chart comparing experimental methods across standard window metrics.
    """
    df = comparison_df.copy()
    df.columns = [c.lower() for c in df.columns]

    method_col = 'method' if 'method' in df.columns else df.columns[0]
    methods = df[method_col].tolist()
    x = np.arange(len(methods))
    
    # Filter available metrics
    avail_metrics = [m for m in metrics if m.lower() in df.columns]
    if not avail_metrics:
        avail_metrics = [c for c in df.columns if c != method_col][:4]
        
    width = 0.8 / max(1, len(avail_metrics))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    for i, metric in enumerate(avail_metrics):
        values = df[metric].astype(float).tolist()
        offset = (i - len(avail_metrics) / 2) * width + width / 2
        label_clean = metric.replace('_', ' ').upper()
        rects = ax.bar(x + offset, values, width, label=label_clean, color=colors[i % len(colors)], alpha=0.85)
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=0)

    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Cross-Dataset Performance Benchmark: CHB-MIT $\\rightarrow$ Siena Scalp EEG', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods], fontsize=9, fontweight='bold')
    ax.set_ylim([0, 1.15])
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)


def plot_event_metrics_comparison(
    event_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plots clinical event-based metrics: Detection Sensitivity, False Alarms/hr, and Latency.
    """
    df = event_df.copy()
    df.columns = [c.lower() for c in df.columns]
    method_col = 'method' if 'method' in df.columns else df.columns[0]
    methods = [str(m).upper() for m in df[method_col]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    sens_col = 'event_sensitivity' if 'event_sensitivity' in df.columns else df.columns[1]
    far_col = 'false_alarms_per_hour' if 'false_alarms_per_hour' in df.columns else df.columns[2]

    # 1. Event Sensitivity
    sens = df[sens_col].astype(float) * 100.0
    x = np.arange(len(methods))
    ax1.bar(x, sens, color='#2ca02c', alpha=0.85, width=0.5)
    ax1.set_ylabel('Event Sensitivity (%)', fontsize=10, fontweight='bold')
    ax1.set_title('Clinical Event Sensitivity', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8, fontweight='bold', rotation=15)
    ax1.set_ylim([0, 110])
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    # 2. False Alarm Rate
    far = df[far_col].astype(float)
    ax2.bar(x, far, color='#d62728', alpha=0.85, width=0.5)
    ax2.set_ylabel('False Alarms / Hour (FA/h)', fontsize=10, fontweight='bold')
    ax2.set_title('False Alarm Rate (Lower is Better)', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8, fontweight='bold', rotation=15)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)
