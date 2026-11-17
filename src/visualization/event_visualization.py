"""
Event Visualization Module for Temporal EEG Seizure Detections.
Plots continuous predicted seizure probability curves over recording timelines,
annotating ground-truth onset/offset intervals, detection thresholds, and latencies.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional


def plot_temporal_seizure_timeline(
    timestamps_sec: np.ndarray,
    predicted_probs: np.ndarray,
    ground_truth_events: List[Dict[str, float]],
    threshold: float = 0.5,
    title: str = "Temporal Seizure Detection Probability Trace",
    save_path: Optional[str] = None
) -> None:
    """
    Plots a continuous probability trace against annotated seizure intervals.
    
    Args:
        timestamps_sec (np.ndarray): 1D array of window start times in seconds.
        predicted_probs (np.ndarray): 1D array of predicted seizure probabilities.
        ground_truth_events (list of dict): Each dict has 'onset_sec' and 'offset_sec'.
        threshold (float): Decision threshold (default 0.5).
        title (str): Plot title.
        save_path (str, optional): Destination file path.
    """
    fig, ax = plt.subplots(figsize=(14, 4), dpi=300)
    
    # 1. Plot probability curve
    ax.plot(timestamps_sec, predicted_probs, color='#1f77b4', linewidth=1.5, label='Predicted $P(\\text{Seizure})$')
    ax.axhline(threshold, color='crimson', linestyle='--', linewidth=1.2, label=f'Threshold ({threshold})')
    
    # 2. Highlight Ground-Truth Seizure Intervals
    for idx, event in enumerate(ground_truth_events):
        onset = event.get('onset_sec', event.get('onset', 0.0))
        offset = event.get('offset_sec', event.get('offset', 0.0))
        lbl = 'Ground-Truth Seizure' if idx == 0 else ""
        ax.axvspan(onset, offset, color='#ff7f0e', alpha=0.35, label=lbl)

    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Time in Recording (seconds)', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)
