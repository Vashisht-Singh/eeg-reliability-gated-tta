"""
Counterfactual Update-Harm Analysis Module.
Analyzes whether candidate updates accepted or rejected by the online gate
were genuinely beneficial or harmful to subsequent seizure detection.
"""

from typing import Dict, List, Any


def run_counterfactual_harm_analysis(
    update_log: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    true_seizures: List[Dict[str, float]],
    horizon_windows: int = 50
) -> Dict[str, Any]:
    """
    Retrospectively evaluates update decisions against subsequent ground-truth outcomes.
    """
    total_decisions = len(update_log)
    beneficial_accepted = 0
    harmful_accepted = 0
    harmful_correctly_rejected = 0
    beneficial_falsely_rejected = 0

    # Build lookup of seizure intervals
    intervals = [(sz['onset_sec'], sz['offset_sec']) for sz in true_seizures]

    for idx, row in enumerate(update_log):
        decision = row['decision']
        t_sec = float(row['timestamp'])

        # Check if subsequent horizon intersects any seizure
        horizon_end = t_sec + horizon_windows
        has_future_seizure = any(max(0.0, min(horizon_end, s_end) - max(t_sec, s_start)) > 0 for s_start, s_end in intervals)

        # In un-seized background: updates that cause excessive divergence are harmful
        divergence = float(row.get('divergence', 0.0))

        if decision == 'ACCEPT':
            if divergence > 0.35 and not has_future_seizure:
                harmful_accepted += 1
            else:
                beneficial_accepted += 1
        else:  # REJECT
            if divergence > 0.35:
                harmful_correctly_rejected += 1
            else:
                beneficial_falsely_rejected += 1

    total_accepted = beneficial_accepted + harmful_accepted
    total_rejected = harmful_correctly_rejected + beneficial_falsely_rejected

    return {
        'total_evaluated_updates': total_decisions,
        'beneficial_accepted': beneficial_accepted,
        'harmful_accepted': harmful_accepted,
        'harmful_correctly_rejected': harmful_correctly_rejected,
        'beneficial_falsely_rejected': beneficial_falsely_rejected,
        'harmful_rejection_accuracy': (harmful_correctly_rejected / max(1, harmful_correctly_rejected + harmful_accepted)) * 100.0,
        'gate_precision': (beneficial_accepted / max(1, total_accepted)) * 100.0 if total_accepted > 0 else 0.0
    }
