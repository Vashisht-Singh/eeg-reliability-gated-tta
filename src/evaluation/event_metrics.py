"""
Clinical Seizure-Event Evaluation Metrics.
Merges consecutive window alarms, detects false alarms per hour (FA/h),
event-level sensitivity, and mean onset detection latency.
"""

import numpy as np
from typing import Dict, List, Any, Tuple


def merge_window_alarms(
    window_preds: List[Dict[str, Any]],
    prob_threshold: float = 0.5,
    min_event_duration_sec: float = 4.0,
    merge_tolerance_sec: float = 10.0
) -> List[Dict[str, float]]:
    """
    Groups window alarms into continuous clinical alarm intervals.
    """
    raw_alarms = []
    for w in window_preds:
        if float(w['probability']) >= prob_threshold:
            raw_alarms.append((float(w['start_sec']), float(w['end_sec'])))

    if not raw_alarms:
        return []

    # Sort alarms by start time
    raw_alarms.sort(key=lambda a: a[0])

    # Merge overlapping or close alarms
    merged = []
    curr_start, curr_end = raw_alarms[0]

    for next_start, next_end in raw_alarms[1:]:
        if next_start - curr_end <= merge_tolerance_sec:
            curr_end = max(curr_end, next_end)
        else:
            if (curr_end - curr_start) >= min_event_duration_sec:
                merged.append({'onset_sec': curr_start, 'offset_sec': curr_end, 'duration_sec': curr_end - curr_start})
            curr_start, curr_end = next_start, next_end

    if (curr_end - curr_start) >= min_event_duration_sec:
        merged.append({'onset_sec': curr_start, 'offset_sec': curr_end, 'duration_sec': curr_end - curr_start})

    return merged


def compute_event_metrics(
    alarms: List[Dict[str, float]],
    true_seizures: List[Dict[str, float]],
    total_recording_hours: float
) -> Dict[str, float]:
    """
    Match clinical alarm intervals against ground truth seizures.
    """
    tp_events = 0
    matched_seizures = set()
    latencies = []

    for alarm in alarms:
        a_start, a_end = alarm['onset_sec'], alarm['offset_sec']
        matched = False

        for s_idx, sz in enumerate(true_seizures):
            s_start, s_end = sz['onset_sec'], sz['offset_sec']

            # Check overlap
            overlap = max(0.0, min(a_end, s_end) - max(a_start, s_start))
            if overlap > 0.0:
                matched = True
                matched_seizures.add(s_idx)
                # Latency: delay from true seizure start to alarm declaration
                latency = max(0.0, a_start - s_start)
                latencies.append(latency)
                break

        if matched:
            tp_events += 1

    total_true_events = len(true_seizures)
    fn_events = total_true_events - len(matched_seizures)
    fp_events = len(alarms) - tp_events

    event_sensitivity = (len(matched_seizures) / total_true_events) if total_true_events > 0 else 1.0
    fa_per_hour = (fp_events / max(1e-4, total_recording_hours))
    event_f1 = (2 * tp_events / (2 * tp_events + fp_events + fn_events)) if (2 * tp_events + fp_events + fn_events) > 0 else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        'total_true_events': total_true_events,
        'detected_events': len(matched_seizures),
        'missed_events': fn_events,
        'false_alarm_events': fp_events,
        'event_sensitivity': float(event_sensitivity),
        'false_alarms_per_hour': float(fa_per_hour),
        'event_f1': float(event_f1),
        'mean_latency_sec': mean_latency
    }
