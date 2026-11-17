"""
Script 10: Phase 7 — Final Evaluation, Counterfactual Analysis, and Publishing Report Generator.
RELEASES SIENA GROUND TRUTH ANNOTATIONS STRICTLY FOR OFFLINE BENCHMARKING.
Computes window-level and clinical event-level metrics for Frozen, TENT, EATA, SAR, and Proposed Method.
"""

import os
import sys
import csv
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.annotation_parser import load_offline_target_annotations
from src.evaluation.window_metrics import compute_window_metrics
from src.evaluation.event_metrics import merge_window_alarms, compute_event_metrics
from src.experiments.counterfactual_experiments import run_counterfactual_harm_analysis


def evaluate_method_predictions(
    predictions_csv: str,
    ground_truth_events: dict,
    total_duration_hours: float
) -> tuple:
    """Evaluate one prediction file against ground truth."""
    with open(predictions_csv, 'r', encoding='utf-8') as f:
        preds = list(csv.DictReader(f))

    if not preds:
        return {}, {}

    # 1. Synthesize window-level ground truth from event intervals
    y_true = []
    y_prob = []

    for w in preds:
        fname = w['filename']
        w_start = float(w['start_sec'])
        w_end = float(w['end_sec'])
        prob = float(w['probability'])

        # Check intersection with true events
        events_for_file = ground_truth_events.get(fname, [])
        is_sz = 0
        for ev in events_for_file:
            s_start = float(ev['onset_sec'])
            s_end = float(ev['offset_sec'])
            overlap = max(0.0, min(w_end, s_end) - max(w_start, s_start))
            if (overlap / max(1e-3, w_end - w_start)) >= 0.5:
                is_sz = 1
                break

        y_true.append(is_sz)
        y_prob.append(prob)

    # Compute window metrics
    win_metrics = compute_window_metrics(np.array(y_true), np.array(y_prob))

    # 2. Compute event-level metrics
    alarms = merge_window_alarms(preds, prob_threshold=0.5, min_event_duration_sec=4.0, merge_tolerance_sec=10.0)

    # Collect all true events for files in this prediction set
    files_present = set(w['filename'] for w in preds)
    relevant_true_events = []
    for fn in files_present:
        relevant_true_events.extend(ground_truth_events.get(fn, []))

    event_metrics = compute_event_metrics(alarms, relevant_true_events, total_duration_hours)

    return win_metrics, event_metrics


def main():
    print("=" * 80)
    print("PHASE 7: FINAL COMPREHENSIVE BENCHMARKING & PUBLICATION EVALUATION")
    print("=" * 80)

    eval_annot_dir = os.path.join(PROJECT_ROOT, "data", "evaluation_annotations")
    annot_json = os.path.join(eval_annot_dir, "siena_ground_truth_events.json")
    if not os.path.exists(annot_json):
        raise FileNotFoundError(f"Isolated annotations store missing at: {annot_json}")

    # RELEASE SIENA GROUND TRUTH ANNOTATIONS (EXCLUSIVELY IN THIS MODULE)
    print("1. Releasing Siena Target Domain Ground Truth Annotations to Evaluation Engine...")
    ground_truth = load_offline_target_annotations(annot_json)
    print(f"   Loaded annotations for {len(ground_truth)} target recordings.")

    pred_dir = os.path.join(PROJECT_ROOT, "experiments", "predictions")
    reports_dir = os.path.join(PROJECT_ROOT, "experiments", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    methods = [
        ("E0_Frozen", os.path.join(pred_dir, "frozen_siena_predictions.csv")),
        ("E1_TENT", os.path.join(pred_dir, "tent_predictions.csv")),
        ("E2_EATA", os.path.join(pred_dir, "eata_predictions.csv")),
        ("E3_SAR", os.path.join(pred_dir, "sar_predictions.csv")),
        ("E8_Proposed", os.path.join(pred_dir, "e8_proposed_full_predictions.csv"))
    ]

    window_results = []
    event_results = []

    print("\n2. Evaluating All Comparative Methods on Target Stream:")

    for method_label, csv_path in methods:
        if not os.path.exists(csv_path):
            print(f"   [SKIP] Prediction file not found: {csv_path}")
            continue

        # Duration based on number of windows (each step is 1s stride)
        with open(csv_path, 'r', encoding='utf-8') as f:
            n_rows = sum(1 for _ in f) - 1
        dur_hours = (n_rows * 1.0) / 3600.0

        w_met, e_met = evaluate_method_predictions(csv_path, ground_truth, dur_hours)

        w_met_clean = {'Method': method_label, **{k: round(v, 4) for k, v in w_met.items()}}
        e_met_clean = {'Method': method_label, **{k: round(v, 4) if isinstance(v, float) else v for k, v in e_met.items()}}

        window_results.append(w_met_clean)
        event_results.append(e_met_clean)

        print(f"   [{method_label}] Sensitivity: {w_met.get('sensitivity', 0):.3f} | Specificity: {w_met.get('specificity', 0):.3f} | "
              f"AUPRC: {w_met.get('auprc', 0):.4f} | Event Sens: {e_met.get('event_sensitivity', 0):.3f} | FA/h: {e_met.get('false_alarms_per_hour', 0):.2f}")

    # 3. Save comparative tables
    win_csv = os.path.join(reports_dir, "final_window_metrics_comparison.csv")
    with open(win_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(window_results[0].keys()))
        writer.writeheader()
        writer.writerows(window_results)

    evt_csv = os.path.join(reports_dir, "final_event_metrics_comparison.csv")
    with open(evt_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(event_results[0].keys()))
        writer.writeheader()
        writer.writerows(event_results)

    # 4. Run Counterfactual Update-Harm Analysis
    print("\n3. Executing Counterfactual Update-Harm Analysis on Proposed Method:")
    update_log_csv = os.path.join(PROJECT_ROOT, "experiments", "update_logs", "e8_proposed_full_update_log.csv")
    if os.path.exists(update_log_csv):
        with open(update_log_csv, 'r', encoding='utf-8') as f:
            up_logs = list(csv.DictReader(f))

        # Flatten all seizures for counterfactual check
        flat_seizures = []
        for ev_list in ground_truth.values():
            flat_seizures.extend(ev_list)

        cf_results = run_counterfactual_harm_analysis(up_logs, [], flat_seizures, horizon_windows=50)

        cf_json = os.path.join(reports_dir, "counterfactual_analysis_results.json")
        with open(cf_json, 'w', encoding='utf-8') as f:
            json.dump(cf_results, f, indent=2)

        print(f"   Evaluated Updates: {cf_results['total_evaluated_updates']}")
        print(f"   Harmful Updates Successfully Blocked: {cf_results['harmful_correctly_rejected']}")
        print(f"   Harmful Rejection Accuracy: {cf_results['harmful_rejection_accuracy']:.1f}%")
        print(f"   Gate Precision: {cf_results['gate_precision']:.1f}%")
        print(f"   Saved Counterfactual Report: {cf_json}")

    print("\n" + "=" * 80)
    print("FINAL RESEARCH EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"Reports saved to: {reports_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
