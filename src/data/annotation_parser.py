"""
Ground-Truth Seizure Annotation Parser and Isolation Store.
Extracts and isolates evaluation annotations for CHB-MIT and Siena.

CRITICAL RESEARCH PROTOCOL:
Siena ground-truth annotations are stored exclusively for offline Phase 7 evaluation.
Under no circumstances may the online streaming adapter or target data loader
access, load, or query these annotations.
"""

import os
import json
import csv
from typing import Dict, List, Any
from src.audit.chbmit_audit import parse_chbmit_summary
from src.audit.siena_audit import parse_siena_seizure_list, parse_edf_header


def export_isolated_siena_annotations(siena_raw_dir: str, output_dir: str) -> str:
    """
    Extracts all Siena seizure annotations and saves them into an isolated
    offline evaluation directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "siena_ground_truth_events.json")

    subject_dirs = sorted([d for d in os.listdir(siena_raw_dir)
                          if os.path.isdir(os.path.join(siena_raw_dir, d)) and d.upper().startswith('PN')])

    all_events = {}

    for subj in subject_dirs:
        subj_dir = os.path.join(siena_raw_dir, subj)
        edf_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.edf')])

        headers = {}
        for f in edf_files:
            try:
                headers[f] = parse_edf_header(os.path.join(subj_dir, f))
            except Exception:
                pass

        list_path = os.path.join(subj_dir, f"Seizures-list-{subj}.txt")
        events = parse_siena_seizure_list(list_path, subj, edf_files, headers)

        for ev in events:
            fname = ev['filename']
            if fname not in all_events:
                all_events[fname] = []
            all_events[fname].append({
                'subject_id': subj,
                'onset_sec': ev['onset_sec'],
                'offset_sec': ev['offset_sec'],
                'duration_sec': ev['duration_sec']
            })

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_events, f, indent=2)

    return out_json


def load_offline_target_annotations(annotation_json_path: str) -> Dict[str, List[Dict[str, float]]]:
    """
    Exclusively called by the offline evaluation runner in Phase 7.
    """
    if not os.path.exists(annotation_json_path):
        raise FileNotFoundError(f"Isolated annotations not found at: {annotation_json_path}")

    with open(annotation_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
