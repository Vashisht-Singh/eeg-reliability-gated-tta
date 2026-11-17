"""
Script 02: Prepare Patient Splits and Window Manifests.
Executes Phase 2 data engineering:
  1. Creates patient-independent train/val/test splits (chb01 & chb21 bound).
  2. Generates window manifests for CHB-MIT source domain.
  3. Generates chronological streaming manifests for Siena target domain (unlabeled).
  4. Exports isolated ground-truth annotations for Siena into offline evaluation store.
  5. Creates rapid smoke-test manifests for end-to-end verification.
"""

import os
import sys
import csv
import json
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.window_generator import WindowGenerator
from src.data.annotation_parser import export_isolated_siena_annotations


def main():
    print("=" * 80)
    print("PHASE 2: PREPARING PATIENT SPLITS AND WINDOW MANIFESTS")
    print("=" * 80)

    metadata_dir = os.path.join(PROJECT_ROOT, "data", "metadata")
    manifests_dir = os.path.join(PROJECT_ROOT, "data", "manifests")
    eval_annot_dir = os.path.join(PROJECT_ROOT, "data", "evaluation_annotations")
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(eval_annot_dir, exist_ok=True)

    # 1. Define Patient Splits (Strictly binding chb01 and chb21 together)
    train_patients = [
        "chb01", "chb02", "chb03", "chb05", "chb06", "chb07", "chb08",
        "chb09", "chb10", "chb11", "chb14", "chb15", "chb18", "chb19",
        "chb20", "chb21", "chb22"
    ]
    val_patients = ["chb04", "chb16", "chb23"]
    test_patients = ["chb12", "chb13", "chb17", "chb24"]

    # Verify no subject leakage
    set_train = set(train_patients)
    set_val = set(val_patients)
    set_test = set(test_patients)
    assert len(set_train.intersection(set_val)) == 0, "Patient leakage between train and val!"
    assert len(set_train.intersection(set_test)) == 0, "Patient leakage between train and test!"
    assert len(set_val.intersection(set_test)) == 0, "Patient leakage between val and test!"
    assert "chb01" in set_train and "chb21" in set_train, "chb01 and chb21 must be in the same split!"

    print("\n1. Patient Split Definition:")
    print(f"   Train Patients ({len(train_patients)}): {train_patients}")
    print(f"   Val Patients   ({len(val_patients)}): {val_patients}")
    print(f"   Test Patients  ({len(test_patients)}): {test_patients}")

    split_csv_path = os.path.join(manifests_dir, "patient_splits.csv")
    with open(split_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "split_role", "bound_subject"])
        for p in train_patients:
            bound = "chb21" if p == "chb01" else ("chb01" if p == "chb21" else "")
            writer.writerow([p, "train", bound])
        for p in val_patients:
            writer.writerow([p, "val", ""])
        for p in test_patients:
            writer.writerow([p, "source_test", ""])
    print(f"   Saved: {split_csv_path}")

    # 2. Load recording inventory and seizure inventory
    rec_inv_path = os.path.join(metadata_dir, "recording_inventory.csv")
    seiz_inv_path = os.path.join(metadata_dir, "seizure_event_inventory.csv")

    with open(rec_inv_path, 'r', encoding='utf-8') as f:
        all_recordings = list(csv.DictReader(f))

    seizures_by_file = defaultdict(list)
    with open(seiz_inv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            seizures_by_file[row['filename']].append({
                'onset_sec': float(row['onset_sec']),
                'offset_sec': float(row['offset_sec']),
                'duration_sec': float(row['duration_sec'])
            })

    win_gen = WindowGenerator(
        window_duration_sec=4.0,
        target_fs=256.0,
        source_seizure_stride_sec=1.0,
        source_bg_stride_sec=4.0,
        target_stream_stride_sec=1.0,
        seizure_overlap_threshold=0.5
    )

    # 3. Generate CHB-MIT Source Windows
    print("\n2. Generating Source Domain (CHB-MIT) Window Manifest...")
    chb_recordings = [r for r in all_recordings if r['dataset'] == 'CHB-MIT']
    source_windows = []

    # Map patient to split
    split_map = {p: 'train' for p in train_patients}
    split_map.update({p: 'val' for p in val_patients})
    split_map.update({p: 'source_test' for p in test_patients})

    for rec in chb_recordings:
        fname = rec['filename']
        subj = rec['subject_id']
        fpath = rec['filepath']
        dur = float(rec['duration_sec'])
        file_seizures = seizures_by_file.get(fname, [])

        rec_windows = win_gen.generate_source_windows_for_recording(
            subject_id=subj,
            filename=fname,
            filepath=fpath,
            total_duration_sec=dur,
            seizures=file_seizures
        )
        for w in rec_windows:
            w['split'] = split_map.get(subj, 'unknown')
        source_windows.extend(rec_windows)

    source_manifest_path = os.path.join(manifests_dir, "source_window_manifest.csv")
    with open(source_manifest_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['window_id', 'dataset', 'subject_id', 'filename', 'filepath', 'start_sec', 'end_sec', 'duration_sec', 'label', 'split']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(source_windows)

    n_pos = sum(1 for w in source_windows if w['label'] == 1)
    n_neg = len(source_windows) - n_pos
    print(f"   Generated {len(source_windows)} source windows across {len(chb_recordings)} files.")
    print(f"   Seizure windows: {n_pos} | Background windows: {n_neg} (Pos class ratio: {n_pos/len(source_windows)*100:.2f}%)")
    print(f"   Saved: {source_manifest_path}")

    # 4. Generate Siena Target Stream Manifest (STRICTLY UNLABELED)
    print("\n3. Generating Target Domain (Siena) Chronological Stream Manifest...")
    siena_recordings = [r for r in all_recordings if r['dataset'] == 'Siena']
    # Sort strictly chronologically by subject and filename
    siena_recordings.sort(key=lambda r: (r['subject_id'], r['filename']))
    target_windows = []

    for rec in siena_recordings:
        fname = rec['filename']
        subj = rec['subject_id']
        fpath = rec['filepath']
        dur = float(rec['duration_sec'])

        rec_windows = win_gen.generate_target_stream_windows_for_recording(
            subject_id=subj,
            filename=fname,
            filepath=fpath,
            total_duration_sec=dur
        )
        target_windows.extend(rec_windows)

    target_manifest_path = os.path.join(manifests_dir, "target_stream_manifest.csv")
    with open(target_manifest_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['window_id', 'dataset', 'subject_id', 'filename', 'filepath', 'start_sec', 'end_sec', 'duration_sec', 'label']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(target_windows)

    print(f"   Generated {len(target_windows)} target streaming windows across {len(siena_recordings)} files.")
    print(f"   Target labels: STRICTLY UNLABELED (All None)")
    print(f"   Saved: {target_manifest_path}")

    # 5. Export Isolated Siena Evaluation Annotations
    print("\n4. Isolating Siena Evaluation Annotations for Phase 7...")
    siena_raw_dir = r"D:\Sandhu_Siena_EEG"
    annot_json_path = export_isolated_siena_annotations(siena_raw_dir, eval_annot_dir)
    print(f"   Saved isolated evaluation annotations: {annot_json_path}")

    # 6. Generate Smoke-Test Manifests (Small 3-file subsets for fast verification)
    print("\n5. Generating Smoke-Test Subset Manifests...")
    smoke_source = [w for w in source_windows if w['filename'] in ['chb01_03.edf', 'chb01_04.edf', 'chb01_01.edf']]
    smoke_source_path = os.path.join(manifests_dir, "source_smoke_test_manifest.csv")
    with open(smoke_source_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['window_id', 'dataset', 'subject_id', 'filename', 'filepath', 'start_sec', 'end_sec', 'duration_sec', 'label', 'split'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(smoke_source)

    smoke_target = [w for w in target_windows if w['filename'] == 'PN00-1.edf'][:300]  # First 300 streaming windows of PN00-1
    smoke_target_path = os.path.join(manifests_dir, "target_smoke_test_manifest.csv")
    with open(smoke_target_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['window_id', 'dataset', 'subject_id', 'filename', 'filepath', 'start_sec', 'end_sec', 'duration_sec', 'label'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(smoke_target)

    print(f"   Source smoke test: {len(smoke_source)} windows ({smoke_source_path})")
    print(f"   Target smoke test: {len(smoke_target)} windows ({smoke_target_path})")

    print("\n" + "=" * 80)
    print("PHASE 2 MANIFEST PREPARATION SUCCESSFULLY COMPLETED!")
    print("=" * 80)


if __name__ == '__main__':
    main()
