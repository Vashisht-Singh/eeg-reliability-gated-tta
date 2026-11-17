# Research Experiment and Progress Log

## Project: Cross-Dataset EEG Seizure Detection & Chronological Test-Time Adaptation
- **Source Dataset:** CHB-MIT Scalp EEG (`F:\chb-mit-scalp-eeg-database-1.0.0`)
- **Target Dataset:** Siena Scalp EEG (`D:\Sandhu_Siena_EEG`)
- **Project Root:** `D:\EEG_Research`

---

## Progress Overview

| Phase | Description | Status | Completion Date |
|---|---|---|---|
| **Phase 1** | **Environment and Dataset Inspection & Audit** | **COMPLETED** | 2026-11-18 |
| **Phase 2** | **Data Engineering, Preprocessing, Harmonization & Manifests** | **COMPLETED** | 2026-11-18 |
| Phase 3 | Source Model (1D CNN) Training & Source Validation (CHB-MIT) | Pending | - |
| Phase 4 | Frozen Target Evaluation (Siena) | Pending | - |
| Phase 5 | TTA Baselines (TENT, EATA, SAR) on Siena Stream | Pending | - |
| Phase 6 | Proposed Seizure-Aware Reliability-Gated Adaptation | Pending | - |
| Phase 7 | Final Evaluation, Counterfactual Analysis & Publication Reports | Pending | - |

---

## Detailed Log: Phase 1 — Dataset Audit & Verification

### Tasks Executed:
1. **System & Resource Audit:**
   - Operating System: Windows 10/11
   - Python: 3.10.4
   - Total System RAM: 13.88 GB (Available: ~4.00 GB)
   - GPU: NVIDIA GeForce GTX 1650 (4096 MiB VRAM), Driver 581.83, CUDA 13.0
   - Storage:
     - `D:\`: 150.3 GB Total, 73.5 GB Free (Project Root `D:\EEG_Research`)
     - `C:\`: 237.4 GB Total, 146.4 GB Free
     - `F:\`: 195.3 GB Total, ~4 MB Free (Preserved Read-Only)

2. **Project Directory Structure Created:**
   - Modular hierarchy created at `D:\EEG_Research` (config, data, src, scripts, experiments, models, tests, docs).
   - Central configuration YAMLs authored: `paths.yaml`, `preprocessing.yaml`, `model.yaml`, `training.yaml`, `adaptation.yaml`, `evaluation.yaml`.

3. **Empirical Dataset Audit:**
   - **CHB-MIT Scalp EEG:**
     - 24 patient folders (`chb01` to `chb24`).
     - 686 total EDF files verified.
     - 686 usable EDF files (0 corrupted).
     - 982.94 total recording hours.
     - 141 files with seizures containing exactly 198 annotated seizure events.
     - 100% of files digitized at 256.0 Hz.
     - 683/686 files natively contain all 18 double-banana bipolar channels; 3 CS2-referenced files (`chb12_27-29`) contain the same constituent electrodes.
   - **Siena Scalp EEG:**
     - 14 subject folders (`PN00` to `PN17`).
     - 41 total EDF files verified.
     - 41 usable EDF files (0 corrupted).
     - 141.02 total recording hours.
     - 47 annotated seizure events cataloged across subjects.
     - 100% of files digitized at 512.0 Hz.
     - All 41 files contain all constituent unipolar electrodes required to reconstruct the exact 18-channel longitudinal bipolar montage.
     - Audited and resolved typographical variations (`PNO6` $\rightarrow$ `PN06`, `PN01.edf` $\rightarrow$ `PN01-1.edf`, `PN11-.edf` $\rightarrow$ `PN11-1.edf`).

4. **Generated Reports & Inventories:**
   - `dataset_audit_report.json`
   - `dataset_audit_report.csv`
   - `channel_inventory.csv`
   - `recording_inventory.csv`
   - `seizure_event_inventory.csv`
   - `dataset_audit_summary.md`


---

## Detailed Log: Phase 2 — Data Engineering & Preprocessing

### Tasks Executed:
1. **Incremental EDF Loader (`src/data/edf_loader.py`):**
   - Memory-efficient binary EDF reader with sample-accurate microvolt calibration.
   - Enables random window access and record-range reading with zero memory blow-up.

2. **Montage & Channel Mapping (`src/data/channel_mapper.py` & `src/data/montage.py`):**
   - Implemented `ChannelMapper` transformation matrix:
     - Direct mapping for 683/686 CHB-MIT files.
     - CS2-referenced subtraction for `chb12_27` file.
     - Referential subtraction for 41/41 Siena files.
   - Guaranteed standard 18-channel double-banana ordering.

3. **Causal Preprocessing Pipeline (`src/data/preprocessing.py`):**
   - 2x anti-aliasing polyphase downsampler for Siena (512 Hz $\rightarrow$ 256 Hz).
   - Causal 4th-order Butterworth bandpass (0.5 – 45.0 Hz) and notch filters (60 Hz / 50 Hz).
   - Mathematically verified causality: $h[n] = 0.0$ for $n < n_0$.
   - Label-free artifact and flatline detection.
   - Robust channel-wise z-scoring.

4. **Window Generator & Streaming Modules (`src/data/window_generator.py`, `src/data/source_dataset.py`, `src/data/target_stream.py`):**
   - 4.0-second windows (1024 samples @ 256 Hz).
   - Source domain: 1.0s stride near seizures, 4.0s for background.
   - Target domain: strict chronological streaming with zero labels exposed.

5. **Patient-Independent Splits & Manifests (`scripts/02_prepare_manifests.py`):**
   - `patient_splits.csv`: 17 train, 3 val, 4 test subjects (`chb01` and `chb21` bound together).
   - `source_window_manifest.csv`: 902,084 total windows (11,809 seizure, 890,275 background; 1.31% positive ratio).
   - `target_stream_manifest.csv`: 507,551 unlabeled chronological windows.
   - `data/evaluation_annotations/siena_ground_truth_events.json`: isolated Siena ground truth.
   - Rapid smoke-test manifests: `source_smoke_test_manifest.csv` (2,837 windows) and `target_smoke_test_manifest.csv` (300 windows).

6. **Automated Verification Tests Passed:**
   - `tests/test_patient_split.py` (2/2 passed)
   - `tests/test_channel_mapping.py` (3/3 passed)
   - `tests/test_preprocessing.py` (4/4 passed)
   - `tests/test_label_isolation.py` (4/4 passed)
   - `tests/test_source_dataset.py` (2/2 passed)
