# EEG Seizure Detection & Chronological Test-Time Adaptation

A modular research codebase for cross-dataset EEG seizure detection transferring from **CHB-MIT** to **Siena Scalp EEG**, incorporating reliability-gated online adaptation.

---

## Overview

- **Source Domain:** CHB-MIT Scalp EEG (686 recordings, 24 subjects, 256 Hz)
- **Target Domain:** Siena Scalp EEG (41 recordings, 14 subjects, 512 Hz)
- **Harmonized Representation:** 18-Channel Longitudinal Bipolar (Double Banana) montage
- **Core Architecture:** 1D Temporal CNN with modular TTA controllers (Frozen, TENT, EATA, SAR, and Reliability-Gated adaptation)

---

## Directory Structure

```
├── config/                  # Experiment and model YAML configurations
├── data/
│   ├── manifests/           # Patient-independent train/val splits
│   ├── metadata/            # Inventory CSVs generated from dataset audit
│   └── evaluation_annotations/ # Isolated target annotations
├── src/
│   ├── audit/               # Dataset inspection and channel auditing
│   ├── data/                # Streaming EDF loaders and causal filters
│   ├── models/              # 1D CNN and TCN architectures
│   ├── training/            # Source training and early stopping
│   ├── adaptation/          # Online TTA and reliability controllers
│   ├── evaluation/          # Window and event-level evaluation
│   └── utils/               # Logging, memory, and reproducibility utilities
├── scripts/                 # Sequential pipeline execution scripts
├── experiments/             # Audit reports, metrics, and logs
└── tests/                   # Data integrity and label isolation tests
```

---

## Quickstart

### 1. Requirements

Install required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Dataset Audit (Phase 1)

Audit local datasets and generate inventory tables:

```bash
python scripts/01_audit_datasets.py
```

### 3. Run Integrity Tests

```bash
python tests/test_data_integrity.py
```

---

## Experimental Protocol & Integrity

1. **Patient Independence:** `chb01` and `chb21` are bound to the same training split partition.
2. **Label Isolation:** Target Siena seizure labels remain strictly isolated during adaptation and are accessed only during offline evaluation.
3. **Causal Processing:** Streaming filters and window generators enforce causality to prevent future-information leakage.
