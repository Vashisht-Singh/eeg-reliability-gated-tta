# End-to-End System Architecture

## Overview

The research system implements a rigorous cross-dataset EEG seizure detection and chronological test-time adaptation pipeline.

```mermaid
flowchart TD
    subgraph S1["Phase 1: Dataset Audit & Verification"]
        D1["CHB-MIT Raw (F:\)"] --> A1["CHB-MIT Audit"]
        D2["Siena Raw (D:\)"] --> A2["Siena Audit"]
        A1 & A2 --> AR["Audit Reports & Channel Inventory"]
    end

    subgraph S2["Phase 2: Data Engineering & Harmonization"]
        AR --> HM["18-Channel Double Banana Harmonizer"]
        HM --> RS["Resampler: 512 Hz -> 256 Hz"]
        RS --> CF["Causal Bandpass & Notch Filter"]
        CF --> WG["Window Generator: 4s Windows"]
        WG --> SP["Patient-Independent Split (chb01 & chb21 bound)"]
    end

    subgraph S3["Phase 3: Source Domain Model Training"]
        SP --> TR["Train 1D CNN Baseline on CHB-MIT"]
        TR --> CK["Best Source Checkpoint (val_auprc)"]
    end

    subgraph S4["Phase 4-6: Chronological Target Stream & Adaptation"]
        D2 --> TS["Siena Chronological Stream"]
        CK --> PR["Step 1: Incumbent Model Predicts Window"]
        TS --> PR
        PR --> PL["Log Prediction & Timestamp"]
        PL --> CG["Step 2: Generate Candidate Update (TTA Objective)"]
        CG --> RG["Step 3: Reliability Assessment & Gate Decision"]
        RG -->|Accept Update| UP["Step 4: Update Model to Candidate"]
        RG -->|Reject Update| RT["Step 4: Retain Incumbent Model"]
        RG -->|Instability Detected| RB["Rollback to Stable Checkpoint"]
    end

    subgraph S5["Phase 7: Offline Evaluation (Target Labels Released)"]
        UP & RT & RB --> AL["Frozen Adaptation & Prediction Logs"]
        TL["Siena Evaluation Annotations"] --> EV["Final Evaluation Runner"]
        AL --> EV
        EV --> MET["Window Metrics: AUROC, AUPRC, F1, MCC"]
        EV --> EVM["Event Metrics: FA/Hour, Latency, Event F1"]
        EV --> CFA["Counterfactual Update-Harm Analysis"]
    end
```

---

## Directory Organization

```
D:\EEG_Research\
├── config/              # Centralized YAML configurations
├── data/
│   ├── manifests/       # Patient-level training/val split CSV manifests
│   ├── metadata/        # Audited inventories (recordings, seizures, channels)
│   ├── processed/       # Incremental cache files (HDF5/NumPy memmap)
│   └── evaluation_annotations/ # Isolated Siena annotations (unlocked in Phase 7 only)
├── src/
│   ├── audit/           # Audit modules (chbmit, siena, channel, reports)
│   ├── data/            # EDF loaders, causal preprocessors, window generators
│   ├── models/          # 1D CNN and lightweight TCN architectures
│   ├── training/        # Source trainer, loss functions, checkpoint manager
│   ├── adaptation/      # TTA baselines (TENT, EATA, SAR) & proposed reliability gate
│   ├── evaluation/      # Window, event, calibration, and counterfactual evaluators
│   └── utils/           # Hardware, reproducibility, memory managers
├── scripts/             # Executable CLI scripts (01 to 10)
├── experiments/         # Logs, metrics, predictions, checkpoints, and reports
└── docs/                # Architecture, methodology, decisions, logs
```
