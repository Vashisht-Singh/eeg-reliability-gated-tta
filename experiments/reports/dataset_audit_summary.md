# Research Dataset Audit Summary Report

**Project:** Cross-Dataset EEG Seizure Detection & Chronological Test-Time Adaptation  
**Source Dataset:** CHB-MIT Scalp EEG Database (v1.0.0)  
**Target Dataset:** Siena Scalp EEG Database  
**Audited Date:** 2026-11-18  
**Status:** Phase 1 Complete — All Checks Passed  

---

## 1. Executive Summary

This dataset audit provides a rigorous, verified empirical inventory of the two local EEG databases. Every single EDF file header was inspected directly via raw byte parsing without loading entire files into memory. No files have been modified, deleted, or relocated, strictly adhering to the read-only data protection mandate.

### Key Findings:
- **Source Dataset (CHB-MIT):** 24 patients, 686 EDF files (982.94 hours total), 141 seizure-containing files, **198 annotated seizure events**. Sampling frequency is exactly 256.0 Hz across 100% of files.
- **Target Dataset (Siena):** 14 patients, 41 EDF files (141.02 hours total), 40 seizure-containing files, **73 annotated seizure events**. Sampling frequency is exactly 512.0 Hz across 100% of files.
- **Data Integrity:** **0 corrupted recordings** found across all 727 EDF files in both datasets.
- **Montage Harmonization:** An **18-channel longitudinal bipolar (double-banana) montage** provides 100% cross-dataset electrode representation. CHB-MIT has 683/686 (99.6%) direct files and 3 CS2-referenced files (which synthesize to the exact same bipolar channels). Siena has 41/41 (100.0%) files containing all constituent referential electrodes, enabling exact referential subtraction with zero synthesized or hallucinated channels.

---

## 2. Dataset Comparative Matrix

| Characteristic | CHB-MIT Scalp EEG (Source) | Siena Scalp EEG (Target) | Research Harmonization Strategy |
|---|---|---|---|
| **Root Directory** | `F:\chb-mit-scalp-eeg-database-1.0.0` | `D:\Sandhu_Siena_EEG` | Read-only access; outputs to `D:\EEG_Research` |
| **Subject Count** | 24 subjects (`chb01`–`chb24`) | 14 subjects (`PN00`–`PN17`) | Patient-independent cross-subject splits |
| **Total EDF Files** | 686 recordings | 41 recordings | Processed individually; zero full-dataset RAM loading |
| **Usable Files** | 686 (100.0%) | 41 (100.0%) | No recordings discarded due to file corruption |
| **Total Duration** | 982.94 hours | 141.02 hours | Incremental chunk-based window caching |
| **Seizure Files** | 141 recordings | 40 recordings | Ground truth strictly isolated during TTA |
| **Seizure Events** | 198 events | 73 events | Window-level and event-level evaluation |
| **Sampling Rate** | 256.0 Hz | 512.0 Hz | Resample target (or both) to 256.0 Hz / 128.0 Hz |
| **Native Montage** | Modified 10-20 Bipolar | Referential (Monopolar) | Derive standard 18-channel double banana montage |

---

## 3. Hardware & Resource Environment

- **Host OS:** Windows-10-10.0.26200-SP0  
- **Python Version:** 3.10.4  
- **Total RAM:** 13.88 GB (Available: 4.44 GB)  
- **GPU Hardware:** NVIDIA GeForce GTX 1650 (4096.0 MB VRAM)  
- **Driver / CUDA:** Driver 581.83 / CUDA 13.0 (Driver supported)  
- **Storage Capacity:**  
  - `D:\` (Research Project Root): 73.52 GB free (out of 150.26 GB)  
  - `C:\` (System Drive): 146.42 GB free  
  - `F:\` (Source Dataset Drive): 0.0 GB free (Read-only, fully preserved)  

---

## 4. Channel Harmonization: 18-Channel Longitudinal Bipolar Montage

The standard clinical **double-banana montage** captures both temporal and parasagittal cerebral activity. Below is the exact channel inventory and derivation scheme established for cross-dataset compatibility:

| Channel | Anatomical Chain | CHB-MIT Native Form | Siena Derivation | CHB Coverage | Siena Coverage |
|---|---|---|---|---|---|
| **FP1-F7** | Left Temporal Chain | Direct Bipolar: FP1-F7 | `Referential Difference: (FP1) - (F7)` | 99.56% | 100.0% |
| **F7-T7** | Left Temporal Chain | Direct Bipolar: F7-T7 | `Referential Difference: (F7) - (T7/T3)` | 99.56% | 100.0% |
| **T7-P7** | Left Temporal Chain | Direct Bipolar: T7-P7 | `Referential Difference: (T7/T3) - (P7/T5)` | 99.56% | 100.0% |
| **P7-O1** | Left Temporal Chain | Direct Bipolar: P7-O1 | `Referential Difference: (P7/T5) - (O1/01)` | 99.56% | 100.0% |
| **FP1-F3** | Left Parasagittal Chain | Direct Bipolar: FP1-F3 | `Referential Difference: (FP1) - (F3)` | 99.56% | 100.0% |
| **F3-C3** | Left Parasagittal Chain | Direct Bipolar: F3-C3 | `Referential Difference: (F3) - (C3)` | 99.56% | 100.0% |
| **C3-P3** | Left Parasagittal Chain | Direct Bipolar: C3-P3 | `Referential Difference: (C3) - (P3)` | 99.56% | 100.0% |
| **P3-O1** | Left Parasagittal Chain | Direct Bipolar: P3-O1 | `Referential Difference: (P3) - (O1/01)` | 99.56% | 100.0% |
| **FP2-F4** | Right Parasagittal Chain | Direct Bipolar: FP2-F4 | `Referential Difference: (FP2) - (F4)` | 99.56% | 100.0% |
| **F4-C4** | Right Parasagittal Chain | Direct Bipolar: F4-C4 | `Referential Difference: (F4) - (C4)` | 99.56% | 100.0% |
| **C4-P4** | Right Parasagittal Chain | Direct Bipolar: C4-P4 | `Referential Difference: (C4) - (P4)` | 99.56% | 100.0% |
| **P4-O2** | Right Parasagittal Chain | Direct Bipolar: P4-O2 | `Referential Difference: (P4) - (O2/02)` | 99.56% | 100.0% |
| **FP2-F8** | Right Temporal Chain | Direct Bipolar: FP2-F8 | `Referential Difference: (FP2) - (F8)` | 99.56% | 100.0% |
| **F8-T8** | Right Temporal Chain | Direct Bipolar: F8-T8 | `Referential Difference: (F8) - (T8/T4)` | 99.56% | 100.0% |
| **T8-P8** | Right Temporal Chain | Direct Bipolar: T8-P8 | `Referential Difference: (T8/T4) - (P8/T6)` | 99.56% | 100.0% |
| **P8-O2** | Right Temporal Chain | Direct Bipolar: P8-O2 | `Referential Difference: (P8/T6) - (O2/02)` | 99.56% | 100.0% |
| **FZ-CZ** | Midline Central Chain | Direct Bipolar: FZ-CZ | `Referential Difference: (FZ) - (CZ)` | 99.56% | 100.0% |
| **CZ-PZ** | Midline Central Chain | Direct Bipolar: CZ-PZ | `Referential Difference: (CZ) - (PZ)` | 99.56% | 100.0% |

### Special Channel Derivations Resolved:
1. **`chb12_27.edf`**: Recorded with `CS2` reference (e.g. `FP1-CS2`, `F7-CS2`). Derivation `(FP1-CS2) - (F7-CS2) = FP1-F7` yields the exact bipolar signal.
2. **`chb12_28.edf` & `chb12_29.edf`**: Recorded in referential montage (`FP1`, `F7`). Standard difference `FP1 - F7` yields the bipolar signal directly.
3. **Siena Referential Derivations**: All 41 Siena files contain referential electrodes `EEG Fp1`, `EEG F7`, `EEG T3`, `EEG T5`, `EEG O1`, `EEG F3`, `EEG C3`, `EEG P3`, `EEG Fp2`, `EEG F4`, `EEG C4`, `EEG P4`, `EEG O2`, `EEG F8`, `EEG T4`, `EEG T6`, `EEG Fz`, `EEG Cz`, `EEG Pz`. Deriving `(Anode - Cathode)` yields exact longitudinal bipolar signals perfectly matching CHB-MIT.
4. **10-20 to 10-10 Terminology**: `T3` = `T7`, `T4` = `T8`, `T5` = `P7`, `T6` = `P8` standard clinical equivalence verified.

---

## 5. Critical Data Leakage & Adaptation Isolation Protocol

- **CHB-MIT Subject Split:** `chb01` and `chb21` belong to the same individual (confirmed in clinical literature) and are strictly locked into the same split partition.
- **Siena Isolation:** Ground-truth seizure annotations from Siena (`Seizures-list-PNXX.txt`) are placed in `data/evaluation_annotations/` and are strictly barred from the data loader during adaptation. They will only be loaded by the final evaluation script after all adaptation predictions and checkpoint logs are written.

---

## 6. Implementation Readiness for Phase 2

- [x] Datasets inspected and verified read-only.
- [x] 100% of EDF headers parsed and verified.
- [x] 100% of seizure events cataloged with onset, offset, and duration.
- [x] Channel harmonization scheme finalized.
- [x] Hardware constraints audited and memory budget calculated.
- [x] Next Phase: Phase 2 — Data Engineering (incremental EDF loader, montage transformer, window generator, patient manifests).