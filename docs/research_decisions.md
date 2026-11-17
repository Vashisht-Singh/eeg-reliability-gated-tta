# Research Decisions and Technical Rationale (Phase 1)

This document records the foundational architectural, clinical neurophysiology, signal processing, and experimental integrity decisions for the cross-dataset EEG seizure detection project.

---

## Decision 1: Common Montage Selection — Standard 18-Channel Longitudinal Bipolar (Double Banana)

### Context:
- **CHB-MIT** scalp recordings are natively stored in a bipolar derivation montage (mostly Double Banana).
- **Siena Scalp EEG** recordings are natively stored in a referential (monopolar) montage referenced to a common physical reference electrode.
- A cross-dataset neural network requires consistent spatial and electrical derivations across both domains.

### Decision:
Adopt the standard **18-channel longitudinal bipolar (Double Banana) montage** as the unified cross-dataset input representation:
1. `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1` (Left Temporal Chain)
2. `FP1-F3`, `F3-C3`, `C3-P3`, `P3-O1` (Left Parasagittal Chain)
3. `FP2-F4`, `F4-C4`, `C4-P4`, `P4-O2` (Right Parasagittal Chain)
4. `FP2-F8`, `F8-T8`, `T8-P8`, `P8-O2` (Right Temporal Chain)
5. `FZ-CZ`, `CZ-PZ` (Midline Central Chain)

### Neurophysiological & Mathematical Justification:
- Clinical electroencephalographers rely on the double-banana montage because subtraction between adjacent electrodes cancels out common reference noise and distant muscle artifacts while highlighting localized cortical epileptiform discharges.
- In Siena, because every recording contains all unipolar constituent electrodes (`Fp1`, `F7`, `T3`, `T5`, `O1`, `F3`, `C3`, `P3`, `Fp2`, `F4`, `C4`, `P4`, `O2`, `F8`, `T4`, `T6`, `Fz`, `Cz`, `Pz`), exact bipolar derivations are obtained by linear subtraction:
  $$\text{Ch}_{A-B}(t) = (V_A(t) - V_{\text{ref}}(t)) - (V_B(t) - V_{\text{ref}}(t)) = V_A(t) - V_B(t)$$
- In CHB-MIT, 683 of 686 recordings natively contain all 18 channels. The 3 remaining files (`chb12_27`, `chb12_28`, `chb12_29`) were verified to contain the same unipolar/CS2 derivations, enabling identical reconstruction.
- No artificial channels are synthesized or hallucinated.

---

## Decision 2: 10-20 to Modified 10-10 Nomenclature Equivalence

### Context:
Different clinical recording centers label intermediate temporal and parietal electrodes under classic 10-20 or modified 10-10 nomenclature.

### Decision:
Formally map the following standard neurophysiological synonyms:
- `T3` $\equiv$ `T7`
- `T4` $\equiv$ `T8`
- `T5` $\equiv$ `P7`
- `T6` $\equiv$ `P8`
- `01` / `02` $\equiv$ `O1` / `O2` (accounting for OCR or typographical zero substitution)

---

## Decision 3: Sampling Frequency Harmonization

### Context:
- CHB-MIT recordings are digitized at 256.0 Hz.
- Siena recordings are digitized at 512.0 Hz.

### Decision:
Harmonize the input stream by downsampling Siena by an exact factor of 2 (512 Hz $\rightarrow$ 256 Hz) using polyphase filtering with an anti-aliasing low-pass filter at the Nyquist cutoff.
- The 256 Hz rate preserves all physiological frequency bands (Delta 0.5–4 Hz, Theta 4–8 Hz, Alpha 8–13 Hz, Beta 13–30 Hz, Gamma 30–45 Hz).
- Keeps memory footprint conservative (1024 samples per 4-second window) suitable for the available 4 GB GPU VRAM.

---

## Decision 4: Causal Filtering for Test-Time Adaptation Integrity

### Context:
Filtering with zero-phase bidirectional filters (e.g. `scipy.signal.filtfilt`) uses backward filtering, which leaks future sample amplitudes into past samples. While acceptable in retrospective batch analysis, this violates temporal causality in streaming online adaptation.

### Decision:
Incorporate causal filtering (forward-only 4th-order Butterworth IIR or FIR filter) for the chronological target stream to guarantee zero future-information leakage during test-time adaptation.

---

## Decision 5: Patient-Independent Partitioning and `chb01` / `chb21` Binding

### Context:
In CHB-MIT, clinical documentation and PhysioNet notes establish that subject `chb01` and subject `chb21` represent the exact same female pediatric patient recorded at different times.

### Decision:
- `chb01` and `chb21` are strictly bound together in the training set partition. Under no circumstances may one be in training and the other in validation.
- All train/val splits are strictly patient-independent. No window from any patient in the validation set is ever seen during source training.

---

## Decision 6: Target Domain Label Isolation Protocol

### Context:
The core research hypothesis examines unsupervised, label-free test-time adaptation on Siena. Accidental exposure to Siena seizure labels during adaptation would invalidate the research.

### Decision:
- Siena seizure annotation lists are stored separately in `data/evaluation_annotations/` and are inaccessible to the streaming data loader and adaptation controller.
- The adaptation controller produces chronological predictions and update logs.
- Siena labels are unlocked strictly during Phase 7 post-adaptation offline evaluation.
- Automated tests (`tests/test_label_isolation.py`) will explicitly verify that the adaptation module cannot import or access target labels.


---

## Decision 7: Chunk-Based Incremental EDF Loading Architecture

### Context:
Standard deep learning pipelines often convert entire EEG datasets into tens of gigabytes of uncompressed `.npy` or `.h5` files, requiring 64+ GB of RAM. The local machine has ~4 GB available RAM.

### Decision:
Implemented `IncrementalEDFLoader` using standard-library binary seeking:
- Calculates exact physical gain $k_i = \frac{\text{phys\_max}_i - \text{phys\_min}_i}{\text{dig\_max}_i - \text{dig\_min}_i}$ and bias $b_i = \text{phys\_min}_i - k_i \cdot \text{dig\_min}_i$.
- Window indexing reads only the exact byte range corresponding to the target 4-second chunk from disk.
- Maximum RAM consumption per window is $< 20\text{ MB}$, completely eliminating out-of-memory risks.

---

## Decision 8: Adaptive Class-Balanced Window Generation

### Context:
In long continuous EEG recordings, clinical seizures occupy $< 1.5\%$ of the total time. A uniform non-overlapping windowing strategy yields severe imbalance ($> 1:100$).

### Decision:
- **Seizure Intervals:** 1.0-second sliding stride ($75\%$ temporal overlap), oversampling transient seizure morphology.
- **Background Intervals:** 4.0-second non-overlapping stride.
- **Overlap Criterion:** A window is labeled positive if and only if $\ge 50\%$ of its duration falls within an annotated clinical seizure.
- **Target Domain (Siena):** Continuous 1.0-second chronological hop with zero labels attached during adaptation.

---

## Decision 9: Causal Second-Order Sections (SOS) Filtering

### Context:
Bidirectional filtering (`filtfilt`) violates causality by running backwards through time. Furthermore, high-order polynomial filters (`ba` coefficients) suffer from numerical instability in single precision.

### Decision:
- Implemented forward-only Second-Order Sections (`sosfilt`) Butterworth 4th-order filter ($0.5 - 45.0\text{ Hz}$).
- Mathematical test verified: impulse response $h[n] = 0.0$ for $n < n_0$, proving zero future information leakage into past or present windows.
