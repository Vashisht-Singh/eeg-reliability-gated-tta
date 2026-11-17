"""
EEG Channel Mapping and Harmonization Module.
Builds linear derivation plans to project native raw EEG signals from
both CHB-MIT and Siena into the standard 18-channel Longitudinal Bipolar (Double Banana) montage.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from src.audit.channel_audit import DOUBLE_BANANA_CHAINS, ELECTRODE_SYNONYMS, normalize_electrode_name, normalize_bipolar_name


class ChannelMapper:
    """
    Computes a linear transformation matrix to extract standard 18-channel
    double banana bipolar montage from any raw EDF recording.
    """

    TARGET_CHANNELS = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ'
    ]

    def __init__(self, raw_channel_labels: List[str]):
        self.raw_labels = raw_channel_labels
        self.num_raw_signals = len(raw_channel_labels)
        self._build_derivation_matrix()

    def _build_derivation_matrix(self):
        """
        Build linear mapping: target_18xT = M @ raw_signals
        """
        # 1. Cleaned label lookups
        bipolar_clean = [normalize_bipolar_name(lbl) for lbl in self.raw_labels]
        monopolar_clean = [normalize_electrode_name(lbl) for lbl in self.raw_labels]

        # Map cleaned -> original indices
        bipolar_to_idx = {name: idx for idx, name in enumerate(bipolar_clean)}
        monopolar_to_idx = {name: idx for idx, name in enumerate(monopolar_clean)}

        # Also support CS2 references: e.g. 'FP1-CS2' -> 'FP1'
        cs2_to_idx = {}
        for idx, lbl in enumerate(self.raw_labels):
            lbl_up = lbl.strip().upper().replace(' ', '')
            if '-CS2' in lbl_up:
                anode = lbl_up.replace('-CS2', '')
                cs2_to_idx[anode] = idx

        # Target definitions
        channel_defs = {}
        for chain_name, pairs in DOUBLE_BANANA_CHAINS.items():
            for ch_name, (anode, cathode) in pairs:
                channel_defs[ch_name] = (anode, cathode)

        self.required_indices = set()
        derivation_plan = []  # list of tuples: (target_ch_name, [(raw_idx, weight), ...])

        for ch_name in self.TARGET_CHANNELS:
            anode, cathode = channel_defs[ch_name]

            # Case A: Direct bipolar match (standard CHB-MIT)
            if ch_name in bipolar_to_idx:
                raw_idx = bipolar_to_idx[ch_name]
                derivation_plan.append((ch_name, [(raw_idx, 1.0)]))
                self.required_indices.add(raw_idx)
                continue

            # Case B: CS2 reference match (CHB-MIT chb12_27: (A-CS2) - (B-CS2) = A - B)
            if anode in cs2_to_idx and cathode in cs2_to_idx:
                idx_a = cs2_to_idx[anode]
                idx_b = cs2_to_idx[cathode]
                derivation_plan.append((ch_name, [(idx_a, 1.0), (idx_b, -1.0)]))
                self.required_indices.add(idx_a)
                self.required_indices.add(idx_b)
                continue

            # Case C: Referential subtraction (Siena Scalp EEG & CHB-MIT chb12_28/29)
            # Find anode candidate
            anode_candidates = ELECTRODE_SYNONYMS.get(anode, [anode])
            idx_a = None
            for cand in anode_candidates:
                if cand in monopolar_to_idx:
                    idx_a = monopolar_to_idx[cand]
                    break

            # Find cathode candidate
            cathode_candidates = ELECTRODE_SYNONYMS.get(cathode, [cathode])
            idx_b = None
            for cand in cathode_candidates:
                if cand in monopolar_to_idx:
                    idx_b = monopolar_to_idx[cand]
                    break

            if idx_a is not None and idx_b is not None:
                derivation_plan.append((ch_name, [(idx_a, 1.0), (idx_b, -1.0)]))
                self.required_indices.add(idx_a)
                self.required_indices.add(idx_b)
                continue

            raise ValueError(f"Cannot resolve derivation for target channel {ch_name} from raw labels: {self.raw_labels}")

        self.sorted_required_indices = sorted(list(self.required_indices))
        raw_to_sub_idx = {raw_idx: sub_idx for sub_idx, raw_idx in enumerate(self.sorted_required_indices)}

        # Build derivation matrix of shape (18, len(sorted_required_indices))
        self.transformation_matrix = np.zeros((len(self.TARGET_CHANNELS), len(self.sorted_required_indices)), dtype=np.float32)

        for target_idx, (ch_name, terms) in enumerate(derivation_plan):
            for raw_idx, weight in terms:
                sub_idx = raw_to_sub_idx[raw_idx]
                self.transformation_matrix[target_idx, sub_idx] += weight

    def get_required_channel_indices(self) -> List[int]:
        """Indices in the raw EDF file that must be loaded to compute the 18 channels."""
        return self.sorted_required_indices

    def transform(self, raw_subset_signals: np.ndarray) -> np.ndarray:
        """
        Apply derivation matrix to the subset of loaded raw signals.
        Input: (len(sorted_required_indices), n_samples)
        Output: (18, n_samples) in double banana order.
        """
        if raw_subset_signals.shape[0] != len(self.sorted_required_indices):
            raise ValueError(
                f"Input has {raw_subset_signals.shape[0]} channels, expected {len(self.sorted_required_indices)}"
            )
        return self.transformation_matrix @ raw_subset_signals
