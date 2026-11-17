"""
Montage and Electrode Topology Definitions.
Defines anatomical chains, cortical lobe assignments, and channel groupings
for the 18-channel Longitudinal Bipolar (Double Banana) montage.
"""

from typing import Dict, List, Tuple

# Anatomical Lobe / Regional Mapping for the 18 bipolar channels
CHANNEL_REGION_MAP = {
    'FP1-F7': 'Anterior Frontal-Temporal (Left)',
    'F7-T7': 'Mid-Temporal (Left)',
    'T7-P7': 'Posterior Temporal-Parietal (Left)',
    'P7-O1': 'Parietal-Occipital (Left)',

    'FP1-F3': 'Anterior Frontal (Left)',
    'F3-C3': 'Frontal-Central (Left)',
    'C3-P3': 'Central-Parietal (Left)',
    'P3-O1': 'Parietal-Occipital (Left)',

    'FP2-F4': 'Anterior Frontal (Right)',
    'F4-C4': 'Frontal-Central (Right)',
    'C4-P4': 'Central-Parietal (Right)',
    'P4-O2': 'Parietal-Occipital (Right)',

    'FP2-F8': 'Anterior Frontal-Temporal (Right)',
    'F8-T8': 'Mid-Temporal (Right)',
    'T8-P8': 'Posterior Temporal-Parietal (Right)',
    'P8-O2': 'Parietal-Occipital (Right)',

    'FZ-CZ': 'Midline Frontal-Central',
    'CZ-PZ': 'Midline Central-Parietal'
}

# Anatomical Chains
CHAINS = {
    'left_temporal': ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1'],
    'left_parasagittal': ['FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1'],
    'right_parasagittal': ['FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'],
    'right_temporal': ['FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2'],
    'midline': ['FZ-CZ', 'CZ-PZ']
}


def get_target_channels() -> List[str]:
    """Return ordered list of the standard 18 double banana channels."""
    return [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ'
    ]


def get_channel_chain(ch_name: str) -> str:
    """Return anatomical chain for a channel."""
    for chain, chs in CHAINS.items():
        if ch_name in chs:
            return chain
    return 'unknown'
