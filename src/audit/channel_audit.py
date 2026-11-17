"""
Cross-Dataset EEG Channel Audit and Harmonization Module.
Analyzes electrode derivations, channel nomenclature, montage compatibility,
and common channel representations between CHB-MIT and Siena Scalp EEG.
"""

from typing import Dict, List, Any, Set, Tuple
from collections import Counter


# Standard 18-channel Longitudinal Bipolar (Double Banana) Montage Definition
# Grouped by anatomical chains
DOUBLE_BANANA_CHAINS = {
    'Left Temporal Chain': [
        ('FP1-F7', ('FP1', 'F7')),
        ('F7-T7', ('F7', 'T7')),
        ('T7-P7', ('T7', 'P7')),
        ('P7-O1', ('P7', 'O1')),
    ],
    'Left Parasagittal Chain': [
        ('FP1-F3', ('FP1', 'F3')),
        ('F3-C3', ('F3', 'C3')),
        ('C3-P3', ('C3', 'P3')),
        ('P3-O1', ('P3', 'O1')),
    ],
    'Right Parasagittal Chain': [
        ('FP2-F4', ('FP2', 'F4')),
        ('F4-C4', ('F4', 'C4')),
        ('C4-P4', ('C4', 'P4')),
        ('P4-O2', ('P4', 'O2')),
    ],
    'Right Temporal Chain': [
        ('FP2-F8', ('FP2', 'F8')),
        ('F8-T8', ('F8', 'T8')),
        ('T8-P8', ('T8', 'P8')),
        ('P8-O2', ('P8', 'O2')),
    ],
    'Midline Central Chain': [
        ('FZ-CZ', ('FZ', 'CZ')),
        ('CZ-PZ', ('CZ', 'PZ')),
    ]
}

# 10-20 to 10-10 electrode nomenclature equivalence map
ELECTRODE_SYNONYMS = {
    'T7': ['T7', 'T3'],
    'T8': ['T8', 'T4'],
    'P7': ['P7', 'T5'],
    'P8': ['P8', 'T6'],
    'O1': ['O1', '01'],  # OCR/typo digit 0 for letter O
    'O2': ['O2', '02'],
}


def normalize_electrode_name(name: str) -> str:
    """
    Normalize an individual electrode name by stripping prefixes (EEG, REF, etc.),
    punctuation, and whitespace, and converting to uppercase standard.
    """
    clean = name.strip().upper()
    for prefix in ['EEG', 'POL', 'REF']:
        if clean.startswith(prefix):
            clean = clean[len(prefix):].strip()
    clean = clean.replace('-', '').replace('.', '').replace(' ', '')
    return clean


def normalize_bipolar_name(name: str) -> str:
    """
    Normalize a bipolar channel label like 'FP1-F7' or 'EEG FP1-REF'.
    """
    clean = name.strip().upper().replace('EEG', '').strip()
    clean = clean.replace(' ', '').replace('.', '')
    return clean


def analyze_channel_compatibility(
    chbmit_recordings: List[Dict[str, Any]],
    siena_recordings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze montage compatibility and derive exact common representations
    across all recordings in both datasets.
    """
    num_chb_files = len(chbmit_recordings)
    num_siena_files = len(siena_recordings)

    # 1. Flatten all 18 target channels
    target_channels = []
    for chain_name, chain_channels in DOUBLE_BANANA_CHAINS.items():
        for ch_name, (anode, cathode) in chain_channels:
            target_channels.append({
                'channel_name': ch_name,
                'chain': chain_name,
                'anode': anode,
                'cathode': cathode
            })

    # 2. Check coverage in CHB-MIT
    chb_coverage = {tc['channel_name']: 0 for tc in target_channels}
    chb_fully_compatible_files = 0

    for rec in chbmit_recordings:
        labels_norm = set(normalize_bipolar_name(l) for l in rec['channels'])
        has_all = True
        for tc in target_channels:
            ch_name = tc['channel_name']
            if ch_name in labels_norm:
                chb_coverage[ch_name] += 1
            else:
                has_all = False
        if has_all:
            chb_fully_compatible_files += 1

    # 3. Check coverage in Siena (monopolar electrodes needed for derivation)
    siena_coverage = {tc['channel_name']: 0 for tc in target_channels}
    siena_fully_compatible_files = 0

    for rec in siena_recordings:
        labels_norm = set(normalize_electrode_name(l) for l in rec['channels'])
        has_all = True
        for tc in target_channels:
            anode, cathode = tc['anode'], tc['cathode']

            # Check anode presence (accounting for synonyms)
            anode_candidates = ELECTRODE_SYNONYMS.get(anode, [anode])
            anode_found = any(cand in labels_norm for cand in anode_candidates)

            # Check cathode presence (accounting for synonyms)
            cathode_candidates = ELECTRODE_SYNONYMS.get(cathode, [cathode])
            cathode_found = any(cand in labels_norm for cand in cathode_candidates)

            if anode_found and cathode_found:
                siena_coverage[tc['channel_name']] += 1
            else:
                has_all = False

        if has_all:
            siena_fully_compatible_files += 1

    # 4. Build channel inventory list
    inventory = []
    for tc in target_channels:
        ch_name = tc['channel_name']
        anode, cathode = tc['anode'], tc['cathode']

        chb_count = chb_coverage[ch_name]
        chb_pct = (chb_count / num_chb_files * 100.0) if num_chb_files > 0 else 0.0

        siena_count = siena_coverage[ch_name]
        siena_pct = (siena_count / num_siena_files * 100.0) if num_siena_files > 0 else 0.0

        # Derivations
        chb_derivation = f"Direct Bipolar: {ch_name}"
        siena_anode_str = "/".join(ELECTRODE_SYNONYMS.get(anode, [anode]))
        siena_cathode_str = "/".join(ELECTRODE_SYNONYMS.get(cathode, [cathode]))
        siena_derivation = f"Referential Difference: ({siena_anode_str}) - ({siena_cathode_str})"

        status = "COMPATIBLE" if (chb_pct >= 99.0 and siena_pct == 100.0) else "PARTIAL"

        inventory.append({
            'channel_name': ch_name,
            'chain': tc['chain'],
            'chbmit_count': chb_count,
            'chbmit_coverage_pct': round(chb_pct, 2),
            'siena_count': siena_count,
            'siena_coverage_pct': round(siena_pct, 2),
            'chbmit_source_mode': chb_derivation,
            'siena_target_mode': siena_derivation,
            'harmonization_status': status
        })

    return {
        'target_channels_count': len(target_channels),
        'double_banana_channels': [tc['channel_name'] for tc in target_channels],
        'chbmit_fully_compatible_files': chb_fully_compatible_files,
        'chbmit_total_files': num_chb_files,
        'chbmit_compatibility_pct': round(chb_fully_compatible_files / num_chb_files * 100.0, 2) if num_chb_files > 0 else 0.0,
        'siena_fully_compatible_files': siena_fully_compatible_files,
        'siena_total_files': num_siena_files,
        'siena_compatibility_pct': round(siena_fully_compatible_files / num_siena_files * 100.0, 2) if num_siena_files > 0 else 0.0,
        'channel_inventory': inventory
    }
