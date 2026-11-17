"""
CHB-MIT Scalp EEG Dataset Audit Module.
Scans and audits raw EDF recordings, summary files, and seizure annotations
for the CHB-MIT database without modifying any source files.
"""

import os
import re
from typing import Dict, List, Any, Tuple
from collections import Counter


def parse_edf_header(filepath: str) -> Dict[str, Any]:
    """
    Parse an EDF file header strictly using Python standard library.
    Reads metadata, signal headers, channel names, and sampling rates
    without loading signal data into memory.
    """
    with open(filepath, 'rb') as f:
        raw_header = f.read(256)
        if len(raw_header) < 256:
            raise ValueError(f"File {filepath} is smaller than 256 bytes (invalid EDF header)")

        version = raw_header[0:8].decode('latin1').strip()
        patient_id = raw_header[8:88].decode('latin1').strip()
        rec_id = raw_header[88:168].decode('latin1').strip()
        start_date = raw_header[168:176].decode('latin1').strip()
        start_time = raw_header[176:184].decode('latin1').strip()
        header_bytes = int(raw_header[184:192].decode('latin1').strip())
        reserved = raw_header[192:236].decode('latin1').strip()
        num_records = int(raw_header[236:244].decode('latin1').strip())
        record_duration = float(raw_header[244:252].decode('latin1').strip())
        ns = int(raw_header[252:256].decode('latin1').strip())

        labels = [f.read(16).decode('latin1').strip() for _ in range(ns)]
        transducers = [f.read(80).decode('latin1').strip() for _ in range(ns)]
        phys_dims = [f.read(8).decode('latin1').strip() for _ in range(ns)]
        phys_mins = [float(f.read(8).decode('latin1').strip()) for _ in range(ns)]
        phys_maxs = [float(f.read(8).decode('latin1').strip()) for _ in range(ns)]
        dig_mins = [int(f.read(8).decode('latin1').strip()) for _ in range(ns)]
        dig_maxs = [int(f.read(8).decode('latin1').strip()) for _ in range(ns)]
        prefilters = [f.read(80).decode('latin1').strip() for _ in range(ns)]
        samples_per_rec = [int(f.read(8).decode('latin1').strip()) for _ in range(ns)]
        reserved_signals = [f.read(32).decode('latin1').strip() for _ in range(ns)]

        fs = [samples_per_rec[i] / record_duration if record_duration > 0 else 0.0 for i in range(ns)]
        duration_sec = num_records * record_duration

        return {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'version': version,
            'patient_id': patient_id,
            'rec_id': rec_id,
            'start_date': start_date,
            'start_time': start_time,
            'header_bytes': header_bytes,
            'num_records': num_records,
            'record_duration': record_duration,
            'num_signals': ns,
            'channel_labels': labels,
            'sampling_frequencies': fs,
            'sampling_frequency': fs[0] if fs else 0.0,
            'duration_sec': duration_sec,
            'duration_hours': duration_sec / 3600.0,
            'physical_dimensions': phys_dims,
        }


def parse_chbmit_summary(summary_path: str) -> Dict[str, Any]:
    """
    Parse a CHB-MIT patient summary text file (chbXX-summary.txt).
    Extracts listed sampling rate, channels, and per-recording seizure annotations.
    """
    results = {
        'sampling_rate': None,
        'channels': [],
        'files': {}
    }

    if not os.path.exists(summary_path):
        return results

    current_file = None
    with open(summary_path, 'r', encoding='latin1', errors='replace') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.lower().startswith('data sampling rate:'):
            m = re.search(r'(\d+)', line)
            if m:
                results['sampling_rate'] = int(m.group(1))

        elif line.lower().startswith('channel ') and ':' in line:
            parts = line.split(':', 1)
            ch_name = parts[1].strip()
            results['channels'].append(ch_name)

        elif line.lower().startswith('file name:'):
            current_file = line.split(':', 1)[1].strip()
            if current_file not in results['files']:
                results['files'][current_file] = {
                    'start_time': None,
                    'end_time': None,
                    'num_seizures': 0,
                    'seizures': []
                }

        elif current_file and line.lower().startswith('file start time:'):
            results['files'][current_file]['start_time'] = line.split(':', 1)[1].strip()

        elif current_file and line.lower().startswith('file end time:'):
            results['files'][current_file]['end_time'] = line.split(':', 1)[1].strip()

        elif current_file and line.lower().startswith('number of seizures in file:'):
            m = re.search(r'(\d+)', line)
            if m:
                results['files'][current_file]['num_seizures'] = int(m.group(1))

        elif current_file and 'seizure' in line.lower() and 'start time:' in line.lower():
            m_start = re.search(r'(\d+)\s*seconds', line, re.IGNORECASE)
            start_sec = int(m_start.group(1)) if m_start else None

            # Look ahead for matching end time
            end_sec = None
            j = i + 1
            while j < min(i + 4, len(lines)):
                next_line = lines[j].strip()
                if 'seizure' in next_line.lower() and 'end time:' in next_line.lower():
                    m_end = re.search(r'(\d+)\s*seconds', next_line, re.IGNORECASE)
                    if m_end:
                        end_sec = int(m_end.group(1))
                        i = j  # advance
                        break
                j += 1

            if start_sec is not None and end_sec is not None:
                results['files'][current_file]['seizures'].append({
                    'onset_sec': start_sec,
                    'offset_sec': end_sec,
                    'duration_sec': end_sec - start_sec
                })

        i += 1

    return results


def audit_chbmit(dataset_dir: str) -> Dict[str, Any]:
    """
    Perform a complete audit of the CHB-MIT dataset directory.
    Returns:
        - subject statistics
        - recording level metadata
        - seizure events inventory
        - channel inventory
        - missing/corrupted file list
    """
    dataset_dir = os.path.abspath(dataset_dir)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"CHB-MIT directory not found: {dataset_dir}")

    # Subject folders chb01 to chb24
    subject_dirs = [f"chb{i:02d}" for i in range(1, 25)]
    existing_subject_dirs = [d for d in subject_dirs if os.path.isdir(os.path.join(dataset_dir, d))]

    recordings = []
    seizure_events = []
    corrupted_files = []
    channel_counter = Counter()
    sampling_freq_counter = Counter()

    total_duration_sec = 0.0

    for subj in existing_subject_dirs:
        subj_dir = os.path.join(dataset_dir, subj)
        summary_path = os.path.join(subj_dir, f"{subj}-summary.txt")
        summary_info = parse_chbmit_summary(summary_path)

        # List all EDF files for this subject
        edf_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.edf')])

        for edf_fname in edf_files:
            edf_path = os.path.join(subj_dir, edf_fname)
            try:
                hdr = parse_edf_header(edf_path)
            except Exception as e:
                corrupted_files.append({
                    'dataset': 'CHB-MIT',
                    'subject_id': subj,
                    'filename': edf_fname,
                    'filepath': edf_path,
                    'error': str(e)
                })
                continue

            # Update counters
            total_duration_sec += hdr['duration_sec']
            sampling_freq_counter[hdr['sampling_frequency']] += 1
            for ch in hdr['channel_labels']:
                channel_counter[ch] += 1

            # Get seizure annotations for this file
            file_annotations = summary_info['files'].get(edf_fname, {})
            seizures_in_file = file_annotations.get('seizures', [])
            num_seizures = len(seizures_in_file)

            rec_info = {
                'dataset': 'CHB-MIT',
                'subject_id': subj,
                'filename': edf_fname,
                'filepath': edf_path,
                'num_channels': hdr['num_signals'],
                'sampling_frequency': hdr['sampling_frequency'],
                'duration_sec': hdr['duration_sec'],
                'duration_hours': hdr['duration_hours'],
                'start_time': hdr['start_time'],
                'num_seizures': num_seizures,
                'has_seizure': num_seizures > 0,
                'channels': hdr['channel_labels']
            }
            recordings.append(rec_info)

            # Record each seizure event
            for idx, sz in enumerate(seizures_in_file):
                seizure_events.append({
                    'dataset': 'CHB-MIT',
                    'subject_id': subj,
                    'filename': edf_fname,
                    'event_index': idx + 1,
                    'onset_sec': sz['onset_sec'],
                    'offset_sec': sz['offset_sec'],
                    'duration_sec': sz['duration_sec'],
                    'relative_onset_pct': (sz['onset_sec'] / hdr['duration_sec'] * 100.0) if hdr['duration_sec'] > 0 else 0.0
                })

    summary_stats = {
        'dataset_name': 'CHB-MIT',
        'dataset_path': dataset_dir,
        'num_subjects': len(existing_subject_dirs),
        'subject_list': existing_subject_dirs,
        'total_recordings': len(recordings) + len(corrupted_files),
        'usable_recordings': len(recordings),
        'corrupted_recordings': len(corrupted_files),
        'recordings_with_seizures': sum(1 for r in recordings if r['has_seizure']),
        'total_seizure_events': len(seizure_events),
        'total_duration_hours': total_duration_sec / 3600.0,
        'sampling_frequencies': dict(sampling_freq_counter),
        'unique_channels_count': len(channel_counter),
        'top_channels': channel_counter.most_common(25)
    }

    return {
        'summary': summary_stats,
        'recordings': recordings,
        'seizures': seizure_events,
        'corrupted': corrupted_files,
        'channel_counts': channel_counter
    }
