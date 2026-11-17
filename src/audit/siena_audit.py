"""
Siena Scalp EEG Dataset Audit Module.
Scans and audits raw EDF recordings, seizure annotation text files,
and patient metadata for the Siena database without modifying any source files.
"""

import os
import re
import csv
from typing import Dict, List, Any, Optional
from collections import Counter
from src.audit.chbmit_audit import parse_edf_header


def parse_time_str_to_seconds(time_str: str) -> Optional[float]:
    """
    Parse a time string formatted as HH.MM.SS or HH:MM:SS or HH.MM.SS.mmm
    into seconds from midnight. Robust against spaces and punctuation.
    """
    if not time_str:
        return None
    # Strip annotations like (CLINICAL ONSET) or opure ...
    clean_str = re.split(r'[\(\;]', time_str)[0].strip()
    # Remove internal spaces like '1 6.49.25' -> '16.49.25'
    clean_str = re.sub(r'(?<=\d)\s+(?=\d)', '', clean_str)

    # Match hours, minutes, seconds
    m = re.search(r'(\d{1,2})[:.](\d{1,2})[:.](\d{1,2}(?:\.\d+)?)', clean_str)
    if not m:
        return None
    h, m_val, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600.0 + m_val * 60.0 + s


def parse_siena_subject_info(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse Siena subject_info.csv containing demographics and clinical metadata.
    """
    subj_info = {}
    if not os.path.exists(csv_path):
        return subj_info

    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get('patient_id', '').strip()
            if pid:
                subj_info[pid] = {k.strip(): v.strip() for k, v in row.items()}
    return subj_info


def resolve_siena_filename(raw_fname: str, available_edfs: List[str]) -> Optional[str]:
    """
    Resolve typographical inconsistencies in Siena annotation files to actual disk filenames.
    E.g.:
      - 'PNO6-1.edf' -> 'PN06-1.edf' (letter O to zero 0)
      - 'PN01.edf' -> 'PN01-1.edf'
      - 'PN11-.edf' -> 'PN11-1.edf'
    """
    raw_fname = raw_fname.strip()
    if raw_fname in available_edfs:
        return raw_fname

    # Try replacing letter O with zero
    fixed = re.sub(r'PNO(\d)', r'PN0\1', raw_fname)
    if fixed in available_edfs:
        return fixed

    # Try matching by subject and number
    m = re.search(r'(PN\d+)[-_]?(\d+)?', raw_fname, re.IGNORECASE)
    if m:
        subj = m.group(1).upper().replace('PNO', 'PN0')
        idx = m.group(2)
        for avail in available_edfs:
            if avail.upper().startswith(subj):
                if idx and idx in avail:
                    return avail
                elif not idx and len(available_edfs) == 1:
                    return avail

    # Case-insensitive match
    for avail in available_edfs:
        if avail.lower() == raw_fname.lower():
            return avail

    return None


def parse_siena_seizure_list(
    list_path: str,
    subject_id: str,
    available_edfs: List[str],
    edf_headers: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Parse Siena Seizures-list-PNXX.txt file.
    Extracts seizure events with registration start time, seizure onset and offset.
    Computes exact relative onset and offset in seconds from EDF recording start.
    """
    if not os.path.exists(list_path):
        return []

    with open(list_path, 'r', encoding='latin1', errors='replace') as f:
        content = f.read()

    # Split into seizure blocks or lines
    lines = content.splitlines()
    seizures = []

    current_file = None
    reg_start_str = None
    reg_start_sec = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check for file name
        if line.lower().startswith('file name:') or line.lower().startswith('file:'):
            raw_fname = line.split(':', 1)[1].strip()
            resolved = resolve_siena_filename(raw_fname, available_edfs)
            if resolved:
                current_file = resolved
                # If we have the EDF header, its start_time is an authoritative backup
                if current_file in edf_headers:
                    hdr_time = edf_headers[current_file].get('start_time')
                    if hdr_time:
                        reg_start_sec = parse_time_str_to_seconds(hdr_time)

        # Check for registration start time
        elif 'registration start time:' in line.lower() or 'start registration:' in line.lower():
            raw_time = line.split(':', 1)[1].strip()
            parsed_sec = parse_time_str_to_seconds(raw_time)
            if parsed_sec is not None:
                reg_start_sec = parsed_sec
                reg_start_str = raw_time

        # Check for seizure definition
        elif 'seizure' in line.lower() and ('start time:' in line.lower() or line.lower().startswith('seizure n') or line.lower().startswith('seizure:')):
            # Look for start time and end time within the next 8 lines
            sz_start_sec = None
            sz_end_sec = None
            sz_file = current_file

            for j in range(i, min(i + 9, len(lines))):
                sub_line = lines[j].strip()
                if sub_line.lower().startswith('file name:'):
                    raw_fn = sub_line.split(':', 1)[1].strip()
                    res_fn = resolve_siena_filename(raw_fn, available_edfs)
                    if res_fn:
                        sz_file = res_fn
                        if sz_file in edf_headers and reg_start_sec is None:
                            reg_start_sec = parse_time_str_to_seconds(edf_headers[sz_file]['start_time'])

                elif 'registration start time:' in sub_line.lower():
                    raw_t = sub_line.split(':', 1)[1].strip()
                    p_t = parse_time_str_to_seconds(raw_t)
                    if p_t is not None:
                        reg_start_sec = p_t

                elif ('seizure start time:' in sub_line.lower() or
                      (sub_line.lower().startswith('start time:') and 'registration' not in sub_line.lower())):
                    raw_t = sub_line.split(':', 1)[1].strip()
                    sz_start_sec = parse_time_str_to_seconds(raw_t)

                elif ('seizure end time:' in sub_line.lower() or
                      (sub_line.lower().startswith('end time:') and 'registration' not in sub_line.lower())):
                    raw_t = sub_line.split(':', 1)[1].strip()
                    sz_end_sec = parse_time_str_to_seconds(raw_t)

            if sz_file and sz_start_sec is not None and sz_end_sec is not None:
                # If registration start was not explicitly found in text, use EDF header start time
                if reg_start_sec is None and sz_file in edf_headers:
                    reg_start_sec = parse_time_str_to_seconds(edf_headers[sz_file]['start_time'])

                if reg_start_sec is not None:
                    # Calculate relative onset and offset from recording start
                    rel_onset = sz_start_sec - reg_start_sec
                    if rel_onset < -3600.0:  # Overnight rollover
                        rel_onset += 24 * 3600.0

                    rel_offset = sz_end_sec - reg_start_sec
                    if rel_offset < -3600.0:  # Overnight rollover
                        rel_offset += 24 * 3600.0

                    # If offset is before onset due to midnight rollover
                    if rel_offset < rel_onset:
                        rel_offset += 24 * 3600.0

                    duration = rel_offset - rel_onset

                    # Check against edf recording duration
                    rec_dur = edf_headers.get(sz_file, {}).get('duration_sec', 0.0)

                    seizures.append({
                        'dataset': 'Siena',
                        'subject_id': subject_id,
                        'filename': sz_file,
                        'onset_sec': max(0.0, rel_onset),
                        'offset_sec': max(0.0, rel_offset),
                        'duration_sec': max(0.0, duration),
                        'clock_start_sec': sz_start_sec,
                        'clock_end_sec': sz_end_sec,
                        'recording_duration_sec': rec_dur
                    })

        i += 1

    # Remove duplicates if any were parsed twice
    unique_seizures = []
    seen = set()
    for sz in seizures:
        key = (sz['filename'], round(sz['onset_sec'], 1), round(sz['offset_sec'], 1))
        if key not in seen:
            seen.add(key)
            sz['event_index'] = len(unique_seizures) + 1
            unique_seizures.append(sz)

    return unique_seizures


def audit_siena(dataset_dir: str) -> Dict[str, Any]:
    """
    Perform a complete audit of the Siena Scalp EEG dataset directory.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Siena directory not found: {dataset_dir}")

    # Load subject info
    subject_info_map = parse_siena_subject_info(os.path.join(dataset_dir, 'subject_info.csv'))

    # Subject folders
    subject_dirs = sorted([d for d in os.listdir(dataset_dir)
                          if os.path.isdir(os.path.join(dataset_dir, d)) and d.upper().startswith('PN')])

    recordings = []
    all_seizures = []
    corrupted_files = []
    channel_counter = Counter()
    sampling_freq_counter = Counter()

    total_duration_sec = 0.0

    for subj in subject_dirs:
        subj_dir = os.path.join(dataset_dir, subj)
        edf_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.edf')])

        # Parse EDF headers first
        subj_edf_headers = {}
        for edf_fname in edf_files:
            edf_path = os.path.join(subj_dir, edf_fname)
            try:
                hdr = parse_edf_header(edf_path)
                subj_edf_headers[edf_fname] = hdr
            except Exception as e:
                corrupted_files.append({
                    'dataset': 'Siena',
                    'subject_id': subj,
                    'filename': edf_fname,
                    'filepath': edf_path,
                    'error': str(e)
                })

        # Parse seizure list
        list_path = os.path.join(subj_dir, f"Seizures-list-{subj}.txt")
        subj_seizures = parse_siena_seizure_list(list_path, subj, edf_files, subj_edf_headers)
        all_seizures.extend(subj_seizures)

        # Build recording records
        for edf_fname, hdr in subj_edf_headers.items():
            total_duration_sec += hdr['duration_sec']
            sampling_freq_counter[hdr['sampling_frequency']] += 1
            for ch in hdr['channel_labels']:
                channel_counter[ch] += 1

            file_seizures = [sz for sz in subj_seizures if sz['filename'] == edf_fname]
            num_seizures = len(file_seizures)

            rec_info = {
                'dataset': 'Siena',
                'subject_id': subj,
                'filename': edf_fname,
                'filepath': hdr['filepath'],
                'num_channels': hdr['num_signals'],
                'sampling_frequency': hdr['sampling_frequency'],
                'duration_sec': hdr['duration_sec'],
                'duration_hours': hdr['duration_hours'],
                'start_time': hdr['start_time'],
                'num_seizures': num_seizures,
                'has_seizure': num_seizures > 0,
                'channels': hdr['channel_labels'],
                'clinical_info': subject_info_map.get(subj, {})
            }
            recordings.append(rec_info)

    summary_stats = {
        'dataset_name': 'Siena Scalp EEG',
        'dataset_path': dataset_dir,
        'num_subjects': len(subject_dirs),
        'subject_list': subject_dirs,
        'total_recordings': len(recordings) + len(corrupted_files),
        'usable_recordings': len(recordings),
        'corrupted_recordings': len(corrupted_files),
        'recordings_with_seizures': sum(1 for r in recordings if r['has_seizure']),
        'total_seizure_events': len(all_seizures),
        'total_duration_hours': total_duration_sec / 3600.0,
        'sampling_frequencies': dict(sampling_freq_counter),
        'unique_channels_count': len(channel_counter),
        'top_channels': channel_counter.most_common(25)
    }

    return {
        'summary': summary_stats,
        'recordings': recordings,
        'seizures': all_seizures,
        'corrupted': corrupted_files,
        'channel_counts': channel_counter,
        'subject_info': subject_info_map
    }
