"""
EEG Window Generator and Indexing Engine.
Generates 4.0-second EEG windows (1024 samples @ 256 Hz) with class-balanced strides
for source training and strict chronological progression for target test-time adaptation.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple


class WindowGenerator:
    """
    Computes window start and end timestamps, class labels, and metadata indices.
    """

    def __init__(
        self,
        window_duration_sec: float = 4.0,
        target_fs: float = 256.0,
        source_seizure_stride_sec: float = 1.0,
        source_bg_stride_sec: float = 4.0,
        target_stream_stride_sec: float = 1.0,
        seizure_overlap_threshold: float = 0.5
    ):
        self.window_duration_sec = window_duration_sec
        self.target_fs = target_fs
        self.source_seizure_stride_sec = source_seizure_stride_sec
        self.source_bg_stride_sec = source_bg_stride_sec
        self.target_stream_stride_sec = target_stream_stride_sec
        self.seizure_overlap_threshold = seizure_overlap_threshold
        self.expected_samples = int(round(self.window_duration_sec * self.target_fs))

    def generate_source_windows_for_recording(
        self,
        subject_id: str,
        filename: str,
        filepath: str,
        total_duration_sec: float,
        seizures: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Generates labeled windows for CHB-MIT source recording.
        Uses 1.0s stride (75% overlap) for seizure regions and 4.0s stride for background.
        """
        windows = []
        if total_duration_sec < self.window_duration_sec:
            return windows

        # Precompute seizure intervals
        seizure_intervals = [(sz['onset_sec'], sz['offset_sec']) for sz in seizures]

        t = 0.0
        window_idx = 0

        while t + self.window_duration_sec <= total_duration_sec:
            w_start = t
            w_end = t + self.window_duration_sec

            # Check overlap with any seizure interval
            is_seizure = False
            max_overlap = 0.0
            for s_start, s_end in seizure_intervals:
                overlap = max(0.0, min(w_end, s_end) - max(w_start, s_start))
                if overlap > max_overlap:
                    max_overlap = overlap

            if (max_overlap / self.window_duration_sec) >= self.seizure_overlap_threshold:
                is_seizure = True

            windows.append({
                'window_id': f"{subject_id}_{filename}_w{window_idx:06d}",
                'dataset': 'CHB-MIT',
                'subject_id': subject_id,
                'filename': filename,
                'filepath': filepath,
                'start_sec': round(w_start, 3),
                'end_sec': round(w_end, 3),
                'duration_sec': self.window_duration_sec,
                'label': 1 if is_seizure else 0,
                'is_seizure': is_seizure
            })
            window_idx += 1

            # Adaptive stride: dense 1.0s stride near seizures, 4.0s for background
            in_seizure_vicinity = False
            for s_start, s_end in seizure_intervals:
                # 30 seconds buffer around seizure
                if (s_start - 30.0) <= w_start <= (s_end + 30.0):
                    in_seizure_vicinity = True
                    break

            if in_seizure_vicinity:
                t += self.source_seizure_stride_sec
            else:
                t += self.source_bg_stride_sec

        return windows

    def generate_target_stream_windows_for_recording(
        self,
        subject_id: str,
        filename: str,
        filepath: str,
        total_duration_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Generates strict chronological streaming windows for Siena target recording.
        IMPORTANT: NO TARGET SEIZURE LABELS ARE LOADED OR ATTACHED.
        """
        windows = []
        if total_duration_sec < self.window_duration_sec:
            return windows

        t = 0.0
        window_idx = 0

        while t + self.window_duration_sec <= total_duration_sec:
            w_start = t
            w_end = t + self.window_duration_sec

            windows.append({
                'window_id': f"{subject_id}_{filename}_tgt{window_idx:06d}",
                'dataset': 'Siena',
                'subject_id': subject_id,
                'filename': filename,
                'filepath': filepath,
                'start_sec': round(w_start, 3),
                'end_sec': round(w_end, 3),
                'duration_sec': self.window_duration_sec,
                # Explicitly unlabeled for online adaptation
                'label': None
            })
            window_idx += 1
            t += self.target_stream_stride_sec

        return windows
