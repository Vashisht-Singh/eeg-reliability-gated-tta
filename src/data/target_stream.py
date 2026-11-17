"""
Target Domain Chronological EEG Stream Generator.
Provides strictly chronological, unlabeled window-by-window streaming of Siena Scalp EEG
for test-time adaptation.

MANDATORY DATA LEAKAGE CONTROLS:
1. No access to ground-truth seizure labels.
2. Strict chronological order (no shuffling, no future lookahead).
3. Real-time timestamp logging for every yielded window.
"""

import os
from typing import Dict, Any, Generator, Optional, List, Tuple
import numpy as np

from src.data.edf_loader import IncrementalEDFLoader
from src.data.channel_mapper import ChannelMapper
from src.data.preprocessing import EEGPreprocessor


class SienaTargetStream:
    """
    Chronological generator of preprocessed 18-channel EEG windows for Siena.
    Enforces total label isolation.
    """

    def __init__(
        self,
        stream_manifest: List[Dict[str, Any]],
        preprocessor: Optional[EEGPreprocessor] = None
    ):
        """
        Args:
            stream_manifest: List of window index dicts (strictly chronological).
            preprocessor: EEGPreprocessor instance (causal filtering + resampling).
        """
        self.manifest = stream_manifest
        self.preprocessor = preprocessor or EEGPreprocessor(target_fs=256.0, causal=True, notch_freq=50.0)

        # Cache active EDF loader to avoid reopening the file repeatedly
        self._current_filepath = None
        self._current_loader = None
        self._current_mapper = None

    def __len__(self) -> int:
        return len(self.manifest)

    def _get_loader(self, filepath: str) -> Tuple[IncrementalEDFLoader, ChannelMapper]:
        if filepath != self._current_filepath:
            self._current_filepath = filepath
            self._current_loader = IncrementalEDFLoader(filepath)
            self._current_mapper = ChannelMapper(self._current_loader.labels)
        return self._current_loader, self._current_mapper

    def stream_windows(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yields chronological windows one by one.
        Yielded dictionary contains:
          - 'x': preprocessed 18-channel array of shape (18, 1024)
          - 'window_id': unique identifier
          - 'subject_id': patient identifier
          - 'filename': recording name
          - 'start_sec': window start timestamp
          - 'end_sec': window end timestamp
          - 'is_valid_quality': boolean quality indicator
          - 'quality_metrics': dictionary of artifact metrics
          NOTE: Absolutely NO ground truth labels are yielded.
        """
        for entry in self.manifest:
            loader, mapper = self._get_loader(entry['filepath'])

            # 1. Read raw channels needed for the 18-channel double banana
            req_indices = mapper.get_required_channel_indices()
            raw_window = loader.read_window_sec(
                start_sec=float(entry['start_sec']),
                duration_sec=float(entry['duration_sec']),
                channel_indices=req_indices
            )

            # 2. Project to standard 18-channel double banana montage
            bipolar_18 = mapper.transform(raw_window)

            # 3. Preprocess: 512 Hz -> 256 Hz resampling, causal filtering, normalization
            orig_fs = loader.get_sampling_rate(req_indices[0])
            x_norm, is_valid, quality_metrics = self.preprocessor.process_window(bipolar_18, orig_fs=orig_fs)

            yield {
                'x': x_norm,
                'window_id': entry['window_id'],
                'dataset': 'Siena',
                'subject_id': entry['subject_id'],
                'filename': entry['filename'],
                'start_sec': entry['start_sec'],
                'end_sec': entry['end_sec'],
                'is_valid_quality': is_valid,
                'quality_metrics': quality_metrics
            }
