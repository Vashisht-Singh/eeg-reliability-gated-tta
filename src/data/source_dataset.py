"""
Source Domain Dataset Loader (CHB-MIT).
Provides PyTorch-compatible lazy window loading with on-the-fly montage projection,
causal filtering, and class balancing support.
"""

import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.data.edf_loader import IncrementalEDFLoader
from src.data.channel_mapper import ChannelMapper
from src.data.preprocessing import EEGPreprocessor


class CHBMITWindowDataset:
    """
    Lazy-loading dataset for CHB-MIT training and validation.
    Avoids loading full gigabyte recordings into RAM.
    """

    def __init__(
        self,
        window_manifest: List[Dict[str, Any]],
        preprocessor: Optional[EEGPreprocessor] = None,
        cache_in_memory: bool = False
    ):
        self.manifest = window_manifest
        self.preprocessor = preprocessor or EEGPreprocessor(target_fs=256.0, causal=True, notch_freq=60.0)
        self.cache_in_memory = cache_in_memory
        self.cache = {} if cache_in_memory else None

        # Active file caching
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

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float, str]:
        """
        Returns:
            x: preprocessed 18-channel array of shape (18, 1024)
            y: float label (1.0 = seizure, 0.0 = non-seizure)
            window_id: string identifier
        """
        if self.cache_in_memory and idx in self.cache:
            x, y, w_id = self.cache[idx]
            return x, y, w_id

        entry = self.manifest[idx]
        loader, mapper = self._get_loader(entry['filepath'])

        req_indices = mapper.get_required_channel_indices()
        raw_window = loader.read_window_sec(
            start_sec=float(entry['start_sec']),
            duration_sec=float(entry['duration_sec']),
            channel_indices=req_indices
        )

        bipolar_18 = mapper.transform(raw_window)
        orig_fs = loader.get_sampling_rate(req_indices[0])

        x_norm, is_valid, quality_metrics = self.preprocessor.process_window(bipolar_18, orig_fs=orig_fs)
        y = float(entry['label'])
        w_id = entry['window_id']

        if self.cache_in_memory:
            self.cache[idx] = (x_norm, y, w_id)

        return x_norm, y, w_id

    def get_labels(self) -> np.ndarray:
        """Return all window labels as a 1D array for class balancing calculation."""
        return np.array([float(entry['label']) for entry in self.manifest], dtype=np.float32)

    def get_class_weights(self) -> float:
        """
        Compute positive class weight for binary cross-entropy:
        pos_weight = N_neg / N_pos
        """
        labels = self.get_labels()
        n_pos = np.sum(labels == 1.0)
        n_neg = np.sum(labels == 0.0)
        if n_pos == 0:
            return 1.0
        return float(n_neg / n_pos)
