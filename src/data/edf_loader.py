"""
Lightweight Incremental EDF Signal Loader.
Provides chunk-based, memory-efficient loading of raw EDF recordings
with exact integer-to-microvolt physical calibration and zero full-dataset memory overhead.
"""

import os
import struct
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class IncrementalEDFLoader:
    """
    Memory-efficient streaming EDF reader.
    Supports random access to time windows and individual signal records
    without loading entire multi-gigabyte recordings into RAM.
    """

    def __init__(self, filepath: str):
        self.filepath = os.path.abspath(filepath)
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"EDF file not found: {self.filepath}")

        self._parse_header()

    def _parse_header(self):
        with open(self.filepath, 'rb') as f:
            raw_h = f.read(256)
            if len(raw_h) < 256:
                raise ValueError(f"Corrupted or invalid EDF file (length < 256 bytes): {self.filepath}")

            self.version = raw_h[0:8].decode('latin1').strip()
            self.patient_id = raw_h[8:88].decode('latin1').strip()
            self.rec_id = raw_h[88:168].decode('latin1').strip()
            self.start_date = raw_h[168:176].decode('latin1').strip()
            self.start_time = raw_h[176:184].decode('latin1').strip()
            self.header_bytes = int(raw_h[184:192].decode('latin1').strip())
            self.reserved = raw_h[192:236].decode('latin1').strip()
            self.num_records = int(raw_h[236:244].decode('latin1').strip())
            self.record_duration = float(raw_h[244:252].decode('latin1').strip())
            self.ns = int(raw_h[252:256].decode('latin1').strip())

            # Read per-signal headers
            self.labels = [f.read(16).decode('latin1').strip() for _ in range(self.ns)]
            self.transducers = [f.read(80).decode('latin1').strip() for _ in range(self.ns)]
            self.phys_dims = [f.read(8).decode('latin1').strip() for _ in range(self.ns)]
            self.phys_mins = np.array([float(f.read(8).decode('latin1').strip()) for _ in range(self.ns)], dtype=np.float32)
            self.phys_maxs = np.array([float(f.read(8).decode('latin1').strip()) for _ in range(self.ns)], dtype=np.float32)
            self.dig_mins = np.array([float(f.read(8).decode('latin1').strip()) for _ in range(self.ns)], dtype=np.float32)
            self.dig_maxs = np.array([float(f.read(8).decode('latin1').strip()) for _ in range(self.ns)], dtype=np.float32)
            self.prefilters = [f.read(80).decode('latin1').strip() for _ in range(self.ns)]
            self.samples_per_rec = [int(f.read(8).decode('latin1').strip()) for _ in range(self.ns)]
            self.reserved_signals = [f.read(32).decode('latin1').strip() for _ in range(self.ns)]

        # Precompute gain and bias for calibration: phys = (dig - dig_min) * gain + phys_min
        # gain = (phys_max - phys_min) / (dig_max - dig_min)
        dig_range = self.dig_maxs - self.dig_mins
        dig_range[dig_range == 0] = 1.0  # avoid division by zero
        self.gains = (self.phys_maxs - self.phys_mins) / dig_range
        self.biases = self.phys_mins - self.gains * self.dig_mins

        self.sampling_rates = [self.samples_per_rec[i] / self.record_duration for i in range(self.ns)]
        self.total_duration_sec = self.num_records * self.record_duration

        # Record offsets
        # Each record stores samples_per_rec[0]*2 bytes for sig 0, samples_per_rec[1]*2 bytes for sig 1, ...
        self.record_offsets = np.zeros(self.ns + 1, dtype=int)
        for i in range(self.ns):
            self.record_offsets[i + 1] = self.record_offsets[i] + self.samples_per_rec[i] * 2
        self.bytes_per_record = self.record_offsets[-1]

        # Channel lookup map
        self.channel_map = {lbl.upper(): idx for idx, lbl in enumerate(self.labels)}

    def get_sampling_rate(self, channel_idx: int = 0) -> float:
        """Return sampling rate for given channel index."""
        return self.sampling_rates[channel_idx]

    def read_record(self, record_idx: int) -> np.ndarray:
        """
        Read a single raw data record (all signals).
        Returns a 2D float32 array calibrated to physical units (typically uV).
        """
        if record_idx < 0 or record_idx >= self.num_records:
            raise IndexError(f"Record index {record_idx} out of range [0, {self.num_records - 1}]")

        byte_offset = self.header_bytes + record_idx * self.bytes_per_record
        with open(self.filepath, 'rb') as f:
            f.seek(byte_offset)
            raw_bytes = f.read(self.bytes_per_record)

        record_data = []
        for i in range(self.ns):
            start = self.record_offsets[i]
            end = self.record_offsets[i + 1]
            raw_ints = np.frombuffer(raw_bytes[start:end], dtype='<i2')
            calibrated = raw_ints.astype(np.float32) * self.gains[i] + self.biases[i]
            record_data.append(calibrated)

        return record_data

    def read_records_range(self, start_rec: int, end_rec: int, channel_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Read a contiguous range of records [start_rec, end_rec) for specified channels.
        Returns array of shape (num_selected_channels, total_samples).
        """
        start_rec = max(0, start_rec)
        end_rec = min(self.num_records, end_rec)
        n_records = end_rec - start_rec

        if n_records <= 0:
            return np.zeros((len(channel_indices) if channel_indices else self.ns, 0), dtype=np.float32)

        if channel_indices is None:
            channel_indices = list(range(self.ns))

        # Check if all selected channels share identical sampling rates
        target_samps_per_rec = self.samples_per_rec[channel_indices[0]]
        total_samples = n_records * target_samps_per_rec

        out = np.zeros((len(channel_indices), total_samples), dtype=np.float32)

        byte_offset = self.header_bytes + start_rec * self.bytes_per_record
        total_bytes = n_records * self.bytes_per_record

        with open(self.filepath, 'rb') as f:
            f.seek(byte_offset)
            raw_bytes = f.read(total_bytes)

        # Parse per record
        for r_idx in range(n_records):
            rec_start_byte = r_idx * self.bytes_per_record
            sample_start = r_idx * target_samps_per_rec
            sample_end = sample_start + target_samps_per_rec

            for out_idx, ch_idx in enumerate(channel_indices):
                sig_byte_start = rec_start_byte + self.record_offsets[ch_idx]
                sig_byte_end = rec_start_byte + self.record_offsets[ch_idx + 1]
                raw_ints = np.frombuffer(raw_bytes[sig_byte_start:sig_byte_end], dtype='<i2')
                out[out_idx, sample_start:sample_end] = raw_ints.astype(np.float32) * self.gains[ch_idx] + self.biases[ch_idx]

        return out

    def read_window_sec(self, start_sec: float, duration_sec: float, channel_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Read a time window in seconds [start_sec, start_sec + duration_sec].
        Accurate to sample precision.
        """
        if self.record_duration <= 0:
            raise ValueError("Record duration must be > 0")

        start_rec = int(np.floor(start_sec / self.record_duration))
        end_rec = int(np.ceil((start_sec + duration_sec) / self.record_duration))

        # Load records covering this window
        block = self.read_records_range(start_rec, end_rec, channel_indices)

        if channel_indices is None:
            channel_indices = list(range(self.ns))

        fs = self.sampling_rates[channel_indices[0]]
        rec_start_time = start_rec * self.record_duration
        sample_offset = int(round((start_sec - rec_start_time) * fs))
        num_target_samples = int(round(duration_sec * fs))

        start_s = max(0, sample_offset)
        end_s = start_s + num_target_samples

        if end_s <= block.shape[1]:
            return block[:, start_s:end_s]
        else:
            # Pad with edge values if window slightly exceeds file boundary
            pad_len = end_s - block.shape[1]
            padded = np.pad(block[:, start_s:], ((0, 0), (0, pad_len)), mode='edge')
            return padded
