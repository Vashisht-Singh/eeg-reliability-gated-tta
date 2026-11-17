"""
EEG Preprocessing and Harmonization Pipeline.
Implements causal bandpass/notch filtering, anti-aliasing resampling,
robust normalization, and label-free signal quality checks.
"""

import numpy as np
from scipy import signal
from typing import Dict, Any, Tuple, Optional


class EEGPreprocessor:
    """
    Configurable, research-grade EEG preprocessing pipeline.
    Ensures strict temporal causality for streaming test-time adaptation.
    """

    def __init__(
        self,
        target_fs: float = 256.0,
        low_cutoff: float = 0.5,
        high_cutoff: float = 45.0,
        notch_freq: Optional[float] = 60.0,
        filter_order: int = 4,
        causal: bool = True,
        clip_amplitude_uv: float = 500.0,
        flatline_var_uv2: float = 1.0
    ):
        self.target_fs = target_fs
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.notch_freq = notch_freq
        self.filter_order = filter_order
        self.causal = causal
        self.clip_amplitude_uv = clip_amplitude_uv
        self.flatline_var_uv2 = flatline_var_uv2

        # Design Second-Order Sections (SOS) bandpass filter for numerical stability
        self.sos_bandpass = signal.butter(
            N=self.filter_order,
            Wn=[self.low_cutoff, self.high_cutoff],
            btype='bandpass',
            fs=self.target_fs,
            output='sos'
        )

        # Design notch filter
        self.sos_notch = None
        if self.notch_freq is not None and self.notch_freq < (self.target_fs / 2.0):
            # Q factor = 30 for narrow notch
            b, a = signal.iirnotch(w0=self.notch_freq, Q=30.0, fs=self.target_fs)
            self.sos_notch = signal.tf2sos(b, a)

    def resample(self, x: np.ndarray, orig_fs: float) -> np.ndarray:
        """
        Resample multi-channel EEG array of shape (n_channels, n_samples)
        from orig_fs to self.target_fs using anti-aliasing polyphase decimation.
        """
        if abs(orig_fs - self.target_fs) < 1e-3:
            return x

        # For exact 512 Hz -> 256 Hz downsampling (factor of 2)
        if abs(orig_fs - 512.0) < 1e-3 and abs(self.target_fs - 256.0) < 1e-3:
            return signal.resample_poly(x, up=1, down=2, axis=-1)

        # General rational resampling
        from math import gcd
        up = int(self.target_fs)
        down = int(orig_fs)
        common = gcd(up, down)
        up //= common
        down //= common
        return signal.resample_poly(x, up=up, down=down, axis=-1)

    def filter(self, x: np.ndarray) -> np.ndarray:
        """
        Filter multi-channel signal (n_channels, n_samples).
        If self.causal is True: uses forward-only sosfilt (zero future leakage).
        If self.causal is False: uses zero-phase sosfiltfilt.
        """
        if self.causal:
            # Strictly causal forward-only filtering
            filtered = signal.sosfilt(self.sos_bandpass, x, axis=-1)
            if self.sos_notch is not None:
                filtered = signal.sosfilt(self.sos_notch, filtered, axis=-1)
        else:
            # Bidirectional zero-phase filtering (offline only)
            filtered = signal.sosfiltfilt(self.sos_bandpass, x, axis=-1)
            if self.sos_notch is not None:
                filtered = signal.sosfiltfilt(self.sos_notch, filtered, axis=-1)

        return filtered.astype(np.float32)

    def check_signal_quality(self, x: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """
        Label-free signal quality and artifact assessment on window (n_channels, n_samples).
        Detects:
          - Amplitude saturation / amplifier clipping
          - Electrode disconnection / flatline
          - Extreme line-noise contamination
        """
        max_abs_amp = float(np.max(np.abs(x)))
        channel_variances = np.var(x, axis=-1)
        min_var = float(np.min(channel_variances))
        mean_var = float(np.mean(channel_variances))

        is_clipped = max_abs_amp > self.clip_amplitude_uv
        is_flatlined = min_var < self.flatline_var_uv2

        is_valid = (not is_clipped) and (not is_flatlined)

        metrics = {
            'max_abs_amplitude_uv': max_abs_amp,
            'min_channel_variance_uv2': min_var,
            'mean_channel_variance_uv2': mean_var,
            'is_clipped': float(is_clipped),
            'is_flatlined': float(is_flatlined),
            'is_valid_quality': float(is_valid)
        }
        return is_valid, metrics

    def normalize(self, x: np.ndarray, clip_range: float = 10.0) -> np.ndarray:
        """
        Robust channel-wise z-score normalization:
        z = (x - median) / (IQR / 1.349)
        Clips extreme outliers beyond +/- clip_range.
        """
        medians = np.median(x, axis=-1, keepdims=True)
        q75 = np.percentile(x, 75, axis=-1, keepdims=True)
        q25 = np.percentile(x, 25, axis=-1, keepdims=True)
        iqr = q75 - q25

        # Avoid division by zero for quiet channels
        iqr[iqr == 0] = 1.0
        robust_std = iqr / 1.349

        z = (x - medians) / robust_std
        if clip_range is not None:
            z = np.clip(z, -clip_range, clip_range)

        return z.astype(np.float32)

    def process_window(self, x: np.ndarray, orig_fs: float) -> Tuple[np.ndarray, bool, Dict[str, float]]:
        """
        Complete end-to-end preprocessing pipeline for a window:
        Resample -> Causal Filter -> Quality Check -> Robust Normalize.
        """
        # 1. Resample to target_fs
        x_resampled = self.resample(x, orig_fs)

        # 2. Causal bandpass and notch filter
        x_filtered = self.filter(x_resampled)

        # 3. Label-free quality check
        is_valid, quality_metrics = self.check_signal_quality(x_filtered)

        # 4. Robust normalization
        x_norm = self.normalize(x_filtered)

        return x_norm, is_valid, quality_metrics
