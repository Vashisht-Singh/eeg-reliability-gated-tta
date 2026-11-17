"""
Test Suite: Preprocessing Pipeline, Causal Filtering & Signal Quality Checks.
Mathematically verifies temporal causality, anti-aliasing resampling, and artifact detection.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessing import EEGPreprocessor


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.preprocessor = EEGPreprocessor(
            target_fs=256.0,
            low_cutoff=0.5,
            high_cutoff=45.0,
            notch_freq=60.0,
            causal=True
        )

    def test_resampling_ratio(self):
        """Verify 512 Hz -> 256 Hz decimation produces exactly half the samples."""
        x_512 = np.random.randn(18, 2048).astype(np.float32)  # 4 seconds @ 512 Hz
        x_256 = self.preprocessor.resample(x_512, orig_fs=512.0)
        self.assertEqual(x_256.shape, (18, 1024))

    def test_filter_temporal_causality(self):
        """
        MATHEMATICAL PROOF OF CAUSALITY:
        An impulse delta[n] located at index n_0 must produce ZERO response for n < n_0.
        Bidirectional filters leak response to n < n_0; causal filters do not.
        """
        n_samples = 512
        impulse_idx = 200
        x_impulse = np.zeros((1, n_samples), dtype=np.float32)
        x_impulse[0, impulse_idx] = 1.0

        y_causal = self.preprocessor.filter(x_impulse)

        # Output before impulse must be exactly zero (within float precision)
        pre_impulse_energy = np.sum(np.abs(y_causal[0, :impulse_idx]))
        self.assertAlmostEqual(pre_impulse_energy, 0.0, places=5,
                               msg="Filter violates causality: response detected before input impulse!")

    def test_robust_normalization(self):
        """Verify robust z-score normalization on synthetic signal."""
        # Mean 50, std 20 with severe outlier
        x = np.random.normal(loc=50.0, scale=20.0, size=(18, 1024)).astype(np.float32)
        x[:, 500] = 5000.0  # extreme artifact spike

        x_norm = self.preprocessor.normalize(x, clip_range=10.0)

        # Median should be ~0 and max clipped to +/- 10.0
        self.assertAlmostEqual(float(np.median(x_norm)), 0.0, delta=0.5)
        self.assertLessEqual(float(np.max(x_norm)), 10.0)
        self.assertGreaterEqual(float(np.min(x_norm)), -10.0)

    def test_artifact_detection(self):
        """Verify label-free artifact and flatline detection."""
        # Clean normal signal
        x_clean = np.random.normal(0, 30, size=(18, 1024)).astype(np.float32)
        is_val, metrics = self.preprocessor.check_signal_quality(x_clean)
        self.assertTrue(is_val)

        # Flatline signal (variance ~ 0)
        x_flatline = np.zeros((18, 1024), dtype=np.float32)
        is_val_fl, _ = self.preprocessor.check_signal_quality(x_flatline)
        self.assertFalse(is_val_fl, "Failed to flag flatline signal as invalid!")

        # Clipped / saturation artifact (> 500 uV)
        x_clipped = np.copy(x_clean)
        x_clipped[0, :] = 1200.0
        is_val_clip, _ = self.preprocessor.check_signal_quality(x_clipped)
        self.assertFalse(is_val_clip, "Failed to flag clipped amplifier artifact as invalid!")


if __name__ == '__main__':
    unittest.main()
