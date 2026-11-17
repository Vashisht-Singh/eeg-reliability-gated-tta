"""
Test Suite: Channel Mapping and 18-Channel Bipolar Montage Extraction.
Tests direct bipolar mapping, CS2 referential derivation, and Siena referential subtraction.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.channel_mapper import ChannelMapper
from src.data.edf_loader import IncrementalEDFLoader


class TestChannelMapping(unittest.TestCase):

    def test_direct_bipolar_chbmit(self):
        """Test standard CHB-MIT recording with native bipolar channels."""
        sample_file = r"F:\chb-mit-scalp-eeg-database-1.0.0\chb01\chb01_01.edf"
        loader = IncrementalEDFLoader(sample_file)
        mapper = ChannelMapper(loader.labels)

        self.assertEqual(len(mapper.TARGET_CHANNELS), 18)
        self.assertEqual(mapper.transformation_matrix.shape[0], 18)

        # Load window using required channel subset
        req_indices = mapper.get_required_channel_indices()
        raw_win = loader.read_window_sec(start_sec=10.0, duration_sec=4.0, channel_indices=req_indices)
        transformed = mapper.transform(raw_win)

        self.assertEqual(transformed.shape, (18, 1024))
        self.assertFalse(np.isnan(transformed).any())
        self.assertFalse(np.isinf(transformed).any())

    def test_referential_siena(self):
        """Test Siena recording with monopolar referential electrodes."""
        sample_file = r"D:\Sandhu_Siena_EEG\PN00\PN00-1.edf"
        loader = IncrementalEDFLoader(sample_file)
        mapper = ChannelMapper(loader.labels)

        self.assertEqual(len(mapper.TARGET_CHANNELS), 18)
        req_indices = mapper.get_required_channel_indices()
        raw_win = loader.read_window_sec(start_sec=10.0, duration_sec=4.0, channel_indices=req_indices)
        transformed = mapper.transform(raw_win)

        # 4.0 seconds @ 512 Hz = 2048 samples
        self.assertEqual(transformed.shape, (18, 2048))
        self.assertFalse(np.isnan(transformed).any())

    def test_cs2_referenced_chbmit(self):
        """Test atypical CHB-MIT file recorded with CS2 reference (chb12_27.edf)."""
        sample_file = r"F:\chb-mit-scalp-eeg-database-1.0.0\chb12\chb12_27.edf"
        loader = IncrementalEDFLoader(sample_file)
        mapper = ChannelMapper(loader.labels)

        self.assertEqual(len(mapper.TARGET_CHANNELS), 18)
        req_indices = mapper.get_required_channel_indices()
        raw_win = loader.read_window_sec(start_sec=0.0, duration_sec=4.0, channel_indices=req_indices)
        transformed = mapper.transform(raw_win)

        self.assertEqual(transformed.shape, (18, 1024))
        self.assertFalse(np.isnan(transformed).any())


if __name__ == '__main__':
    unittest.main()
