"""
Unit and Integration Tests for Phase 1 Data Integrity & Verification.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audit.chbmit_audit import parse_edf_header, parse_chbmit_summary
from src.audit.siena_audit import parse_time_str_to_seconds, resolve_siena_filename
from src.audit.channel_audit import DOUBLE_BANANA_CHAINS, ELECTRODE_SYNONYMS, normalize_electrode_name


class TestDataIntegrity(unittest.TestCase):

    def setUp(self):
        self.chb_path = r"F:\chb-mit-scalp-eeg-database-1.0.0"
        self.siena_path = r"D:\Sandhu_Siena_EEG"

    def test_paths_exist(self):
        """Verify both dataset root paths exist."""
        self.assertTrue(os.path.exists(self.chb_path), f"CHB-MIT path missing: {self.chb_path}")
        self.assertTrue(os.path.exists(self.siena_path), f"Siena path missing: {self.siena_path}")

    def test_chbmit_header_parsing(self):
        """Verify pure-Python EDF header parser correctly reads CHB-MIT files."""
        sample_file = os.path.join(self.chb_path, "chb01", "chb01_01.edf")
        self.assertTrue(os.path.exists(sample_file), f"Sample file not found: {sample_file}")
        hdr = parse_edf_header(sample_file)

        self.assertEqual(hdr['sampling_frequency'], 256.0)
        self.assertEqual(hdr['num_signals'], 23)
        self.assertEqual(hdr['duration_sec'], 3600.0)
        self.assertIn("FP1-F7", hdr['channel_labels'])

    def test_siena_header_parsing(self):
        """Verify pure-Python EDF header parser correctly reads Siena files."""
        sample_file = os.path.join(self.siena_path, "PN00", "PN00-1.edf")
        self.assertTrue(os.path.exists(sample_file), f"Sample file not found: {sample_file}")
        hdr = parse_edf_header(sample_file)

        self.assertEqual(hdr['sampling_frequency'], 512.0)
        self.assertGreater(hdr['num_signals'], 30)
        self.assertIn("EEG Fp1", hdr['channel_labels'])

    def test_chbmit_summary_parser(self):
        """Verify CHB-MIT summary file parser extracts seizure annotations."""
        summary_file = os.path.join(self.chb_path, "chb01", "chb01-summary.txt")
        self.assertTrue(os.path.exists(summary_file))
        res = parse_chbmit_summary(summary_file)

        self.assertEqual(res['sampling_rate'], 256)
        # chb01_03.edf has 1 seizure (onset 2996s, offset 3036s)
        self.assertIn("chb01_03.edf", res['files'])
        seizures = res['files']["chb01_03.edf"]['seizures']
        self.assertEqual(len(seizures), 1)
        self.assertEqual(seizures[0]['onset_sec'], 2996)
        self.assertEqual(seizures[0]['offset_sec'], 3036)

    def test_siena_time_parsing_and_rollover(self):
        """Verify time parsing and midnight rollover logic."""
        reg_start = parse_time_str_to_seconds("23.00.00")
        sz_start = parse_time_str_to_seconds("01.00.00")
        self.assertEqual(reg_start, 23 * 3600)
        self.assertEqual(sz_start, 1 * 3600)

        # Check relative calculation with overnight rollover
        rel_onset = sz_start - reg_start
        if rel_onset < 0:
            rel_onset += 24 * 3600
        self.assertEqual(rel_onset, 2 * 3600)

    def test_siena_filename_resolution(self):
        """Verify handling of typographical discrepancies in Siena filenames."""
        avail = ["PN06-1.edf", "PN06-2.edf"]
        resolved = resolve_siena_filename("PNO6-1.edf", avail)
        self.assertEqual(resolved, "PN06-1.edf")

        avail_single = ["PN01-1.edf"]
        resolved_single = resolve_siena_filename("PN01.edf", avail_single)
        self.assertEqual(resolved_single, "PN01-1.edf")

    def test_double_banana_montage_definition(self):
        """Verify that double banana montage contains exactly 18 standard channels."""
        total_channels = sum(len(chs) for chs in DOUBLE_BANANA_CHAINS.values())
        self.assertEqual(total_channels, 18)


if __name__ == '__main__':
    unittest.main()
