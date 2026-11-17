"""
Test Suite: Strict Target-Domain Label Isolation and Chronological Progression.
Verifies that the target streaming adapter cannot access or leak target seizure annotations.
"""

import os
import sys
import csv
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.target_stream import SienaTargetStream


class TestLabelIsolation(unittest.TestCase):

    def test_target_manifest_has_no_labels(self):
        """Verify target streaming manifest contains strictly None/empty labels."""
        manifest_path = os.path.join(PROJECT_ROOT, "data", "manifests", "target_stream_manifest.csv")
        self.assertTrue(os.path.exists(manifest_path))

        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                lbl = row.get('label', '')
                self.assertIn(lbl, ['', 'None', None],
                              f"FATAL LEAKAGE: Target window {row['window_id']} has label '{lbl}'!")
                if idx > 1000:  # Sample check first 1000 rows
                    break

    def test_target_stream_generator_yields_unlabeled_windows(self):
        """Verify SienaTargetStream yields windows with zero label information."""
        smoke_manifest_path = os.path.join(PROJECT_ROOT, "data", "manifests", "target_smoke_test_manifest.csv")
        self.assertTrue(os.path.exists(smoke_manifest_path))

        with open(smoke_manifest_path, 'r', encoding='utf-8') as f:
            entries = list(csv.DictReader(f))

        stream = SienaTargetStream(entries[:5])  # Test first 5 windows
        for win in stream.stream_windows():
            self.assertNotIn('label', win, "Target stream window contains 'label' key!")
            self.assertNotIn('is_seizure', win, "Target stream window contains 'is_seizure' key!")
            self.assertEqual(win['x'].shape, (18, 1024), "Preprocessed target window shape must be (18, 1024)")

    def test_strict_chronological_ordering(self):
        """Verify target stream windows advance strictly chronologically without shuffling."""
        smoke_manifest_path = os.path.join(PROJECT_ROOT, "data", "manifests", "target_smoke_test_manifest.csv")
        with open(smoke_manifest_path, 'r', encoding='utf-8') as f:
            entries = list(csv.DictReader(f))

        prev_time = -1.0
        for entry in entries:
            curr_time = float(entry['start_sec'])
            self.assertGreaterEqual(curr_time, prev_time,
                                    f"Target stream chronological violation: {curr_time} < {prev_time}")
            prev_time = curr_time

    def test_adaptation_code_has_no_annotation_imports(self):
        """Static analysis check: target_stream.py must not import evaluation annotations."""
        target_stream_py = os.path.join(PROJECT_ROOT, "src", "data", "target_stream.py")
        with open(target_stream_py, 'r', encoding='utf-8') as f:
            code = f.read()

        prohibited = ['annotation_parser', 'siena_ground_truth', 'load_offline_target_annotations']
        for term in prohibited:
            self.assertNotIn(term, code,
                             f"SECURITY VIOLATION: '{term}' referenced in target_stream.py!")


if __name__ == '__main__':
    unittest.main()
