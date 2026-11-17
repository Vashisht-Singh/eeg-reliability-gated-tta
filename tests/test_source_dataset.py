"""
Test Suite: Source Domain Window Dataset Loading.
Verifies lazy loading of preprocessed CHB-MIT windows and class weighting calculation.
"""

import os
import sys
import csv
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.source_dataset import CHBMITWindowDataset


class TestSourceDataset(unittest.TestCase):

    def setUp(self):
        self.smoke_manifest = os.path.join(PROJECT_ROOT, "data", "manifests", "source_smoke_test_manifest.csv")
        self.assertTrue(os.path.exists(self.smoke_manifest), "Smoke test manifest missing!")

        with open(self.smoke_manifest, 'r', encoding='utf-8') as f:
            self.entries = list(csv.DictReader(f))

    def test_lazy_loading_window_shape_and_label(self):
        """Verify on-the-fly window reading, montage derivation, and preprocessing."""
        dataset = CHBMITWindowDataset(self.entries)
        self.assertEqual(len(dataset), len(self.entries))

        x, y, w_id = dataset[0]
        self.assertEqual(x.shape, (18, 1024))
        self.assertIn(y, [0.0, 1.0])
        self.assertFalse(np.isnan(x).any())
        self.assertFalse(np.isinf(x).any())

    def test_class_weights_computation(self):
        """Verify positive class weight calculation on source windows."""
        dataset = CHBMITWindowDataset(self.entries)
        pos_weight = dataset.get_class_weights()
        self.assertGreater(pos_weight, 0.0)


if __name__ == '__main__':
    unittest.main()
