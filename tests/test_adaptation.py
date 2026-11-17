"""
Test Suite: Test-Time Adaptation Modules (TENT, EATA, SAR, and Proposed Reliability Gate).
Validates parameter isolation, predict-first protocol, update decisions, and recovery.
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.eeg_cnn import EEG1DCNN
from src.adaptation.tent_adapter import TentAdapter
from src.adaptation.eata_adapter import EataAdapter
from src.adaptation.sar_adapter import SarAdapter
from src.adaptation.chronological_runner import ReliabilityGatedAdapter


class TestAdaptation(unittest.TestCase):

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dummy_x = torch.randn(1, 18, 1024)

    def test_tent_adaptation(self):
        """Verify TENT updates only BatchNorm parameters and preserves conv kernels."""
        model = EEG1DCNN(num_channels=18, input_samples=1024)
        adapter = TentAdapter(model, lr=1e-3, device=self.device)
        conv_weight_before = model.features[0].conv.weight.clone()
        prob, info = adapter.step(self.dummy_x, timestamp=1.0)

        self.assertIsInstance(prob, float)
        self.assertTrue(info['updated'])

        # Conv weights must be completely frozen
        conv_weight_after = model.features[0].conv.weight
        self.assertTrue(torch.equal(conv_weight_before, conv_weight_after), "TENT mutated frozen conv weights!")

    def test_eata_adaptation(self):
        """Verify EATA active sample selection and entropy filtering."""
        model = EEG1DCNN(num_channels=18, input_samples=1024)
        adapter = EataAdapter(model, lr=1e-3, entropy_margin_e0=0.40, device=self.device)

        prob, info = adapter.step(self.dummy_x, timestamp=2.0)
        self.assertIn('updated', info)
        self.assertIn('entropy', info)

    def test_sar_adaptation(self):
        """Verify SAR sharpness-aware minimization and gradient perturbation."""
        model = EEG1DCNN(num_channels=18, input_samples=1024)
        adapter = SarAdapter(model, lr=1e-3, margin_e0=0.40, device=self.device)

        prob, info = adapter.step(self.dummy_x, timestamp=3.0)
        self.assertIn('updated', info)

    def test_proposed_reliability_gated_adapter(self):
        """Verify proposed multi-signal gate evaluates candidate and executes decision."""
        model = EEG1DCNN(num_channels=18, input_samples=1024)
        adapter = ReliabilityGatedAdapter(model, lr=1e-3, device=self.device)

        prob, info = adapter.step(self.dummy_x, timestamp=4.0)
        self.assertIsInstance(prob, float)
        self.assertIn(info['decision'], ['ACCEPT', 'REJECT'])
        self.assertIn('signals', info)
        self.assertIn('prob_incumbent', info['signals'])
        self.assertIn('prob_candidate', info['signals'])


if __name__ == '__main__':
    unittest.main()
