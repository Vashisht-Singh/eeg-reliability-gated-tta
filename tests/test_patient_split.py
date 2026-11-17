"""
Test Suite: Patient-Independent Split Verification.
Validates patient-level isolation, zero subject leakage, and co-location of chb01 and chb21.
"""

import os
import sys
import csv
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestPatientSplit(unittest.TestCase):

    def setUp(self):
        self.split_csv = os.path.join(PROJECT_ROOT, "data", "manifests", "patient_splits.csv")

    def test_split_file_exists(self):
        """Verify patient split CSV is created."""
        self.assertTrue(os.path.exists(self.split_csv), f"Split CSV missing: {self.split_csv}")

    def test_zero_leakage_and_chb01_chb21_colocation(self):
        """Verify strict disjoint patient partitions and chb01/chb21 binding."""
        train_subjs = set()
        val_subjs = set()
        test_subjs = set()

        with open(self.split_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                role = row['split_role']
                subj = row['subject_id']
                if role == 'train':
                    train_subjs.add(subj)
                elif role == 'val':
                    val_subjs.add(subj)
                elif role == 'source_test':
                    test_subjs.add(subj)

        # 1. Total 24 subjects
        all_subjs = train_subjs.union(val_subjs).union(test_subjs)
        self.assertEqual(len(all_subjs), 24, "All 24 CHB-MIT subjects must be partitioned.")

        # 2. Strict pairwise disjointness (zero patient leakage)
        self.assertEqual(len(train_subjs.intersection(val_subjs)), 0, "Leakage between train and val!")
        self.assertEqual(len(train_subjs.intersection(test_subjs)), 0, "Leakage between train and test!")
        self.assertEqual(len(val_subjs.intersection(test_subjs)), 0, "Leakage between val and test!")

        # 3. Critical: chb01 and chb21 must be in the exact same partition
        chb01_split = 'train' if 'chb01' in train_subjs else ('val' if 'chb01' in val_subjs else 'test')
        chb21_split = 'train' if 'chb21' in train_subjs else ('val' if 'chb21' in val_subjs else 'test')
        self.assertEqual(chb01_split, chb21_split, "chb01 and chb21 represent the same person and must share the same split!")


if __name__ == '__main__':
    unittest.main()
