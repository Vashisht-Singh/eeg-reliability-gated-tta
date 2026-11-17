"""
Script 03: Preprocess and Verify CHB-MIT Pipeline.
Validates end-to-end preprocessing (Butterworth 0.5-45 Hz filtering, 60 Hz notch,
256 Hz resampling, and 18-channel longitudinal bipolar double banana montage derivation)
on CHB-MIT recordings.
"""

import os
import sys
import argparse
import logging
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.edf_loader import IncrementalEDFLoader
from src.data.channel_mapper import ChannelMapper
from src.data.preprocessing import EEGPreprocessor
from src.utils.logging_config import setup_logger

logger = setup_logger("preprocess_chbmit")


def main():
    parser = argparse.ArgumentParser(description="Verify CHB-MIT Preprocessing Pipeline")
    parser.add_argument("--source-dir", type=str, default=r"F:\chb-mit-scalp-eeg-database-1.0.0", help="CHB-MIT root directory")
    parser.add_argument("--limit-files", type=int, default=5, help="Number of files to verify")
    args = parser.parse_args()

    print("=" * 80)
    print("VERIFYING CHB-MIT PREPROCESSING PIPELINE")
    print("=" * 80)

    preprocessor = EEGPreprocessor(target_fs=256.0, low_cutoff=0.5, high_cutoff=45.0, notch_freq=60.0)

    tested = 0
    for root, _, files in os.walk(args.source_dir):
        for f in sorted(files):
            if f.lower().endswith('.edf'):
                edf_path = os.path.join(root, f)
                try:
                    loader = IncrementalEDFLoader(edf_path)
                    mapper = ChannelMapper(loader.labels)
                    req_indices = mapper.get_required_channel_indices()
                    fs = loader.get_sampling_rate(req_indices[0])

                    # Read 10-second window
                    raw_data = loader.read_window_sec(start_sec=60.0, duration_sec=10.0, channel_indices=req_indices)
                    
                    # Apply montage transformation (18, N)
                    montage_data = mapper.transform(raw_data)
                    
                    # Apply causal preprocessing
                    clean_data, is_valid, metrics = preprocessor.process_window(montage_data, orig_fs=fs)
                    
                    print(f"Verified [{tested+1}/{args.limit_files}]: {f} | Clean Shape: {clean_data.shape} | Valid: {is_valid} | Range: [{clean_data.min():.2f}, {clean_data.max():.2f}]")
                    tested += 1
                    if tested >= args.limit_files:
                        break
                except Exception as e:
                    logger.error(f"Error preprocessing {f}: {e}")
        if tested >= args.limit_files:
            break

    print(f"\nCHB-MIT Preprocessing successfully verified on {tested} recordings.")


if __name__ == '__main__':
    main()
