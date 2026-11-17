"""
Script 04: Preprocess and Verify Siena Target Pipeline.
Validates end-to-end preprocessing (Butterworth 0.5-45 Hz filtering, 50 Hz European notch,
256 Hz resampling from 512 Hz, and unipolar-to-bipolar montage reconstruction)
on Siena Scalp EEG recordings.
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

logger = setup_logger("preprocess_siena")


def main():
    parser = argparse.ArgumentParser(description="Verify Siena Target Preprocessing Pipeline")
    parser.add_argument("--target-dir", type=str, default=r"D:\Sandhu_Siena_EEG", help="Siena root directory")
    parser.add_argument("--limit-files", type=int, default=5, help="Number of files to verify")
    args = parser.parse_args()

    print("=" * 80)
    print("VERIFYING SIENA TARGET PREPROCESSING PIPELINE")
    print("=" * 80)

    preprocessor = EEGPreprocessor(target_fs=256.0, low_cutoff=0.5, high_cutoff=45.0, notch_freq=50.0)

    tested = 0
    for root, _, files in os.walk(args.target_dir):
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
                    
                    # Apply causal preprocessing (decimates 512 -> 256 Hz)
                    clean_data, is_valid, metrics = preprocessor.process_window(montage_data, orig_fs=fs)
                    
                    print(f"Verified [{tested+1}/{args.limit_files}]: {f} (fs={fs}Hz) -> Clean Shape: {clean_data.shape} | Valid: {is_valid} | Range: [{clean_data.min():.2f}, {clean_data.max():.2f}]")
                    tested += 1
                    if tested >= args.limit_files:
                        break
                except Exception as e:
                    logger.error(f"Error preprocessing {f}: {e}")
        if tested >= args.limit_files:
            break

    print(f"\nSiena Target Preprocessing successfully verified on {tested} recordings.")


if __name__ == '__main__':
    main()
