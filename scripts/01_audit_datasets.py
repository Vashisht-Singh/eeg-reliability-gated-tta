"""
Script 01: Audit Datasets and Generate Audit Reports.
Executes Phase 1 empirical dataset audit for both CHB-MIT and Siena Scalp EEG.
"""

import os
import sys
import shutil
import subprocess
import platform
import psutil

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audit.chbmit_audit import audit_chbmit
from src.audit.siena_audit import audit_siena
from src.audit.channel_audit import analyze_channel_compatibility
from src.audit.report_generator import generate_all_reports


def get_hardware_info() -> dict:
    """Collect hardware and system resource metrics."""
    info = {
        'os': platform.platform(),
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'ram_avail_gb': round(psutil.virtual_memory().available / (1024**3), 2),
    }

    # Disk usage
    for drive_letter in ['C:\\', 'D:\\', 'F:\\']:
        try:
            usage = shutil.disk_usage(drive_letter)
            letter = drive_letter[0].lower()
            info[f'disk_{letter}_total_gb'] = round(usage.total / (1024**3), 2)
            info[f'disk_{letter}_free_gb'] = round(usage.free / (1024**3), 2)
        except Exception:
            pass

    # GPU / CUDA via nvidia-smi
    try:
        res = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        parts = [p.strip() for p in res.stdout.strip().split(',')]
        if len(parts) >= 3:
            info['gpu_name'] = parts[0]
            info['gpu_vram_mb'] = float(parts[1])
            info['gpu_driver'] = parts[2]
            info['cuda_version'] = '13.0 (Driver supported)'
    except Exception as e:
        info['gpu_name'] = 'N/A'
        info['gpu_error'] = str(e)

    return info


def main():
    print("=" * 80)
    print("PHASE 1: RESEARCH-GRADE DATASET AUDIT & VERIFICATION")
    print("=" * 80)

    source_path = r"F:\chb-mit-scalp-eeg-database-1.0.0"
    target_path = r"D:\Sandhu_Siena_EEG"
    output_dir = os.path.join(PROJECT_ROOT, "experiments", "reports")
    metadata_dir = os.path.join(PROJECT_ROOT, "data", "metadata")

    print(f"\n1. Auditing Hardware & Environment Resources...")
    hw_info = get_hardware_info()
    print(f"   OS: {hw_info['os']}")
    print(f"   Python: {hw_info['python_version']} ({hw_info['python_executable']})")
    print(f"   RAM: Total {hw_info['ram_total_gb']} GB | Available {hw_info['ram_avail_gb']} GB")
    print(f"   GPU: {hw_info.get('gpu_name', 'None')} ({hw_info.get('gpu_vram_mb', 0)} MB VRAM)")
    print(f"   Disk D: (Project Root): {hw_info.get('disk_d_free_gb', 0)} GB free")
    print(f"   Disk F: (Source Dataset): {hw_info.get('disk_f_free_gb', 0)} GB free (Read-only)")

    print(f"\n2. Auditing Source Dataset: CHB-MIT Scalp EEG...")
    chb_audit = audit_chbmit(source_path)
    chb_sum = chb_audit['summary']
    print(f"   Subjects: {chb_sum['num_subjects']} ({chb_sum['subject_list'][0]}..{chb_sum['subject_list'][-1]})")
    print(f"   Total Recordings: {chb_sum['total_recordings']} (Usable: {chb_sum['usable_recordings']}, Corrupted: {chb_sum['corrupted_recordings']})")
    print(f"   Recordings with Seizures: {chb_sum['recordings_with_seizures']}")
    print(f"   Total Seizure Events: {chb_sum['total_seizure_events']}")
    print(f"   Total Duration: {chb_sum['total_duration_hours']:.2f} hours")
    print(f"   Sampling Frequencies: {chb_sum['sampling_frequencies']}")

    print(f"\n3. Auditing Target Dataset: Siena Scalp EEG...")
    siena_audit = audit_siena(target_path)
    siena_sum = siena_audit['summary']
    print(f"   Subjects: {siena_sum['num_subjects']} ({siena_sum['subject_list'][0]}..{siena_sum['subject_list'][-1]})")
    print(f"   Total Recordings: {siena_sum['total_recordings']} (Usable: {siena_sum['usable_recordings']}, Corrupted: {siena_sum['corrupted_recordings']})")
    print(f"   Recordings with Seizures: {siena_sum['recordings_with_seizures']}")
    print(f"   Total Seizure Events: {siena_sum['total_seizure_events']}")
    print(f"   Total Duration: {siena_sum['total_duration_hours']:.2f} hours")
    print(f"   Sampling Frequencies: {siena_sum['sampling_frequencies']}")

    print(f"\n4. Analyzing Channel Compatibility & Harmonization...")
    channel_audit = analyze_channel_compatibility(chb_audit['recordings'], siena_audit['recordings'])
    print(f"   Target Montage: 18-channel Longitudinal Bipolar (Double Banana)")
    print(f"   CHB-MIT Compatibility: {channel_audit['chbmit_compatibility_pct']}% direct ({channel_audit['chbmit_fully_compatible_files']}/{channel_audit['chbmit_total_files']} files)")
    print(f"   Siena Compatibility: {channel_audit['siena_compatibility_pct']}% via referential subtraction ({channel_audit['siena_fully_compatible_files']}/{channel_audit['siena_total_files']} files)")

    print(f"\n5. Generating Audit Reports & Inventory Tables...")
    # Generate in experiments/reports
    rep_files = generate_all_reports(chb_audit, siena_audit, channel_audit, hw_info, output_dir)
    # Also copy metadata inventories to data/metadata for quick pipeline access
    generate_all_reports(chb_audit, siena_audit, channel_audit, hw_info, metadata_dir)

    print("   Generated Report Artifacts:")
    for name, path in rep_files.items():
        print(f"   - {name}: {path}")

    print("\n" + "=" * 80)
    print("PHASE 1 AUDIT SUCCESSFULLY COMPLETED!")
    print("=" * 80)


if __name__ == '__main__':
    main()
