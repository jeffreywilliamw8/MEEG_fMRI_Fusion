"""
Aggregates the per-time-point whole-brain MEG-fMRI searchlight RSA fusion correlation maps --
saved one file per original time point by THINGS_Searchlight_MEG_fMRI_RSA_Fusion.py, across its
3 time point splits of 50 time points each -- into a single per-subject correlation time course
of shape (141 time points, n_voxels).

Unlike the NSD/BMD searchlight aggregation (which splits into separate lh/rh hemisphere files),
THINGS' fMRI searchlight results are a single whole-brain array per subject, so there is no
hemisphere loop here.

Reads:  .../searchlight_fusion/eeg_rdm_metric-{metric}/radius-{radius}/subject-{subject:02d}/
        time_point_{t:04d}.npy         (141 files, each (n_voxels,))
Writes: .../searchlight_fusion/eeg_rdm_metric-{metric}/radius-{radius}/aggregated_results/
        subject-{subject:02d}/subject-{subject:02d}_timecourse.npy   (141, n_voxels)

One instance loops over all 3 THINGS fMRI subjects (1-3) for a single eeg_rdm_metric/radius
combination -- a single run covers everyone.

Parameters
----------
eeg_rdm_metric : str
    Which EEG RDM metric's searchlight results to aggregate.
radius : float
    The searchlight radius (mm) whose results to aggregate -- must match the radius
    THINGS_Searchlight_MEG_fMRI_RSA_Fusion.py was run with.
n_time_points : int
    Total number of MEG time points (default 141, for tmax=0.6).
"""

import os
import argparse
import time

import numpy as np
from tqdm import tqdm

start_time = time.time()

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--eeg_rdm_metric', type=str, default='pearsonr',
                     choices=['pearsonr', 'crossnobis', 'decoding_accuracy'])
parser.add_argument('--radius', type=float, default=10.0)
parser.add_argument('--n_time_points', type=int, default=141)
args = parser.parse_args()

print('>>> Aggregating THINGS Searchlight RSA Fusion time courses across time point splits <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

fmri_subjects = [1, 2, 3]  # all THINGS-fMRI subjects

for subject in fmri_subjects:

    print(f"\n>>> Subject {subject:02d} <<<")

    # =========================================================================
    # 1. Directory of the per-time-point files written by THINGS_Searchlight_MEG_fMRI_RSA_Fusion.py
    # =========================================================================
    results_dir = os.path.join(
        f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/correlations/'
        f'searchlight_fusion/eeg_rdm_metric-{args.eeg_rdm_metric}/radius-{args.radius}/'
        f'subject-{subject:02d}'
    )
    print(f"Reading per-time-point files from: {results_dir}")

    # =========================================================================
    # 2. Load and stack every time point, in order, into one (n_time_points, n_voxels) array
    # =========================================================================
    timecourse = None
    n_missing = 0

    for t in tqdm(range(args.n_time_points), desc=f'Aggregating time points (subject {subject:02d})'):
        file_path = os.path.join(results_dir, f'time_point_{t:04d}.npy')
        try:
            corrs = np.load(file_path)
        except FileNotFoundError:
            print(f"  Missing: {file_path}")
            n_missing += 1
            if timecourse is not None:
                timecourse[t] = np.nan
            continue

        if timecourse is None:
            # Allocate on the first successfully loaded time point, so the array's voxel count
            # always matches what was actually written rather than an assumed constant.
            timecourse = np.full((args.n_time_points, corrs.shape[0]), np.nan, dtype=np.float32)

        timecourse[t] = corrs

    if timecourse is None:
        print(f"  No time point files found in {results_dir} -- skipping subject {subject:02d}.")
        continue

    if n_missing > 0:
        print(f"  Warning: {n_missing} / {args.n_time_points} time points were missing and left as NaN.")

    print(f"  Aggregated time course shape: {timecourse.shape} (time points, voxels)")

    # =========================================================================
    # 3. Save -- mirrors the aggregated_results/subject-.../... convention already used for the
    # NSD/BMD searchlight fusion results
    # =========================================================================
    aggregated_dir = os.path.join(
        f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/correlations/'
        f'searchlight_fusion/eeg_rdm_metric-{args.eeg_rdm_metric}/radius-{args.radius}/'
        f'aggregated_results/subject-{subject:02d}'
    )
    os.makedirs(aggregated_dir, exist_ok=True)

    save_path = os.path.join(aggregated_dir, f'subject-{subject:02d}_timecourse.npy')
    np.save(save_path, timecourse)
    print(f"  Saved: {save_path}")

print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds.")