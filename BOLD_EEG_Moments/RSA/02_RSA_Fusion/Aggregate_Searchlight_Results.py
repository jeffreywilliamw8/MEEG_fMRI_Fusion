"""
Aggregates the per-time-point whole-brain searchlight RSA fusion correlation maps -- saved one
file per original time point by Searchlight_Fusion_RSA.py, across its 18 time point splits of
100 time points each -- into a single per-subject, per-hemisphere correlation time course of
shape (1800 time points, 163842 vertices).

Reads:  .../searchlight_fusion/eeg_rdm_metric-{metric}/n_neighbours-{k}/subject-{subject}/
        {hemisphere}_hemisphere/time_point_{t:04d}.npy        (1800 files, each (163842,))
Writes: .../searchlight_fusion/eeg_rdm_metric-{metric}/n_neighbours-{k}/aggregated_results/
        subject-{subject}/subject-{subject}_{hemisphere}_hemisphere_timecourse.npy  (1800, 163842)

The output path mirrors the aggregated_results/subject-{subject}/... convention already used for
the NSD project's searchlight fusion results, so a BMD-specific ROI-averaging/plotting script can
be adapted from the NSD one with the same file-loading logic.

One instance loops over both hemispheres and all 10 BOLD_EEG_Moments subjects (1-10),
aggregating each (hemisphere, subject) time course in turn -- a single run covers everyone.

Parameters
----------
eeg_rdm_metric : str
    Which EEG RDM metric's searchlight results to aggregate.
n_neighbours : int
    The searchlight neighbourhood size (number of vertices per searchlight) whose results to
    aggregate.
n_time_points : int
    Total number of EEG time points (default 1800).
n_vertices : int
    Expected number of fsaverage vertices per hemisphere (default 163842) -- used only as a
    sanity check against the first successfully loaded time point's shape.
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
parser.add_argument('--eeg_rdm_metric', type=str, default='pearsonr', choices=['pearsonr', 'crossnobis', 'decoding_accuracy'])
parser.add_argument('--n_neighbours', type=int, default=100)
parser.add_argument('--n_time_points', type=int, default=1800)
parser.add_argument('--n_vertices', type=int, default=163842)
args = parser.parse_args()

print('>>> Aggregating Searchlight RSA Fusion time courses across time point splits <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

hemisphere_list = ['left', 'right']
subject_list = list(range(1, 11))  # all 10 BOLD_EEG_Moments subjects

for hemisphere in hemisphere_list:
    for subject in subject_list:

        print(f"\n>>> Hemisphere {hemisphere}, Subject {subject} <<<")

        # =========================================================================
        # 1. Directory of the per-time-point files written by Searchlight_Fusion_RSA.py
        # =========================================================================
        results_dir = os.path.join(
            f'/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/results/correlations/'
            f'searchlight_fusion/eeg_rdm_metric-{args.eeg_rdm_metric}/n_neighbours-{args.n_neighbours}/'
            f'subject-{subject}/{hemisphere}_hemisphere'
        )
        print(f"Reading per-time-point files from: {results_dir}")

        # =========================================================================
        # 2. Load and stack every time point, in order, into one (n_time_points, n_vertices) array
        # =========================================================================
        timecourse = None
        n_missing = 0

        for t in tqdm(range(args.n_time_points),
                       desc=f'Aggregating time points ({hemisphere}, subject {subject})'):
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
                # Allocate on the first successfully loaded time point, so the array's vertex count
                # always matches what was actually written rather than just an assumed constant.
                assert corrs.shape[0] == args.n_vertices, \
                    f"Expected {args.n_vertices} vertices, got {corrs.shape[0]} in {file_path}"
                timecourse = np.full((args.n_time_points, corrs.shape[0]), np.nan, dtype=np.float32)

            timecourse[t] = corrs

        if timecourse is None:
            print(f"  No time point files found in {results_dir} -- skipping "
                  f"hemisphere {hemisphere}, subject {subject}.")
            continue

        if n_missing > 0:
            print(f"  Warning: {n_missing} / {args.n_time_points} time points were missing and left as NaN.")

        print(f"  Aggregated time course shape: {timecourse.shape} (time points, vertices)")

        # =========================================================================
        # 3. Save -- mirrors the aggregated_results/subject-{subject}/... convention already used
        # for the NSD project's searchlight fusion results
        # =========================================================================
        aggregated_dir = os.path.join(
            f'/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/results/correlations/'
            f'searchlight_fusion/eeg_rdm_metric-{args.eeg_rdm_metric}/n_neighbours-{args.n_neighbours}/'
            f'aggregated_results/subject-{subject}'
        )
        # '/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/results/correlations/searchlight_fusion/eeg_rdm_metric-pearsonr/n_neighbours-100/aggregated_results/subject-1'
        os.makedirs(aggregated_dir, exist_ok=True)

        save_path = os.path.join(
            aggregated_dir, f'subject-{subject}_{hemisphere}_hemisphere_timecourse.npy'
        )
        np.save(save_path, timecourse)
        print(f"  Saved: {save_path}")

print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds.")