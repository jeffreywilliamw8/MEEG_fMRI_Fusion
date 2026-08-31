"""
Aggregates the per-timepoint searchlight RSA fusion files
(time_point_TTTT.npy, each (n_voxels,)) into one whole-brain time course per
fMRI subject, shape (n_time, n_voxels). Reports any missing timepoint jobs.

Subjects are saved to separate files because THINGS fMRI voxel counts differ
across subjects (211,339 / 226,950 / 189,164), so they cannot be stacked.

Parameters
----------
fmri_subjects : fMRI participants to aggregate.
eeg_rdm_metric : {'pearsonr', 'crossnobis', 'decoding_accuracy'} results to aggregate.
radius : searchlight radius (mm) the results were computed with.
allow_incomplete : save even if some timepoint files are missing (gaps stay NaN).
"""

import os
import argparse
import time

import numpy as np


start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--eeg_rdm_metric', type=str, default='pearsonr',
                     choices=['pearsonr', 'crossnobis', 'decoding_accuracy'])
parser.add_argument('--radius', type=float, default=10.0)
parser.add_argument('--tmax', type=float, default=0.8)
parser.add_argument('--allow_incomplete', action='store_true')
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--results_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/correlations/'
                             'searchlight_fusion')
parser.add_argument('--save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/correlations/'
                             'searchlight_fusion_whole_brain')
args = parser.parse_args()

print('>>> Aggregating Searchlight RSA Fusion whole-brain results <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

times = np.load(os.path.join(args.data_dir, 'meg_times.npy')) # pre-saved file of MEG times from -100 to +800 ms
n_time = len(times)
print(f"\nExpecting {n_time} timepoints per subject")

base_results_dir = os.path.join(
    args.results_dir, f'eeg_rdm_metric-{args.eeg_rdm_metric}', f'radius-{args.radius}')
save_dir = os.path.join(
    args.save_dir, f'eeg_rdm_metric-{args.eeg_rdm_metric}', f'radius-{args.radius}')
os.makedirs(save_dir, exist_ok=True)

for subject in args.fmri_subjects:

    subject_dir = os.path.join(base_results_dir, f'subject-{subject:02d}')
    if not os.path.isdir(subject_dir):
        print(f"\nfMRI sub-{subject:02d}: results directory not found ({subject_dir}), skipping.")
        continue

    print(f"\n>>> fMRI sub-{subject:02d} <<<")

    timecourse = None
    missing = []
    for t in range(n_time):
        path = os.path.join(subject_dir, f'time_point_{t:04d}.npy')
        if not os.path.exists(path):
            missing.append(t)
            continue
        corrs = np.load(path)
        if timecourse is None:
            timecourse = np.full((n_time, corrs.shape[0]), np.nan, dtype=np.float32)
            print(f"  {corrs.shape[0]} voxels")
        timecourse[t] = corrs

    if timecourse is None:
        print("  No timepoint files found, skipping.")
        continue

    if missing:
        print(f"  MISSING {len(missing)}/{n_time} timepoints: {missing[:20]}"
              f"{' ...' if len(missing) > 20 else ''}")
        if not args.allow_incomplete:
            print("  Not saving (pass --allow_incomplete to save with NaN gaps).")
            del timecourse
            continue
    else:
        print(f"  All {n_time} timepoints present")

    save_path = os.path.join(save_dir, f'subject-{subject:02d}.npy')
    np.save(save_path, timecourse)
    print(f"  Saved: {save_path} (shape {timecourse.shape})")

    del timecourse

print(f"\nDone! Total Time: {time.time() - start_time:.2f} seconds.")