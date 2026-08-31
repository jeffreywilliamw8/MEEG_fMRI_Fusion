"""
Aggregates the per-subject VDNN vs LLM RSA variance partitioning files into a
whole-brain results tree: checks all three partitions are present
with matching (n_time, n_voxels) shapes, and re-saves each subject's
consolidated time courses.

Subjects are kept in separate files because THINGS fMRI voxel counts differ
across subjects (211,339 / 226,950 / 189,164), so they cannot be stacked.

Parameters
----------
fmri_subjects : fMRI participants to aggregate.
eeg_rdm_metric : {'pearsonr', 'crossnobis', 'decoding_accuracy'} results to aggregate.
radius : searchlight radius (mm) the results were computed with.
allow_incomplete : save even if some partitions are missing (gaps stay NaN).
"""

import os
import argparse
import time

import numpy as np

start_time = time.time()

partitions = ['unique_vision', 'unique_language']

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--eeg_rdm_metric', type=str, default='pearsonr',
                     choices=['pearsonr', 'crossnobis', 'decoding_accuracy'])
parser.add_argument('--radius', type=float, default=10.0)
parser.add_argument('--allow_incomplete', action='store_true')
parser.add_argument('--results_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/'
                             'variance_partitioning')
parser.add_argument('--save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/'
                             'variance_partitioning_whole_brain')
args = parser.parse_args()

print('>>> Aggregating RSA Variance Partitioning whole-brain results <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

base_results_dir = os.path.join(
    args.results_dir, f'eeg_rdm_metric-{args.eeg_rdm_metric}', f'radius-{args.radius}')
save_dir = os.path.join(
    args.save_dir, f'eeg_rdm_metric-{args.eeg_rdm_metric}', f'radius-{args.radius}')
os.makedirs(save_dir, exist_ok=True)

for subject in args.fmri_subjects:

    path = os.path.join(base_results_dir, f'subject-{subject:02d}.npy')
    if not os.path.exists(path):
        print(f"\nfMRI sub-{subject:02d}: results file not found ({path}), skipping.")
        continue

    print(f"\n>>> fMRI sub-{subject:02d} <<<")
    results = np.load(path, allow_pickle=True).item()

    reference_shape = None
    for part in partitions:
        if part in results:
            reference_shape = results[part].shape
            break

    if reference_shape is None:
        print("  None of the three partitions found in the file, skipping.")
        continue

    n_time, n_voxels = reference_shape
    print(f"  {n_time} timepoints x {n_voxels} voxels")

    aggregated = {}
    missing = []
    for part in partitions:
        if part not in results:
            missing.append(part)
            aggregated[part] = np.full(reference_shape, np.nan, dtype=np.float32)
            continue
        arr = results[part]
        if arr.shape != reference_shape:
            raise ValueError(
                f"fMRI sub-{subject:02d}: partition '{part}' has shape {arr.shape}, expected "
                f"{reference_shape}. The three partitions must share the same grid."
            )
        aggregated[part] = arr.astype(np.float32)

    if missing:
        print(f"  MISSING {len(missing)}/{len(partitions)} partitions: {missing}")
        if not args.allow_incomplete:
            print("  Not saving (pass --allow_incomplete to save with NaN gaps).")
            del results, aggregated
            continue
    else:
        print(f"  All {len(partitions)} partitions present")

    save_path = os.path.join(save_dir, f'subject-{subject:02d}.npy')
    np.save(save_path, aggregated)
    print(f"  Saved: {save_path}")

    del results, aggregated

print(f"\nDone! Total Time: {time.time() - start_time:.2f} seconds.")