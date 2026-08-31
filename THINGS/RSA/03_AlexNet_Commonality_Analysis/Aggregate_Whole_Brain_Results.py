"""
Aggregates the per-layer AlexNet commonality analysis files (layer-L.npy, each
already a whole-brain (n_time, n_voxels) array) into one file per fMRI subject
holding all layers, shape (n_layers, n_time, n_voxels). Reports any missing
layer jobs.

Subjects are saved to separate files because THINGS fMRI voxel counts differ
across subjects (211,339 / 226,950 / 189,164), so they cannot be stacked.

Parameters
----------
fmri_subjects : fMRI participants to aggregate.
eeg_rdm_metric : {'pearsonr', 'crossnobis', 'decoding_accuracy'} results to aggregate.
radius : searchlight radius (mm) the results were computed with.
allow_incomplete : save even if some layer files are missing (gaps stay NaN).
"""

import os
import argparse
import time

import numpy as np

start_time = time.time()

alexnet_layers = [
    'features.2', 'features.5', 'features.7', 'features.9', 'features.12',
    'classifier.2', 'classifier.5', 'classifier.6'
]
layer_display_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC6', 'FC7', 'FC8']

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--eeg_rdm_metric', type=str, default='pearsonr',
                     choices=['pearsonr', 'crossnobis', 'decoding_accuracy'])
parser.add_argument('--radius', type=float, default=10.0)
parser.add_argument('--allow_incomplete', action='store_true')
parser.add_argument('--results_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/'
                             'commonality_analysis/layerwise_alexnet')
parser.add_argument('--save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/'
                             'commonality_analysis/layerwise_alexnet_whole_brain')
args = parser.parse_args()

print('>>> Aggregating AlexNet Commonality Analysis whole-brain results <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

n_layers = len(alexnet_layers)
print(f"\nExpecting {n_layers} layers per subject")

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

    stacked = None
    missing = []
    for l_idx, layer in enumerate(alexnet_layers):
        path = os.path.join(subject_dir, f'layer-{layer}.npy')
        if not os.path.exists(path):
            missing.append(layer_display_names[l_idx])
            continue
        r2 = np.load(path)  # (n_time, n_voxels)
        if stacked is None:
            n_time, n_voxels = r2.shape
            stacked = np.full((n_layers, n_time, n_voxels), np.nan, dtype=np.float32)
            print(f"  {n_time} timepoints x {n_voxels} voxels")
        stacked[l_idx] = r2
        del r2

    if stacked is None:
        print("  No layer files found, skipping.")
        continue

    if missing:
        print(f"  MISSING {len(missing)}/{n_layers} layers: {missing}")
        if not args.allow_incomplete:
            print("  Not saving (pass --allow_incomplete to save with NaN gaps).")
            del stacked
            continue
    else:
        print(f"  All {n_layers} layers present")

    save_path = os.path.join(save_dir, f'subject-{subject:02d}.npy')
    np.save(save_path, {
        'commonality': stacked,  # (n_layers, n_time, n_voxels)
        'layers': alexnet_layers,
        'layer_display_names': layer_display_names,
    })
    print(f"  Saved: {save_path} (shape {stacked.shape})")

    del stacked

print(f"\nDone! Total Time: {time.time() - start_time:.2f} seconds.")