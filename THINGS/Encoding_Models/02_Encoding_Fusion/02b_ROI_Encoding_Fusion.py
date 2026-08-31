"""
ROI-restricted MEG-to-fMRI encoding fusion. Same modeling pipeline as
THINGS_Whole_Brain_Encoding_Fusion.py, but instead of sweeping --fmri_split over
np.array_split chunks of the whole brain, it takes an ROI name (e.g. 'V1')
and an ncsnr (noise ceiling) threshold, and fits/predicts/correlates only
on that ROI's voxels

Parameters:
- fmri_subject: The THINGS-fMRI subject (1, 2, or 3).
- roi: ROI name (e.g. 'V1'), matching a key in the precomputed metadata_fmri['roi'] dict.
- ncsnr_threshold: Only ROI voxels whose noise ceiling exceeds this value are kept.
- data_dir: Directory containing the pre-computed fmri_sub-XX.npy, meg_train.npy,
  and meg_test.npy files.
- corrs_save_dir: Root for the correlation time courses (correlations.npy only).
- weights_save_dir: Root for the regression weights and voxel indices.
- n_jobs: Parallel jobs across timepoints (-1 = all available cores).
- tmax: Upper time bound (seconds post stimulus onset) the MEG data is truncated to.
"""

import argparse
import os
import time
import gc
import numpy as np
from joblib import Parallel, delayed
from utils import load_fmri_roi_data, load_meg_data, fit_predict_correlate_timepoint

start_time = time.time()

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)  # 1, 2, or 3
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--ncsnr_threshold', type=float, default=20.0)
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--corrs_save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/correlations/roi_encoding_fusion')
parser.add_argument('--weights_save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/regression_weights/roi_encoding_fusion')
parser.add_argument('--n_jobs', type=int, default=-1)
parser.add_argument('--tmax', type=float, default=0.8)
args = parser.parse_args()

print('>>> MEG-fMRI Encoding Fusion (ROI) <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

seed = 8
np.random.seed(seed)

eps = 1e-8

# =============================================================================
# Loading this run's fMRI (ROI + ncsnr filtered) and MEG data
# =============================================================================
fmri_train, fmri_test, voxel_idx = load_fmri_roi_data(args.fmri_subject, args.roi, args.ncsnr_threshold)
meg_train, meg_test = load_meg_data(tmax=args.tmax)
n_time = meg_train.shape[2]
n_voxels = fmri_train.shape[1]

print(f"\n[fMRI sub-{args.fmri_subject:02d}, ROI {args.roi}, ncsnr >= "
      f"{args.ncsnr_threshold}] train {fmri_train.shape}, test {fmri_test.shape} "
      f"({n_voxels} voxels)")
print(f"[MEG] train {meg_train.shape}, test {meg_test.shape}, {n_time} timepoints")

assert meg_train.shape[0] == fmri_train.shape[0], \
    f"sub-{args.fmri_subject:02d}: MEG train ({meg_train.shape[0]}) / fMRI train " \
    f"({fmri_train.shape[0]}) stimulus count mismatch."
assert meg_test.shape[0] == fmri_test.shape[0], \
    f"sub-{args.fmri_subject:02d}: MEG test ({meg_test.shape[0]}) / fMRI test " \
    f"({fmri_test.shape[0]}) stimulus count mismatch."

fmri_test_z = (fmri_test - fmri_test.mean(0)) / (fmri_test.std(0) + eps)

# =============================================================================
# Fitting/predicting/correlating, parallelized across timepoints
# (return_weights=True: also get back this timepoint's RidgeCV weights)
# =============================================================================
results = Parallel(n_jobs=args.n_jobs)(
    delayed(fit_predict_correlate_timepoint)(
        t, meg_train, meg_test, fmri_train, fmri_test_z, return_weights=True
    )
    for t in range(n_time)
)
correlations = np.array([r[0] for r in results], dtype=np.float32)  # (n_time, n_voxels)
coefs = np.array([r[1] for r in results], dtype=np.float32)         # (n_time, n_voxels, n_channels)
intercepts = np.array([r[2] for r in results], dtype=np.float32)    # (n_time, n_voxels)

# =============================================================================
# Saving: correlations and regression weights go to separate roots
# =============================================================================

corrs_save_dir = os.path.join(args.corrs_save_dir, f'fmri_sub-{args.fmri_subject:02d}')
os.makedirs(corrs_save_dir, exist_ok=True)
corrs_path = os.path.join(corrs_save_dir, f'{args.roi}.npy')
np.save(corrs_path, correlations)
print(f"Saved: {corrs_path} (shape {correlations.shape})")

weights_save_dir = os.path.join(args.weights_save_dir, f'fmri_sub-{args.fmri_subject:02d}')
os.makedirs(weights_save_dir, exist_ok=True)
weights_path = os.path.join(weights_save_dir, f'{args.roi}.npy')
np.save(weights_path, {'coef': coefs, 'intercept': intercepts})
print(f"Saved: {weights_path} (coef {coefs.shape}, intercept {intercepts.shape})")

del fmri_train, fmri_test, fmri_test_z, meg_train
gc.collect()

print(f"\nExecution complete! Total time: {time.time() - start_time:.2f} seconds.")