"""
Stimulus encoding fusion, Phase 1.
Train MEG-to-fMRI regression models on half of the MEG training set
(4320 samples)
-----------------------------
Parameters:
- fmri_subject: The THINGS-fMRI subject (1, 2, or 3) for which the encoding
  fusion is performed.
- half: Which half of the training set to fit on (1 or 2). Half 1 = the
  first 4320 training stimuli in stimulus order; half 2 = the last 4320.
  Not a random split.
- fmri_split: The split index for fMRI voxels. The fMRI data is split into
  n_splits chunks via np.array_split to cover all voxels (211,339 / 226,950
  / 189,164 for subjects 1/2/3 respectively) -- same shared scheme as
  THINGS_Whole_Brain_Encoding_Fusion.py.
- n_splits: The number of voxel splits the whole-brain fMRI data is divided
  into (same value used for every subject; chunk size varies slightly by
  subject since voxel counts differ). Default 500.
- data_dir: Directory containing the pre-computed fmri_sub-XX.npy,
  meg_train.npy, and meg_test.npy files (from
  THINGS_Precompute_MEG_fMRI_Data.py).
- save_dir: Directory where the per-split regression_weights.npy file is
  saved, under a fmri_sub-XX/half-{1,2} subfolder.
- n_jobs: The number of parallel jobs to run (using joblib) across
  timepoints. Set to -1 to use all available cores.
- tmax: Upper time bound (seconds post stimulus onset) the MEG data is
  truncated to (passed through to THINGS_utils.load_meg_data). Default 0.6.
"""

import argparse
import os
import time
import gc
import numpy as np
from joblib import Parallel, delayed
from utils import load_fmri_wb_data, load_meg_data, fit_timepoint_weights

start_time = time.time()

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)  # 1, 2, or 3
parser.add_argument('--half', type=int, default=1)  # 1 or 2
parser.add_argument('--fmri_split', type=int, default=1)  # 1 to n_splits
parser.add_argument('--n_splits', type=int, default=500)
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--save_dir', type=str, default = '/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/regression_weights/stimulus_feature_encoding_fusion/phase_1/wb')
parser.add_argument('--n_jobs', type=int, default=-1)
parser.add_argument('--tmax', type=float, default=0.6)
args = parser.parse_args()


print('>>> Stimulus Feature Encoding Fusion -- Phase 1 <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

seed = 8
np.random.seed(seed)

# =============================================================================
# Loading this run's fMRI and MEG data (train only -- no test data needed)
# =============================================================================
fmri_train_full, _ = load_fmri_wb_data(args.fmri_subject)
meg_train_full, _ = load_meg_data(tmax=args.tmax)
n_time = meg_train_full.shape[2]

print(f"\n[fMRI sub-{args.fmri_subject:02d}] full-brain train {fmri_train_full.shape}")
print(f"[MEG] train {meg_train_full.shape}, {n_time} timepoints")


# =============================================================================
# Split the training set into 2 halves of 4320 stimuli
# =============================================================================
n_train = fmri_train_full.shape[0]
half_size = n_train // 2  # 4320

if args.half == 1:
    half_idx = slice(0, half_size)
else:
    half_idx = slice(half_size, n_train)

fmri_train_half = fmri_train_full[half_idx]
meg_train_half = meg_train_full[half_idx]

print(f"\nHalf {args.half}: {fmri_train_half.shape[0]} training stimuli "
      f"(fmri {fmri_train_half.shape}, meg {meg_train_half.shape})")


fmri_train_splits = np.array_split(fmri_train_half, args.n_splits, axis=1)
fmri_train = fmri_train_splits[args.fmri_split - 1]
n_voxels = fmri_train.shape[1]

print(f"Split {args.fmri_split}/{args.n_splits}: {n_voxels} voxels (train {fmri_train.shape})")

# =============================================================================
# Fitting, parallelized across timepoints
# =============================================================================
results = Parallel(n_jobs=args.n_jobs)(
    delayed(fit_timepoint_weights)(t, meg_train_half, fmri_train)
    for t in range(n_time)
)
coefs = np.array([r[0] for r in results], dtype=np.float32)      # (n_time, n_voxels, n_channels)
intercepts = np.array([r[1] for r in results], dtype=np.float32)  # (n_time, n_voxels)

# =============================================================================
# Saving
# =============================================================================
save_dir = os.path.join(args.save_dir, f'fmri_sub-{args.fmri_subject:02d}', f'half-{args.half}')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'fmri_split-{args.fmri_split:03d}.npy')
np.save(save_path, {'coef': coefs, 'intercept': intercepts})
print(f"Saved: {save_path} (coef {coefs.shape}, intercept {intercepts.shape})")

del fmri_train_full, meg_train_full, fmri_train_half, meg_train_half, fmri_train
gc.collect()

print(f"\nExecution complete! Total time: {time.time() - start_time:.2f} seconds.")