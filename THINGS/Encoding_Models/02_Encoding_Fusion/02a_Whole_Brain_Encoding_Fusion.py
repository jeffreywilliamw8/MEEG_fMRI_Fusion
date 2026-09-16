"""
Whole-brain MEG-to-fMRI encoding fusion, one script for each THINGS-fMRI
subject (pass --fmri_subject)

-----------------------------
Parameters:
- fmri_subject: The THINGS-fMRI subject (1, 2, or 3) for which the encoding
  fusion is performed.
- fmri_split: The split index for fMRI voxels. The fMRI data is split into
  n_splits chunks via np.array_split to cover all voxels (211,339 / 226,950
  / 189,164 for subjects 1/2/3 respectively).
- n_splits: The number of voxel splits the whole-brain fMRI data is divided
  into (same value used for every subject; chunk size varies slightly by
  subject since voxel counts differ). Default 500.
- data_dir: Directory containing the pre-computed fmri_sub-XX.npy,
  meg_train.npy, and meg_test.npy files (from
  Precompute_THINGS_MEG_fMRI_Data.py).
- save_dir: Directory where the per-split correlations.npy file is saved,
  under a fmri_sub-XX subfolder.
- n_jobs: The number of parallel jobs to run (using joblib) across
  timepoints. Set to -1 to use all available cores.
- tmax: Upper time bound (seconds post stimulus onset) the MEG data is
  truncated to (passed through to utils.load_meg_data). Default 0.6.
"""

import argparse
import os
import time
import gc
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits
from utils import load_fmri_wb_data, load_meg_data

start_time = time.time()

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)  # 1, 2, or 3
parser.add_argument('--fmri_split', type=int, default=1)  # 1 to n_splits
parser.add_argument('--n_splits', type=int, default=500)
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')

parser.add_argument('--save_dir', type=str, default='/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/correlations/whole_brain_encoding_fusion')
parser.add_argument('--n_jobs', type=int, default=-1)
parser.add_argument('--tmax', type=float, default=0.6)
args = parser.parse_args()

print('>>> MEG-fMRI Encoding Fusion (Whole-Brain) <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

seed = 8
np.random.seed(seed)

eps = 1e-8
alphas = np.logspace(-6, 10, 17)

# =============================================================================
# Per-timepoint fit/predict/correlate, run in parallel across timepoints
# =============================================================================
def fit_predict_correlate_timepoint(t, meg_train, meg_test, fmri_train, fmri_test_z):
    with threadpool_limits(limits=1):
        meg2fmri = RidgeCV(alphas=alphas, alpha_per_target=True)
        meg2fmri.fit(meg_train[:, :, t], fmri_train)

        # Predict via explicit matrix multiplication instead of .predict()
        t_fmri = meg_test[:, :, t] @ meg2fmri.coef_.T + meg2fmri.intercept_

        # Center/normalize, then correlate via matrix multiplication
        t_fmri_z = (t_fmri - t_fmri.mean(0)) / (t_fmri.std(0) + eps)
        corr_t = (t_fmri_z * fmri_test_z).mean(axis=0)
    return corr_t


# =============================================================================
# Loading this run's fMRI and MEG data
# =============================================================================
fmri_train_full, fmri_test_full = load_fmri_wb_data(args.fmri_subject)
meg_train, meg_test = load_meg_data(tmax=args.tmax)
n_time = meg_train.shape[2]

print(f"\n[fMRI sub-{args.fmri_subject:02d}] full-brain train {fmri_train_full.shape}, "
      f"test {fmri_test_full.shape}")
print(f"[MEG] train {meg_train.shape}, test {meg_test.shape}, {n_time} timepoints")

assert meg_train.shape[0] == fmri_train_full.shape[0], \
    f"sub-{args.fmri_subject:02d}: MEG train ({meg_train.shape[0]}) / fMRI train " \
    f"({fmri_train_full.shape[0]}) stimulus count mismatch."
assert meg_test.shape[0] == fmri_test_full.shape[0], \
    f"sub-{args.fmri_subject:02d}: MEG test ({meg_test.shape[0]}) / fMRI test " \
    f"({fmri_test_full.shape[0]}) stimulus count mismatch."


fmri_train_splits = np.array_split(fmri_train_full, args.n_splits, axis=1)
fmri_test_splits = np.array_split(fmri_test_full, args.n_splits, axis=1)
fmri_train = fmri_train_splits[args.fmri_split - 1]
fmri_test = fmri_test_splits[args.fmri_split - 1]
n_voxels = fmri_train.shape[1]

print(f"Split {args.fmri_split}/{args.n_splits}: {n_voxels} voxels "
      f"(train {fmri_train.shape}, test {fmri_test.shape})")

fmri_test_z = (fmri_test - fmri_test.mean(0)) / (fmri_test.std(0) + eps)

# =============================================================================
# Fitting+predicting+correlating, parallelized across timepoints
# =============================================================================
results = Parallel(n_jobs=args.n_jobs)(
    delayed(fit_predict_correlate_timepoint)(
        t, meg_train, meg_test, fmri_train, fmri_test_z
    )
    for t in range(n_time)
)
correlations = np.array(results, dtype=np.float32)  # (n_time, n_voxels)

# =============================================================================
# Saving
# =============================================================================
save_dir = os.path.join(args.save_dir, f'fmri_sub-{args.fmri_subject:02d}')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'fmri_split-{args.fmri_split:03d}.npy')
np.save(save_path, correlations)
print(f"Saved: {save_path} (shape {correlations.shape})")

del fmri_train_full, fmri_test_full, fmri_train, fmri_test, fmri_test_z, meg_train
gc.collect()

print(f"\nExecution complete! Total time: {time.time() - start_time:.2f} seconds.")