"""
Stimulus Feature Encoding Fusion, Phase 1 ROI-level.


-----------------------------
Parameters:
- fmri_subject: The THINGS-fMRI subject (1, 2, or 3) for which the encoding
  fusion is performed.
- half: Which half of the training set to fit on (1 or 2). Half 1 = the
  first 4320 training stimuli in stimulus order; half 2 = the last 4320.
  Not a random split.
- roi: ROI name (e.g. 'V1'), matching a key in the precomputed
  metadata_fmri['roi'] dict for this subject.
- ncsnr_threshold: Noise-ceiling threshold. Only ROI voxels whose noise
  ceiling exceeds this value are kept.
- data_dir: Directory containing the pre-computed fmri_sub-XX.npy,
  meg_train.npy, and meg_test.npy files (from
  THINGS_Precompute_MEG_fMRI_Data.py).
- save_dir: Directory where the per-half regression_weights.npy (and
  voxel_idx.npy) files are saved, under a
  fmri_sub-XX/roi-<ROI>/half-{1,2} subfolder.
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
from utils import load_fmri_roi_data, load_meg_data, fit_timepoint_weights

start_time = time.time()

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)  # 1, 2, or 3
parser.add_argument('--half', type=int, default=1)  # 1 or 2
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--ncsnr_threshold', type=float, default=20.0)
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/regression_weights/joint_meg_fmri_fusion/phase_1/roi')
parser.add_argument('--n_jobs', type=int, default=-1)
parser.add_argument('--tmax', type=float, default=0.6)
args = parser.parse_args()

assert args.half in (1, 2), f"--half must be 1 or 2, got {args.half}"

print('>>> Stimulus Feature Encoding Fusion -- Phase 1 (ROI) <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

seed = 8
np.random.seed(seed)

# =============================================================================
# Loading this run's fMRI (ROI + ncsnr filtered) and MEG data (train only --
# no test data needed for Phase 1)
# =============================================================================
fmri_train_roi, _, voxel_idx = load_fmri_roi_data(args.fmri_subject, args.roi, args.ncsnr_threshold)
meg_train_full, _ = load_meg_data(tmax=args.tmax)
n_time = meg_train_full.shape[2]
n_voxels = fmri_train_roi.shape[1]

print(f"\n[fMRI sub-{args.fmri_subject:02d}, ROI {args.roi}, ncsnr > "
      f"{args.ncsnr_threshold}] train {fmri_train_roi.shape} ({n_voxels} voxels)")
print(f"[MEG] train {meg_train_full.shape}, {n_time} timepoints")

if n_voxels == 0:
    print(f"No vertices/voxels above the noise ceiling threshold "
          f"({args.ncsnr_threshold}) in ROI '{args.roi}' (subject "
          f"{args.fmri_subject}). Nothing to fit -- exiting.")
    raise SystemExit(0)

assert meg_train_full.shape[0] == fmri_train_roi.shape[0], \
    f"sub-{args.fmri_subject:02d}: MEG train ({meg_train_full.shape[0]}) / fMRI train " \
    f"({fmri_train_roi.shape[0]}) stimulus count mismatch."

# =============================================================================
# Split the training set into 2 halves of 4320 stimuli.
# =============================================================================
n_train = fmri_train_roi.shape[0]
assert n_train == 8640, \
    f"Expected 8640 THINGS training stimuli for sub-{args.fmri_subject:02d}, got {n_train}. "
half_size = n_train // 2  # 4320

if args.half == 1:
    half_idx = slice(0, half_size)
else:
    half_idx = slice(half_size, n_train)

fmri_train_half = fmri_train_roi[half_idx]
meg_train_half = meg_train_full[half_idx]

print(f"\nHalf {args.half}: {fmri_train_half.shape[0]} training stimuli "
      f"(fmri {fmri_train_half.shape}, meg {meg_train_half.shape})")

# =============================================================================
# Fitting, parallelized across timepoints -- weights only, no prediction/correlation
# =============================================================================
results = Parallel(n_jobs=args.n_jobs)(
    delayed(fit_timepoint_weights)(t, meg_train_half, fmri_train_half)
    for t in range(n_time)
)
coefs = np.array([r[0] for r in results], dtype=np.float32)      # (n_time, n_voxels, n_channels)
intercepts = np.array([r[1] for r in results], dtype=np.float32)  # (n_time, n_voxels)

# =============================================================================
# Saving
# =============================================================================
save_dir = os.path.join(
    args.save_dir, f'fmri_sub-{args.fmri_subject:02d}', f'roi-{args.roi}', f'half-{args.half}'
)
os.makedirs(save_dir, exist_ok=True)

weights_path = os.path.join(save_dir, 'regression_weights.npy')
np.save(weights_path, {'coef': coefs, 'intercept': intercepts})
print(f"Saved: {weights_path} (coef {coefs.shape}, intercept {intercepts.shape})")

voxel_idx_path = os.path.join(save_dir, 'voxel_idx.npy')
np.save(voxel_idx_path, voxel_idx)
print(f"Saved: {voxel_idx_path} (whole-brain indices of the {n_voxels} selected voxels)")

del fmri_train_roi, meg_train_full, fmri_train_half, meg_train_half
gc.collect()

print(f"\nExecution complete! Total time: {time.time() - start_time:.2f} seconds.")