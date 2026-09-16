"""
Whole-brain MEG-fMRI searchlight RSA fusion for THINGS, one timepoint split per job. Each split
covers 50 consecutive of the 141 total timepoints (tmax=0.6), split s covering original
timepoints [s * 50, s * 50 + 50) (the last split is shorter -- 141 isn't a multiple of 50). Each
of those timepoints is processed, and saved, exactly as before: at each fMRI voxel's searchlight
neighborhood, computes the Spearman correlation between the fMRI RDM and the MEG RDM at that
timepoint -- the MEG RDM is the average across the 4 MEG subjects' RDMs.

Parameters
----------
subject : fMRI participant (1-3).
eeg_rdm_metric : {'pearsonr', 'crossnobis', 'decoding_accuracy'} -- which precomputed MEG RDM to use.
time_point_split : time point split index. Each split covers 50 of the 141 total timepoints,
    mapped to original timepoints [time_point_split * 50, time_point_split * 50 + 50).
radius : searchlight radius (mm), must match the fMRI searchlight RDM file.
chunk_size : voxels per joblib worker task.
n_jobs : parallel workers.
"""

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import time

import numpy as np
from scipy.stats import rankdata
from joblib import Parallel, delayed

start_time = time.time()
seed = 8
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--eeg_rdm_metric', type=str, default='pearsonr',
                     choices=['pearsonr', 'crossnobis', 'decoding_accuracy'])
parser.add_argument('--time_point_split', type=int, default=0,
                     help='Time point split index; each split covers 50 of the 141 total timepoints.')
parser.add_argument('--radius', type=float, default=10.0)
parser.add_argument('--meg_subjects', type=int, nargs='+', default=[1, 2, 3, 4])
parser.add_argument('--chunk_size', type=int, default=2000)
parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()

print('>>> Parallel MEG-fMRI Searchlight RSA Fusion (single timepoint) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =============================================================================
# 0. Map the time point split to its original (0-140) time point indices
# =============================================================================
N_TIME_POINTS_TOTAL = 141
SPLIT_SIZE = 50
start_time_point = args.time_point_split * SPLIT_SIZE
end_time_point = min(start_time_point + SPLIT_SIZE, N_TIME_POINTS_TOTAL)
assert start_time_point < N_TIME_POINTS_TOTAL, \
    f"time_point_split {args.time_point_split} maps to a start index of {start_time_point}, " \
    f"which is beyond the {N_TIME_POINTS_TOTAL} total time points."
print(f"\nTime point split {args.time_point_split} -> original time points "
      f"[{start_time_point}, {end_time_point})")

# =============================================================================
# 1. fMRI searchlight RDM file (metadata + chunk layout, static across time points)
# =============================================================================
# f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/searchlight_rdms/sub-{args.subject:02d}/f'searchlight_rdms_r-{args.radius}.npy''
fmri_npy_file = (
    f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/searchlight_rdms/'
    f'sub-{args.subject:02d}/searchlight_rdms_r-{args.radius}.npy'
)
fmri_mmap = np.load(fmri_npy_file, mmap_mode='r')
n_voxels, n_pairs_fmri = fmri_mmap.shape
print(f"fMRI RDMs on disk: {n_voxels} voxels x {n_pairs_fmri} pairs (per-chunk, in workers)")


def compute_chunk_spearman(start_v, end_v, fmri_npy_file, meg_centered, meg_norm):
    fmri_mmap = np.load(fmri_npy_file, mmap_mode='r')
    fmri_chunk = np.array(fmri_mmap[start_v:end_v, :])  # (chunk_len, n_pairs)

    fmri_ranked = rankdata(fmri_chunk, axis=1)
    fmri_centered = fmri_ranked - fmri_ranked.mean(axis=1, keepdims=True)
    fmri_norms = np.linalg.norm(fmri_centered, axis=1)

    denom = fmri_norms * meg_norm
    denom[denom == 0] = np.nan

    numerator = fmri_centered @ meg_centered
    chunk_corrs = (numerator / denom).astype(np.float32)

    return start_v, end_v, chunk_corrs


# =============================================================================
# 2. Vertex chunks (static across time points, so computed once for the whole split)
# =============================================================================
chunks = []
for start in range(0, n_voxels, args.chunk_size):
    end = min(start + args.chunk_size, n_voxels)
    chunks.append((start, end))

# =============================================================================
# 3. Loop over the timepoints covered by this split
# =============================================================================
data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'
save_dir = (
    f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/correlations/searchlight_fusion/'
    f'eeg_rdm_metric-{args.eeg_rdm_metric}/radius-{args.radius}/subject-{args.subject:02d}'
)
os.makedirs(save_dir, exist_ok=True)

for time_point in range(start_time_point, end_time_point):

    # --- Load and average the MEG subjects' RDMs at this timepoint ---
    meg_rdm_sum = np.zeros(4950, dtype=np.float32)  # 4950 is the number of unique pairwise distances corresponding to 100 stimuli
    # f"pearsonr_rdm_meg_sub-P{msub}.npy"
    for msub in args.meg_subjects:
        meg_rdm_sum += np.load(os.path.join(data_dir, f"{args.eeg_rdm_metric}_rdm_meg_sub-P{msub}.npy"))[time_point]

    meg_rdm = (meg_rdm_sum / len(args.meg_subjects)).astype(np.float32)
    n_pairs = meg_rdm.shape[0]

    assert n_pairs_fmri == n_pairs, f"Pair count mismatch: fMRI has {n_pairs_fmri}, MEG has {n_pairs}"
    print(f"\nMEG RDM ({args.eeg_rdm_metric}), averaged over {len(args.meg_subjects)} subjects, "
          f"time point {time_point}: {meg_rdm.shape}")

    meg_ranked = rankdata(meg_rdm)
    meg_centered = meg_ranked - meg_ranked.mean()
    meg_norm = np.linalg.norm(meg_centered)

    # --- Dispatch chunks to the joblib pool ---
    print(f"Dispatching {len(chunks)} voxel chunks to Joblib pool for time point {time_point}...")

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(compute_chunk_spearman)(start, end, fmri_npy_file, meg_centered, meg_norm)
        for start, end in chunks
    )

    searchlight_corrs = np.zeros(n_voxels, dtype=np.float32)
    for start, end, chunk_data in results:
        searchlight_corrs[start:end] = chunk_data

    # --- Saving results ---
    file_name = f'time_point_{time_point:04d}.npy'
    np.save(os.path.join(save_dir, file_name), searchlight_corrs)

    print(f"Searchlight complete for time point {time_point}")
    print(f"Results saved to: {os.path.join(save_dir, file_name)}")

execution_time = time.time() - start_time
print(f"\nSearchlight complete for time point split {args.time_point_split} "
      f"(original time points {start_time_point}-{end_time_point - 1})")
print(f"Total Execution time: {execution_time:.2f} seconds.")