"""
Split-half reliability of the searchlight MEG-fMRI RSA fusion, one timepoint
per job. Splitting the 100 test conditions into halves means selecting the
pairwise entries whose both conditions fall in the same half out of the
already-computed full RDMs -- no RDM is recomputed. Condition splits are
loaded from the shared file produced by THINGS_00_Generate_Test_Split_Indices.py,
so this and the encoding-fusion split-half analysis partition conditions
identically.

Parameters
----------
subject : fMRI participant (1-3).
eeg_rdm_metric : {'pearsonr', 'crossnobis', 'decoding_accuracy'} MEG RDM to use.
time_point : MEG timepoint index.
radius : searchlight radius (mm), must match the fMRI searchlight RDM file.
n_shuffles : condition-split shuffles; must match the shared split file.
chunk_size : voxels per joblib worker task.
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
parser.add_argument('--time_point', type=int, default=0)
parser.add_argument('--radius', type=float, default=10.0)
parser.add_argument('--n_shuffles', type=int, default=10)
parser.add_argument('--split_seed', type=int, default=8)
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--meg_subjects', type=int, nargs='+', default=[1, 2, 3, 4])
parser.add_argument('--chunk_size', type=int, default=2000)
parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()

print('>>> Searchlight RSA Fusion -- Split-Half Reliability (single timepoint) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =============================================================================
# 1. Load and average the MEG subjects' RDMs at this timepoint
# =============================================================================
data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'

meg_rdm_sum = np.zeros(4950, dtype=np.float32) # 4950 is the number of unique pairwise distances correspoding to 100 stimuli
for msub in args.meg_subjects:
    meg_rdm_sum += np.load(os.path.join(data_dir, f"{args.eeg_rdm_metric}_rdm_meg_sub-P{msub}.npy"))[args.time_point]


meg_rdm = (meg_rdm_sum / len(args.meg_subjects)).astype(np.float32)
n_pairs = meg_rdm.shape[0]
print(f"MEG RDM ({args.eeg_rdm_metric}), averaged over {len(args.meg_subjects)} subjects: {meg_rdm.shape}")

meg_ranked = rankdata(meg_rdm)
meg_centered = meg_ranked - meg_ranked.mean()
meg_norm = np.linalg.norm(meg_centered)


# =============================================================================
# 2. Shared condition-split file
# =============================================================================
split_file = args.split_file
n_stim=meg_rdm.shape[0]
rows, cols = np.triu_indices(n_stim, k=1)

if split_file is None:
    split_file = os.path.join(
        '/scratch/jeffreykatab/Projects/fusion/THINGS/shared_splits',
        f'test_split_shuffles_ntest-{n_stim}_nshuffles-{args.n_shuffles}_seed-{args.split_seed}.npy'
    )

if not os.path.exists(split_file):
    raise FileNotFoundError(
        f"Shared condition split file not found at: {split_file}\n"
        f"Run THINGS_00_Generate_Test_Split_Indices.py --n_test {n_stim} --n_shuffles "
        f"{args.n_shuffles} --seed {args.split_seed} first."
    )

split_labels = np.load(split_file)  # (n_shuffles, n_stim), 0 = half 1, 1 = half 2
if split_labels.shape != (args.n_shuffles, n_stim):
    raise ValueError(
        f"Loaded split file shape {split_labels.shape} doesn't match expected "
        f"({args.n_shuffles}, {n_stim}). Regenerate it with matching --n_test/--n_shuffles."
    )
print(f"\nLoaded shared condition split file: {split_file}")

# =============================================================================
# 3. Per (shuffle, half): pair-mask plus the ranked/centered MEG sub-vector
# =============================================================================
meg_halves = []
for s in range(args.n_shuffles):
    for h, half_conditions in enumerate([np.where(split_labels[s] == 0)[0],
                                          np.where(split_labels[s] == 1)[0]]):
        membership = np.zeros(n_stim, dtype=bool)
        membership[half_conditions] = True
        pair_mask = membership[rows] & membership[cols]

        meg_ranked = rankdata(meg_rdm[pair_mask])
        meg_centered = meg_ranked - meg_ranked.mean()
        meg_norm = np.linalg.norm(meg_centered)

        meg_halves.append((s, h, pair_mask, meg_centered, meg_norm))
        print(f"  Shuffle {s}, half {h}: {len(half_conditions)} conditions -> "
              f"{pair_mask.sum()} within-half pairs")

# =============================================================================
# 4. fMRI searchlight RDMs (memory-mapped, read per chunk in workers)
# =============================================================================
fmri_npy_file = (
    f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/searchlight_rdms/'
    f'sub-{args.subject:02d}/searchlight_rdms_r-{args.radius}.npy'
)
fmri_mmap = np.load(fmri_npy_file, mmap_mode='r')
n_voxels, n_pairs_fmri = fmri_mmap.shape
assert n_pairs_fmri == n_pairs, f"Pair count mismatch: fMRI has {n_pairs_fmri}, MEG has {n_pairs}"
print(f"\nfMRI searchlight RDMs: {n_voxels} voxels x {n_pairs_fmri} pairs")


def compute_chunk_spearman_split_half(start_v, end_v, fmri_npy_file, meg_halves):
    fmri_mmap = np.load(fmri_npy_file, mmap_mode='r')
    fmri_chunk_full = np.array(fmri_mmap[start_v:end_v, :])  # read once per chunk

    chunk_len = fmri_chunk_full.shape[0]
    chunk_corrs = np.zeros((len(meg_halves), chunk_len), dtype=np.float32)

    for combo_idx, (s, h, pair_mask, meg_centered, meg_norm) in enumerate(meg_halves):
        fmri_sub = fmri_chunk_full[:, pair_mask]

        fmri_ranked = rankdata(fmri_sub, axis=1)
        fmri_centered = fmri_ranked - fmri_ranked.mean(axis=1, keepdims=True)
        fmri_norms = np.linalg.norm(fmri_centered, axis=1)

        denom = fmri_norms * meg_norm
        denom[denom == 0] = np.nan

        chunk_corrs[combo_idx, :] = ((fmri_centered @ meg_centered) / denom).astype(np.float32)

    return start_v, end_v, chunk_corrs


# =============================================================================
# 5. Dispatch chunks
# =============================================================================
chunks = []
for start in range(0, n_voxels, args.chunk_size):
    end = min(start + args.chunk_size, n_voxels)
    chunks.append((start, end))

print(f"\nDispatching {len(chunks)} voxel chunks for time point {args.time_point} "
      f"({args.n_shuffles} shuffles x 2 halves = {len(meg_halves)} combos per chunk)...")

results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
    delayed(compute_chunk_spearman_split_half)(start, end, fmri_npy_file, meg_halves)
    for start, end in chunks
)

searchlight_corrs = np.zeros((args.n_shuffles, 2, n_voxels), dtype=np.float32)
for start, end, chunk_data in results:
    searchlight_corrs[:, :, start:end] = chunk_data.reshape(args.n_shuffles, 2, end - start)

# =============================================================================
# 6. Saving
# =============================================================================
save_dir = (
    f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/correlations/'
    f'searchlight_fusion_split_half_reliability/eeg_rdm_metric-{args.eeg_rdm_metric}/'
    f'radius-{args.radius}/subject-{args.subject:02d}'
)
os.makedirs(save_dir, exist_ok=True)

file_name = f'time_point_{args.time_point:04d}.npy'
np.save(os.path.join(save_dir, file_name), searchlight_corrs)

print(f"\nSplit-half searchlight complete for time point {args.time_point}")
print(f"Results saved to: {os.path.join(save_dir, file_name)}, shape={searchlight_corrs.shape}")
print(f"Total Execution time: {time.time() - start_time:.2f} seconds.")