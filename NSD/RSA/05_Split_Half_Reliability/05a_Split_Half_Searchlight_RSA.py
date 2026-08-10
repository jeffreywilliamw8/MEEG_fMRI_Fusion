"""
Split-Half Searchlight RSA Fusion (single timepoint, per-job architecture)

Split-half reliability for RSA fusion WITHOUT ever recomputing EEG or fMRI RDMs on split data.
Both the EEG RDM (n_pairs,) and every vertex's fMRI RDM (n_vertices, n_pairs) are already fully
computed over all 515 test conditions. Splitting conditions into two halves just means selecting
the subset of pairwise whose both conditions fall in that half, out of the one full RDM
already sitting on disk -- e.g. for conditions A,B,C,D,E,F split into {A,B,C} / {D,E,F}, half 1 keeps
{AB,AC,BC} and half 2 keeps {DE,DF,EF} out of the full 15-entry RDM; the cross-block (AD, AE, ...)
belongs to neither half and is dropped. No new RDM computation, just indexing.

The condition split itself is NOT generated here -- it's loaded from the shared split file
produced by Generate_Test_Split_Indices.py, the same file the encoding-fusion split-half
script uses. That's what guarantees RSA and encoding split-half reliability are computed
over identical condition partitions of the testing set per shuffle, so the two are directly comparable.


Parameters
----------
subject : int
    The NSD participant pair ID for which the searchlight fusion is performed.
hemisphere : str
    The hemisphere of the brain to analyze ('lh' for left hemisphere, 'rh' for right hemisphere).
time_point : int
    The time point index for which to compute the searchlight fusion.
n_neighbours : int
    The number of nearest neighbors to include in the searchlight for each vertex.
chunk_size : int
    The number of vertices to process in each parallel job.
n_jobs : int
    The number of parallel jobs to run (-1 for all available cores). 

n_shuffles: int
    The number of times the data is randomly shuffled before being split in 2


"""

import numpy as np
import os
import argparse
import time
from scipy.stats import rankdata
import h5py

# --- Set Thread Caps BEFORE importing joblib/scipy to prevent core thrashing ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from joblib import Parallel, delayed

start_time = time.time()
seed = 8
np.random.seed(seed)

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh')
parser.add_argument('--time_point', type=int, default=0)
parser.add_argument('--n_neighbours', type=int, default=100)
parser.add_argument('--chunk_size', type=int, default=1974, help='Number of vertices per joblib worker task (which makes for 83 chunks for 163842 vertices).')
parser.add_argument('--n_shuffles', type=int, default=100,
                     help='Number of independent random condition-split shuffles. Must match the '
                          '--n_shuffles used to generate --split_file (Generate_Test_Split_Indices.py) '
                          'and the value used by the encoding-fusion split-half script.')
parser.add_argument('--split_seed', type=int, default=8,
                     help='Seed used when generating the shared test-condition split file. Only '
                          'used to build the default --split_file path below.')
parser.add_argument('--split_file', type=str, default=None,
                     help='Path to the shared (n_shuffles, n_test) condition split-label file '
                          'produced by 00_Generate_Test_Split_Indices.py. If not given, defaults to '
                          'the standard shared_splits path for the observed n_stim, --n_shuffles, '
                          'and --split_seed. This is the SAME file the encoding-fusion split-half '
                          'script loads, guaranteeing both analyses partition conditions identically.')
parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()

print('>>> Parallel Searchlight RSA Fusion -- Split-Half Reliability (single timepoint) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =============================================================================
# 1. Loading the full EEG RDM for this timepoint only (cheap: a single length-n_pairs vector)
# =============================================================================
# /scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlation_rdms/correlation_rdm_eeg_sub-8.npy
#
data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlation_rdms'
eeg_rdm = np.load(os.path.join(data_dir, f"correlation_rdm_eeg_sub-{args.subject}.npy"))[args.time_point]  # (n_pairs,)
n_pairs = eeg_rdm.shape[0]

# Recover n_stim from n_pairs = n_stim*(n_stim-1)/2, and get the (row, col) condition pair for
# every position in the flattened RDM vector
n_stim = round((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
assert n_stim * (n_stim - 1) // 2 == n_pairs, f"n_pairs={n_pairs} is not a valid C(n,2) for any integer n"
rows, cols = np.triu_indices(n_stim, k=1)  # both shape (n_pairs,), rows[k]/cols[k] = condition pair for entry k

# =============================================================================
# 2. Load the shared condition-split file (NOT regenerated here -- see module docstring)
# =============================================================================
split_file = args.split_file

if split_file is None:
    split_file = os.path.join(
        '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/shared_splits',
        f'test_split_shuffles_ntest-{n_stim}_nshuffles-{args.n_shuffles}_seed-{args.split_seed}.npy'
    )

if not os.path.exists(split_file):
    raise FileNotFoundError(
        f"Shared condition split file not found at: {split_file}\n"
        f"Run 00_Generate_Test_Split_Indices.py --n_test {n_stim} --n_shuffles {args.n_shuffles} "
        f"--seed {args.split_seed} first, so this script and the encoding split-half script use "
        f"identical condition splits."
    )

split_labels = np.load(split_file)  # (n_shuffles, n_stim), 0 = half 1, 1 = half 2
if split_labels.shape != (args.n_shuffles, n_stim):
    raise ValueError(
        f"Loaded split file shape {split_labels.shape} doesn't match expected "
        f"({args.n_shuffles}, {n_stim}). Regenerate it with matching --n_test/--n_shuffles."
    )
print(f"\nLoaded shared condition split file: {split_file}")

# =============================================================================
# 3. For every (shuffle, half), build the pair-mask selecting only within-half entries out of
# the full n_pairs-length RDM, and rank/center/norm the EEG side ONCE per (shuffle, half) here
# in the main process -- these are small, reused identically by every vertex chunk below.
# =============================================================================
eeg_halves = []  # list of (shuffle_idx, half_idx, pair_mask, eeg_centered, eeg_norm)
for s in range(args.n_shuffles):
    half_condition_sets = [np.where(split_labels[s] == 0)[0], np.where(split_labels[s] == 1)[0]]
    for h, half_conditions in enumerate(half_condition_sets):
        membership = np.zeros(n_stim, dtype=bool)
        membership[half_conditions] = True
        pair_mask = membership[rows] & membership[cols]  # True only where BOTH conditions are in this half

        eeg_sub = eeg_rdm[pair_mask]
        eeg_ranked = rankdata(eeg_sub)
        eeg_centered = eeg_ranked - eeg_ranked.mean()
        eeg_norm = np.linalg.norm(eeg_centered)

        eeg_halves.append((s, h, pair_mask, eeg_centered, eeg_norm))
        print(f"  Shuffle {s}, half {h}: {len(half_conditions)} conditions -> "
              f"{pair_mask.sum()} within-half pairs")

# =============================================================================
# 4. fMRI RDM file handle 
# =============================================================================
fmri_h5_file = os.path.join(
    f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/fmri_searchlight_rdms/n_neighbours-{args.n_neighbours}',
    f'fmri_sub-{args.subject}_hemi-{args.hemisphere}_rdms.h5'
)
with h5py.File(fmri_h5_file, 'r') as f:
    n_vertices, n_pairs_fmri = f['rdms'].shape
assert n_pairs_fmri == n_pairs, f"Pair count mismatch: fMRI has {n_pairs_fmri}, EEG has {n_pairs}"
print(f"\nfMRI RDMs on disk: {n_vertices} vertices x {n_pairs_fmri} pairs (read lazily, per-chunk, in workers)")

# =============================================================================
# Parallel Worker: reads its OWN chunk directly from disk ONCE (never materializes the full
# fMRI array anywhere, and never re-reads the same chunk for different shuffles/halves), then
# for every (shuffle, half) combo, subselects that combo's pair-mask columns out of the SAME
# in-memory chunk and computes the vectorized rank correlation against that combo's EEG sub-vector.
# =============================================================================
def compute_chunk_spearman_split_half(start_v, end_v, fmri_h5_file, eeg_halves):
    with h5py.File(fmri_h5_file, 'r') as f:
        fmri_chunk_full = f['rdms'][start_v:end_v, :]  # (chunk_len, n_pairs) -- read once per chunk

    chunk_len = fmri_chunk_full.shape[0]
    n_combos = len(eeg_halves)
    chunk_corrs = np.zeros((n_combos, chunk_len), dtype=np.float32)

    for combo_idx, (shuffle_idx, half_idx, pair_mask, eeg_centered, eeg_norm) in enumerate(eeg_halves):
        fmri_sub = fmri_chunk_full[:, pair_mask]  # (chunk_len, sub_n_pairs) -- this combo's within-half pairs only

        fmri_ranked = rankdata(fmri_sub, axis=1)                             # (chunk_len, sub_n_pairs)
        fmri_centered = fmri_ranked - fmri_ranked.mean(axis=1, keepdims=True)
        fmri_norms = np.linalg.norm(fmri_centered, axis=1)                   # (chunk_len,)

        denom = fmri_norms * eeg_norm
        denom[denom == 0] = np.nan  # guard against degenerate (zero-variance) searchlights

        numerator = fmri_centered @ eeg_centered  # (chunk_len,) -- one BLAS matvec for the whole chunk
        chunk_corrs[combo_idx, :] = (numerator / denom).astype(np.float32)

    return start_v, end_v, chunk_corrs  # (n_shuffles * 2, chunk_len)


# =============================================================================
# 5. Dispatch chunks to the joblib pool -- one task per vertex chunk, each task internally
# looping over all (shuffle, half) combos against its own single chunk read.
# =============================================================================
chunks = []
for start in range(0, n_vertices, args.chunk_size):
    end = min(start + args.chunk_size, n_vertices)
    chunks.append((start, end))

print(f"\nDispatching {len(chunks)} vertex chunks to Joblib pool for time point {args.time_point} "
      f"({args.n_shuffles} shuffles x 2 halves = {len(eeg_halves)} combos per chunk)...")

results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
    delayed(compute_chunk_spearman_split_half)(start, end, fmri_h5_file, eeg_halves)
    for start, end in chunks
)

print("Assembling results into (n_shuffles, 2, n_vertices) array...")
searchlight_corrs = np.zeros((args.n_shuffles, 2, n_vertices), dtype=np.float32)
for start, end, chunk_data in results:
    chunk_len = end - start
    searchlight_corrs[:, :, start:end] = chunk_data.reshape(args.n_shuffles, 2, chunk_len)

# =============================================================================
# 6. Saving Results (separate results tree from the point-estimate searchlight fusion results)
# =============================================================================
save_dir = (
    f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/'
    f'split_half_reliability/n_neighbours-{args.n_neighbours}/'
    f'subject-{args.subject}/{args.hemisphere}_hemisphere'
)
os.makedirs(save_dir, exist_ok=True)

file_name = f'time_point_{args.time_point:04d}.npy'
np.save(os.path.join(save_dir, file_name), searchlight_corrs)

execution_time = time.time() - start_time
print(f"\nSplit-half searchlight complete for time point {args.time_point}")
print(f"Results saved to: {os.path.join(save_dir, file_name)}, shape={searchlight_corrs.shape}")
print(f"Total Execution time: {execution_time:.2f} seconds.")