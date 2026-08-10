"""

This script performs the whole-brain EEG-fMRI fusion in a searchlight fashion:
At each time point, it computes the Spearman correlation between the EEG RDM and the fMRI RDMs for each vertex's searchlight neighborhood.
The fMRI RDMs are pre-computed and stored in an HDF5 file, and for parallelization on the HPC, each instance runs the analysis for a separate time point

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
    The number of vertices to process in each parallel joblib job.
n_jobs : int
    The number of parallel jobs to run (-1 for all available cores).   
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
parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()

print('>>> Parallel Searchlight RSA Fusion (single timepoint, per-job architecture) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =============================================================================
# 1. Loading the EEG RDM for this timepoint only
# =============================================================================
data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlation_rdms'
eeg_rdm = np.load(os.path.join(data_dir, f"correlation_rdm_eeg_sub-{args.subject}.npy"))[args.time_point]  # (n_pairs,)
n_pairs = eeg_rdm.shape[0]

# Rank-transform the EEG side once for this job (it's reused, unranked, by every vertex/chunk below)
eeg_ranked = rankdata(eeg_rdm)
eeg_centered = eeg_ranked - eeg_ranked.mean()
eeg_norm = np.linalg.norm(eeg_centered)

# =============================================================================
# 2. fMRI RDM file handle (NOT loaded into memory here -- workers read their own chunks)
# =============================================================================
fmri_h5_file = os.path.join(
    f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/fmri_searchlight_rdms/n_neighbours-{args.n_neighbours}',
    f'fmri_sub-{args.subject}_hemi-{args.hemisphere}_rdms.h5'
)
with h5py.File(fmri_h5_file, 'r') as f:
    n_vertices, n_pairs_fmri = f['rdms'].shape
assert n_pairs_fmri == n_pairs, f"Pair count mismatch: fMRI has {n_pairs_fmri}, EEG has {n_pairs}"
print(f"fMRI RDMs on disk: {n_vertices} vertices x {n_pairs_fmri} pairs (read lazily, per-chunk, in workers)")

# =============================================================================
# Parallel Worker: reads its own chunk directly from disk,
# then computes vectorized rank correlation for this timepoint.
# =============================================================================
def compute_chunk_spearman(start_v, end_v, fmri_h5_file, eeg_centered, eeg_norm):
    with h5py.File(fmri_h5_file, 'r') as f:
        fmri_chunk = f['rdms'][start_v:end_v, :]  # (chunk_len, n_pairs) -- only this slice is ever read

    # Vectorized rank correlation: rank + center the whole chunk in one shot,
    # then a single matrix-vector product replaces chunk_len individual spearmanr() calls.
    fmri_ranked = rankdata(fmri_chunk, axis=1)                              # (chunk_len, n_pairs)
    fmri_centered = fmri_ranked - fmri_ranked.mean(axis=1, keepdims=True)   # (chunk_len, n_pairs)
    fmri_norms = np.linalg.norm(fmri_centered, axis=1)                     # (chunk_len,)

    denom = fmri_norms * eeg_norm
    denom[denom == 0] = np.nan  # guard against degenerate (zero-variance) searchlights

    numerator = fmri_centered @ eeg_centered  # (chunk_len,) -- one BLAS matvec for the whole chunk
    chunk_corrs = (numerator / denom).astype(np.float32)

    return start_v, end_v, chunk_corrs


# =============================================================================
# 3. Dispatch chunks to the joblib pool
# =============================================================================
chunks = []
for start in range(0, n_vertices, args.chunk_size):
    end = min(start + args.chunk_size, n_vertices)
    chunks.append((start, end))

print(f"Dispatching {len(chunks)} vertex chunks to Joblib pool for time point {args.time_point}...")

results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
    delayed(compute_chunk_spearman)(start, end, fmri_h5_file, eeg_centered, eeg_norm)
    for start, end in chunks
)

print("Assembling results array...")
searchlight_corrs = np.zeros(n_vertices, dtype=np.float32)
for start, end, chunk_data in results:
    searchlight_corrs[start:end] = chunk_data

# =============================================================================
# 4. Saving Results
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/searchlight_fusion/eeg_rdm_metric-correlation/n_neighbours-{args.n_neighbours}/subject-{args.subject}/{args.hemisphere}_hemisphere'
os.makedirs(save_dir, exist_ok=True)

file_name = f'time_point_{args.time_point:04d}.npy'
np.save(os.path.join(save_dir, file_name), searchlight_corrs)

execution_time = time.time() - start_time
print(f"Searchlight complete for time point {args.time_point}")
print(f"Results saved to: {os.path.join(save_dir, file_name)}")
print(f"Total Execution time: {execution_time:.2f} seconds.")