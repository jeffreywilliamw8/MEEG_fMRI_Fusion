"""
Computes EEG RDMs using pairwise SVM decoding accuracy.
"""

import os
# --- Core Environment Safeguards against worker over-subscription ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import argparse
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
import random
import time

start_time = time.time()
seed = 8
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--n_jobs', type=int, default=-1, help='Parallel workers across time points (-1 = all available cores).')
args = parser.parse_args()

print(f'>>> Parallelized Image-Wise Decoding Analysis, full stimulus set (Sub-{args.subject}) <<<')

# Paths
data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
out_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/eeg_rdms'
os.makedirs(out_dir, exist_ok=True)

# Load data 
eeg_dict = np.load(os.path.join(data_dir, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True).item()
eeg_data = eeg_dict['eeg_test'].astype(np.float32)
print(f"EEG data shape: {eeg_data.shape} (Stimuli, Trials, Channels, Time)")
n_stim, n_trials, n_chan, n_time = eeg_data.shape

# 1. Pseudo-trials (6 pseudo-trials)
n_pseudo = 6
trials_per_pseudo = n_trials // n_pseudo
pseudo_data = eeg_data.reshape(n_stim, n_pseudo, trials_per_pseudo, n_chan, n_time).mean(axis=2)
print(f"Data reshaped to pseudo-trials: {pseudo_data.shape} (Stimuli, Pseudo-trials, Channels, Time)")

# 2. Setup pairs -- full upper triangle over all 515 stimuli (132,355 pairs)
rows, cols = np.triu_indices(n_stim, k=1)
n_pairs = len(rows)
n_pairs_expected = n_stim * (n_stim - 1) // 2
assert n_pairs == n_pairs_expected
print(f"Number of stimuli: {n_stim} -> {n_pairs} pairs")

# =============================================================================
# Precompute the label vector and the cross-validation fold split ONCE, globally: both are
# identical for every pair and every timepoint 
# =============================================================================
y = np.concatenate([np.zeros(n_pseudo), np.ones(n_pseudo)]).astype(np.int8)
cv_splits = list(
    StratifiedKFold(n_splits=n_pseudo, shuffle=True, random_state=seed).split(np.zeros_like(y), y)
)

# =============================================================================
# Joblib Parallel Worker Definition
# =============================================================================
def decode_single_timepoint(t, pseudo_data, rows, cols, y, cv_splits, n_pseudo, n_pairs, n_chan):
    """
    Worker handling all image-pair decodings for a single time point t.

    """
    current_data = pseudo_data[:, :, :, t]  # (n_stim, n_pseudo, n_chan)
    timepoint_rdm = np.empty(n_pairs, dtype=np.float32)

    clf = LinearSVC(C=1.0, max_iter=1000, tol=1e-3, dual=True, random_state=seed)
    X_buf = np.empty((2 * n_pseudo, n_chan), dtype=np.float32)
    fold_accs = np.empty(len(cv_splits), dtype=np.float32)

    for p_idx in range(n_pairs):
        i, j = rows[p_idx], cols[p_idx]
        X_buf[:n_pseudo] = current_data[i]
        X_buf[n_pseudo:] = current_data[j]

        for f_idx, (train_idx, test_idx) in enumerate(cv_splits):
            clf.fit(X_buf[train_idx], y[train_idx])
            fold_accs[f_idx] = np.mean(clf.predict(X_buf[test_idx]) == y[test_idx])

        timepoint_rdm[p_idx] = fold_accs.mean()

    return t, timepoint_rdm


# =============================================================================
# Run Parallel Engine
# =============================================================================
print(f"\n>>> Dispatching {n_time} Timepoints to Joblib Parallel Pool (n_jobs={args.n_jobs}) <<<")

parallel_outputs = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(decode_single_timepoint)(t, pseudo_data, rows, cols, y, cv_splits, n_pseudo, n_pairs, n_chan)
    for t in range(n_time)
)

print("Assembling independent time rows into the final RDM tensor space...")
rdms = np.zeros((n_time, n_pairs), dtype=np.float32)
for t, timepoint_rdm in parallel_outputs:
    rdms[t] = timepoint_rdm
print(f"Final compiled RDM matrix shape: {rdms.shape}")

# Save output array
save_path = os.path.join(out_dir, f"decoding_accuracy_rdm_eeg_sub-{args.subject}.npy")
np.save(save_path, rdms)

print(f"Success! Data written to: {save_path}")
print(f"Done! Total Time: {time.time() - start_time:.2f} seconds.")