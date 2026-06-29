"""
EEG Image-Wise Decoding Analysis 
Parallelized via Joblib across timepoints, yielding an output array saved as a standard .npy file.
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from joblib import Parallel, delayed
import random
import time

start_time = time.time()
seed = 8
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
args = parser.parse_args()

print(f'>>> Parallelized Image-Wise Decoding Analysis (Sub-{args.subject}) <<<')

# Paths
data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
out_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/decoding_rdms'
os.makedirs(out_dir, exist_ok=True)

# Load data
eeg_dict = np.load(os.path.join(data_dir, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True).item()
eeg_data = eeg_dict['eeg_test']

# Fixed the indexing selection bug here
sub_sample_idx = np.random.choice(eeg_data.shape[0], size=100, replace=False)
eeg_data = eeg_data[sub_sample_idx]
print(f"EEG data shape: {eeg_data.shape} (Stimuli, Trials, Channels, Time)")
n_stim, n_trials, n_chan, n_time = eeg_data.shape

# 1. PSEUDO-TRIALS (6 pseudo-trials)
n_pseudo = 6
trials_per_pseudo = n_trials // n_pseudo
pseudo_data = eeg_data.reshape(n_stim, n_pseudo, trials_per_pseudo, n_chan, n_time).mean(axis=2)
print(f"Data reshaped to pseudo-trials: {pseudo_data.shape} (Stimuli, Pseudo-trials, Channels, Time)")

# 2. SETUP PAIRS
rows, cols = np.triu_indices(n_stim, k=1)
pair_indices = list(zip(rows, cols))
n_pairs = len(pair_indices)

# =============================================================================
# Joblib Parallel Worker Definition
# =============================================================================
def decode_single_timepoint(t, pseudo_data, pair_indices, n_pseudo, n_pairs):
    """
    Worker handling all image pair decodings for a singular time point 't'.
    """
    current_data = pseudo_data[:, :, :, t]
    timepoint_rdm = np.zeros(n_pairs, dtype=np.float32)
    
    # Pre-configure estimators inside worker boundary
    clf = LinearSVC(C=1.0, max_iter=1000, tol=1e-3, random_state=8)
    cv = StratifiedKFold(n_splits=n_pseudo, shuffle=True, random_state=8)
    y = np.concatenate([np.zeros(n_pseudo), np.ones(n_pseudo)])
    
    # Loop over all image combinations for this time step
    for p_idx, (i, j) in enumerate(pair_indices):
        X_i = current_data[i] 
        X_j = current_data[j] 

        X = np.concatenate([X_i, X_j], axis=0)
        
        scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
        timepoint_rdm[p_idx] = np.mean(scores)
        
    return timepoint_rdm

# =============================================================================
# Run Parallel Engine
# =============================================================================
print(f"\n>>> Dispatching {n_time} Timepoints to Joblib Parallel Pool <<<")

# n_jobs=-1 naturally provisions all hardware threads allotted to your task
parallel_outputs = Parallel(n_jobs=10, verbose=10)(
    delayed(decode_single_timepoint)(t, pseudo_data, pair_indices, n_pseudo, n_pairs)
    for t in range(n_time)
)

print("Assembling independent time rows into the final RDM tensor space...")
rdms = np.stack(parallel_outputs, axis=0)  # Shape: (n_time, n_pairs)
print(f"Final compiled RDM matrix shape: {rdms.shape}")

# Save output array
save_path = os.path.join(out_dir, f"decoding_rdm_eeg_sub-{args.subject}.npy")
np.save(save_path, rdms)

print(f"✅ Success! Data written to: {save_path}")
print(f"Done! Total Time: {time.time() - start_time:.2f} seconds.")