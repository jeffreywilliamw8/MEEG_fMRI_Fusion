"""
This script computes the Pearson correlation distance RDMs for EEG data from the NSD dataset.
It averages across all trials for each stimulus and computes the pairwise Pearson correlation distance (1 - r) for each timepoint.
It then outputs the vectorized upper-triangle RDMs for each timepoint, which will be used for downstream RSA analyses.

Parameters:
- subject: The subject number (integer) for which to compute the EEG RDMs.


"""

import os
import numpy as np
import argparse
import time

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
args = parser.parse_args()

print(f'>>> EEG RDM Analysis -- Pearson Correlation Distance (Sub-{args.subject}) <<<')

data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
out_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlation_rdms'
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# Load data)
# =============================================================================
eeg_dict = np.load(os.path.join(data_dir, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True).item()
eeg_data = eeg_dict['eeg_test']  # Shape: (n_stim, n_trials, n_chan, n_time)
print(f"EEG data shape: {eeg_data.shape} (Stimuli, Trials, Channels, Time)")
n_stim, n_trials, n_chan, n_time = eeg_data.shape

n_pairs_expected = n_stim * (n_stim - 1) // 2
print(f"Number of stimuli: {n_stim} -> {n_pairs_expected} pairs")

# =============================================================================
# Average across all repeats
# =============================================================================
avg_data = eeg_data.mean(axis=1)  # Shape: (n_stim, n_chan, n_time) : (515, 160, 359)
print(f"Averaged EEG data shape (repeats collapsed): {avg_data.shape}")

# =============================================================================
# Pairwise Pearson correlation distance (1 - r) per timepoint,
# then vectorize the upper triangle of the RDM
# =============================================================================
rows, cols = np.triu_indices(n_stim, k=1)
rdms = np.zeros((n_time, n_pairs_expected), dtype=np.float32)

print(f"\n>>> Computing Pearson-correlation RDMs for {n_time} timepoints <<<")
for t in range(n_time):
    patterns = avg_data[:, :, t]         # (n_stim, n_chan)
    corr_matrix = np.corrcoef(patterns)  # (n_stim, n_stim) -- Pearson r between every pair of stimulus patterns
    dissim_matrix = 1.0 - corr_matrix
    rdms[t] = dissim_matrix[rows, cols].astype(np.float32)

print(f"Final compiled RDM matrix shape: {rdms.shape}")

save_path = os.path.join(out_dir, f"correlation_rdm_eeg_sub-{args.subject}.npy")
np.save(save_path, rdms)

print(f"Success! Data written to: {save_path}")
print(f"Done! Total Time: {time.time() - start_time:.2f} seconds.")