"""
Computes EEG RDMs using the crossnobis distance.

Pseudo-trial scheme: with n_pseudo=2, the 30 raw repetitions of each stimulus are split
into two pseudo-trials -- one plays the role of the "training" fold, the other the
"testing" fold for crossnobis' cross-validation. Since which 15 repetitions land in
which half is an arbitrary, noisy split, this assignment is redone over n_shuffles
independent random shuffles of the repetitions, and the final RDM is the average of the
crossnobis RDM obtained from each shuffle
"""

import os
# --- Core Environment Safeguards against worker over-subscription ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import argparse
from sklearn.covariance import LedoitWolf
from joblib import Parallel, delayed
from tqdm import tqdm
import rsatoolbox
from rsatoolbox.data import Dataset
import random
import time

start_time = time.time()
seed = 8
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--n_pseudo', type=int, default=2,
                     help='Number of pseudo-trials to average raw repeats into. With 2, one '
                          'pseudo-trial serves as the crossnobis train fold and the other as '
                          'the test fold.')
parser.add_argument('--n_shuffles', type=int, default=100,
                     help='Number of independent random re-assignments of the 30 repetitions '
                          'into pseudo-trials. The final RDM is the average crossnobis RDM '
                          'across all shuffles.')
parser.add_argument('--n_jobs', type=int, default=10)
args = parser.parse_args()

print(f'>>> Parallelized EEG Crossnobis RDM Analysis (Sub-{args.subject}) <<<')

# Paths
data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
out_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/crossnobis_rdms'
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# Load data (full test set -- no subsampling)
# =============================================================================
eeg_dict = np.load(os.path.join(data_dir, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True).item()
eeg_data = eeg_dict['eeg_test']  # Shape: (n_stim, n_trials, n_chan, n_time)
print(f"EEG data shape: {eeg_data.shape} (Stimuli, Trials, Channels, Time)")
n_stim, n_trials, n_chan, n_time = eeg_data.shape

n_pairs_expected = n_stim * (n_stim - 1) // 2
print(f"Number of stimuli: {n_stim} -> {n_pairs_expected} pairs")

# =============================================================================
# 1. Pseudo-trial setup 
# =============================================================================
n_pseudo = args.n_pseudo
if n_trials % n_pseudo != 0:
    print(f"Warning: n_trials ({n_trials}) not evenly divisible by n_pseudo ({n_pseudo}); "
          f"{n_trials % n_pseudo} trailing trial(s) will be dropped.")
trials_per_pseudo = n_trials // n_pseudo
usable_trials = trials_per_pseudo * n_pseudo

# Observation-level descriptors (same for every timepoint AND every shuffle, so build once)
conds = np.repeat(np.arange(n_stim), n_pseudo)          # condition/stimulus id per observation
cv_folds = np.tile(np.arange(n_pseudo), n_stim)          # pseudo-trial fold id per observation

rng = np.random.default_rng(seed)


# =============================================================================
# Joblib Parallel Worker Definition
# =============================================================================
def compute_crossnobis_single_timepoint(t, pseudo_data, conds, cv_folds, n_stim, n_pseudo, n_chan):
    """
    Worker computing the full crossnobis RDM (all condition pairs) for a single timepoint t,
    using rsatoolbox with a shrinkage-regularized noise precision matrix for multivariate
    noise normalization.
    """
    current_data = pseudo_data[:, :, :, t]  # (n_stim, n_pseudo, n_chan)
    measurements = current_data.reshape(n_stim * n_pseudo, n_chan)  # (n_obs, n_chan)

    # --- Multivariate noise normalization: estimate a shrinkage-regularized precision matrix ---
    # Residual = each pseudo-trial's pattern minus its own condition's across-pseudo-trial mean
    condition_means = current_data.mean(axis=1)  # (n_stim, n_chan)
    residuals = (current_data - condition_means[:, None, :]).reshape(n_stim * n_pseudo, n_chan)

    lw = LedoitWolf().fit(residuals)
    covariance = lw.covariance_
    precision = np.linalg.inv(covariance)  # rsatoolbox expects a precision (inverse covariance) matrix

    # --- Build rsatoolbox dataset and compute crossnobis RDM ---
    dataset = Dataset(
        measurements=measurements,
        obs_descriptors={'conds': conds, 'cv_desc': cv_folds}
    )

    rdm_obj = rsatoolbox.rdm.calc_rdm(
        dataset,
        method='crossnobis',
        descriptor='conds',
        cv_descriptor='cv_desc',
        noise=precision
    )

    # rdm_obj.dissimilarities has shape (1, n_pairs); flatten to (n_pairs,)
    return rdm_obj.dissimilarities[0].astype(np.float32)


# =============================================================================
# 2. Shuffle loop: for each of n_shuffles independent random assignments of the 30
#    repetitions into n_pseudo pseudo-trials, compute the full (n_time, n_pairs)
#    crossnobis RDM
# =============================================================================
rdm_sum = np.zeros((n_time, n_pairs_expected), dtype=np.float64)

print(f"\n>>> Running {args.n_shuffles} random pseudo-trial shuffles "
      f"({n_time} timepoints each, {args.n_jobs} parallel workers) <<<")

for shuffle_idx in tqdm(range(args.n_shuffles), desc="Pseudo-trial shuffles"):
    # new random assignment of repetitions to pseudo-trials for this shuffle
    trial_order = rng.permutation(n_trials)[:usable_trials]
    eeg_data_shuffled = eeg_data[:, trial_order, :, :]

    pseudo_data = eeg_data_shuffled.reshape(
        n_stim, n_pseudo, trials_per_pseudo, n_chan, n_time
    ).mean(axis=2)  # Shape: (n_stim, n_pseudo, n_chan, n_time)

    parallel_outputs = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(compute_crossnobis_single_timepoint)(
            t, pseudo_data, conds, cv_folds, n_stim, n_pseudo, n_chan
        )
        for t in range(n_time)
    )

    rdm_shuffle = np.stack(parallel_outputs, axis=0).astype(np.float32)  # (n_time, n_pairs)
    assert rdm_shuffle.shape[1] == n_pairs_expected, \
        f"Unexpected number of pairs: got {rdm_shuffle.shape[1]}, expected {n_pairs_expected}"

    rdm_sum += rdm_shuffle

print("Averaging RDMs across all shuffles...")
rdms = (rdm_sum / args.n_shuffles).astype(np.float32)  # Shape: (n_time, n_pairs)
print(f"Final compiled RDM matrix shape: {rdms.shape}")

# Save output array
save_path = os.path.join(out_dir, f"crossnobis_rdm_eeg_sub-{args.subject}.npy")
np.save(save_path, rdms)

print(f"Success! Data written to: {save_path}")
print(f"Done! Total Time: {time.time() - start_time:.2f} seconds.")