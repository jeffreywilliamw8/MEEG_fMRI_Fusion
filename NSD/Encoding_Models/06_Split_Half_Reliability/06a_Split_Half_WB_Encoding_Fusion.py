"""

This script performs a split-half reliability analysis of the EEG-fMRI encoding fusion model across the whole brain. 
It fits a Ridge regression model to predict fMRI responses from EEG data, using independent train/test splits for multiple shuffles. 
The results are saved as correlation values for each shuffle, half, timepoint, and vertex.

Parameters
----------
subject : int
    The participant pair ID.
hemisphere : str
    The hemisphere to analyze ('lh' or 'rh').
fmri_split : int
    The fMRI split to analyze (1-83).
n_shuffles : int
    The number of shuffles to perform for the split-half reliability analysis.
    Must match the --n_shuffles used when generating --split_file (Generate_Test_Split_Indices.py) and
    the value used by the RSA split-half script, so both analyses run the
    same number of shuffles over the same condition splits.
"""



import os
# --- Thread caps BEFORE importing joblib/sklearn, to prevent core oversubscription ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import random
import argparse
import time
from sklearn.linear_model import RidgeCV
from fracridge import FracRidgeRegressorCV
from utils import load_fmri_hemi_data
from joblib import Parallel, delayed

# Start time
start_time = time.time()

# Random seed for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

# =====================================================
# Input arguments
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh')  # lh -> left hemisphere, rh -> right hemisphere
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--n_shuffles', type=int, default=100,
                     help='Number of independent random train/test halving shuffles for the '
                          'split-half reliability test drive. Must match the --n_shuffles used '
                          'when generating --split_file (00_Generate_Test_Split_Indices.py) and '
                          'the value used by the RSA split-half script, so both analyses run the '
                          'same number of shuffles over the same condition splits.')
parser.add_argument('--split_seed', type=int, default=8,
                     help='Seed used when generating the shared test-condition split file. Only '
                          'used to build the default --split_file path below.')
parser.add_argument('--split_file', type=str, default=None,
                     help='Path to the shared (n_shuffles, n_test) test-condition split-label file '
                          'produced by 00_Generate_Test_Split_Indices.py. If not given, defaults to '
                          'the standard shared_splits path for the observed n_test, --n_shuffles, '
                          'and --split_seed. This file -- NOT an internally generated permutation -- '
                          'is what determines the test-set halves, so that this script and the RSA '
                          'split-half script are guaranteed to use identical condition splits.')
parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()

print(f'>>> EEG-fMRI Encoding Fusion -- Split-Half Reliability (Whole-brain), Parallelized <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =====================================================
# Loading the EEG responses
# ======================================================
data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
eeg_train = np.load(
    os.path.join(data_path, f'eeg_train_sub-{args.subject:02d}_trial_avg-all.npy'), allow_pickle=True
).item()['eeg_train']  # Shape: (~9000, 160, 359)
eeg_test = np.load(
    os.path.join(data_path, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True
).item()['eeg_test']  # Shape: (515, 30, 160, 359)
eeg_test = np.mean(eeg_test, axis=1)  # (515, 160, 359)
print('Shape of the EEG data (train, test):', eeg_train.shape, eeg_test.shape)

# =============================================================================
# Load the fMRI responses 
# =============================================================================
FMRI_SPLIT_SIZE = 1974
fmri_train, fmri_test = load_fmri_hemi_data(args.subject, args.hemisphere)  # (9000, 163842), (515, 163842)
fmri_train = fmri_train[:, FMRI_SPLIT_SIZE * (args.fmri_split - 1):FMRI_SPLIT_SIZE * args.fmri_split]
fmri_test = fmri_test[:, FMRI_SPLIT_SIZE * (args.fmri_split - 1):FMRI_SPLIT_SIZE * args.fmri_split]
print("Shape of the fMRI data (train, test):", fmri_train.shape, fmri_test.shape)


# =============================================================================
# Preparing save path (separate results tree from the point-estimate encoding fusion results,
# so this reliability analysis never collides with / overwrites the main results)
# =============================================================================
file_name = f'fmri_split-{args.fmri_split:02d}.npy'
corrs_save_dir = (
    f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/'
    f'encoding_fusion_split_half_reliability/whole_brain/subject-{args.subject}/hemisphere-{args.hemisphere}'
)
os.makedirs(corrs_save_dir, exist_ok=True)

# =============================================================================
# Build the n_shuffles x 2 disjoint train/test index halves up front 
#
# TRAIN halves: independently randomized per shuffle via this script's own rng. The train set
# has no RSA equivalent (RSA never fits a model), so there's nothing to keep it aligned with --
# it's fine for it to be regenerated freely.
#
# TEST halves: loaded from the shared split file rather than generated here. RSA's split-half
# reliability analysis partitions the SAME 515 test conditions, and needs to land on the exact
# same partition per shuffle for the two analyses to be comparable. Relying on "the same seed" in
# both scripts is not robust to that -- this script also draws a train permutation that RSA has
# no equivalent draw for, which would desynchronize the two RNG streams. So instead, the test
# split is generated once by 00_Generate_Test_Split_Indices.py and loaded here as data, not
# regenerated from a seed.

# =============================================================================
n_train = eeg_train.shape[0]
n_test = eeg_test.shape[0]
n_train_half1 = n_train // 2

split_file = args.split_file
if split_file is None:
    split_file = os.path.join(
        '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/shared_splits',
        f'test_split_shuffles_ntest-{n_test}_nshuffles-{args.n_shuffles}_seed-{args.split_seed}.npy'
    )

if not os.path.exists(split_file):
    raise FileNotFoundError(
        f"Shared test-condition split file not found at: {split_file}\n"
        f"Run 00_Generate_Test_Split_Indices.py --n_test {n_test} --n_shuffles {args.n_shuffles} "
        f"--seed {args.split_seed} first, so this script and the RSA split-half script use "
        f"identical test-condition splits."
    )

split_labels = np.load(split_file)  # (n_shuffles, n_test), 0 = half 1, 1 = half 2
if split_labels.shape != (args.n_shuffles, n_test):
    raise ValueError(
        f"Loaded split file shape {split_labels.shape} doesn't match expected "
        f"({args.n_shuffles}, {n_test}). Regenerate it with matching --n_test/--n_shuffles."
    )

print(f"Train halves: {n_train_half1} / {n_train - n_train_half1} samples (n_train={n_train}), "
      f"independently randomized per shuffle.")
print(f"Test halves loaded from shared split file: {split_file}")

rng = np.random.default_rng(seed)
shuffle_splits = []  # shuffle_splits[s] = {'train': [idx_half1, idx_half2], 'test': [idx_half1, idx_half2]}
for s in range(args.n_shuffles):
    train_perm = rng.permutation(n_train)
    test_half1_idx = np.where(split_labels[s] == 0)[0]
    test_half2_idx = np.where(split_labels[s] == 1)[0]
    print(f"  Shuffle {s}: test halves = {len(test_half1_idx)} / {len(test_half2_idx)} conditions")
    shuffle_splits.append({
        'train': [train_perm[:n_train_half1], train_perm[n_train_half1:]],
        'test': [test_half1_idx, test_half2_idx],
    })

# =============================================================================
# Worker: fit + evaluate the encoding model for a single (shuffle, half, timepoint) triplet.
# Receives the FULL eeg_train/eeg_test/fmri_train/fmri_test arrays (shared, memory-mapped by
# loky across workers rather than copied) plus this task's own small index arrays, and does
# the halving itself -- this keeps each dispatched task's payload small regardless of n_shuffles.
# =============================================================================
alphas = np.logspace(-6, 5, 20)  # List of alphas for Ridge regression


def compute_shuffle_half(shuffle_idx, half_idx, t, eeg_train, eeg_test, fmri_train, fmri_test,
                          train_idx, test_idx):
    eeg_train_half = eeg_train[train_idx, :, t]
    fmri_train_half = fmri_train[train_idx, :]
    eeg_test_half = eeg_test[test_idx, :, t]
    fmri_test_half = fmri_test[test_idx, :]

    # Standardize the test half using its own/std (not the full 515-sample set's) --
    # this is the correctness detail called out above.
    fmri_test_half_z = (fmri_test_half - fmri_test_half.mean(0)) / (fmri_test_half.std(0) + 1e-8)

    eeg2fmri = RidgeCV(alphas=alphas, alpha_per_target=True)
    eeg2fmri.fit(eeg_train_half, fmri_train_half)

    pred_fmri = eeg_test_half @ eeg2fmri.coef_.T + eeg2fmri.intercept_
    pred_fmri_z = (pred_fmri - pred_fmri.mean(0)) / (pred_fmri.std(0) + 1e-8)
    corr = np.diag(pred_fmri_z.T @ fmri_test_half_z) / len(pred_fmri_z)

    return shuffle_idx, half_idx, t, corr.astype(np.float32)


# =============================================================================
# Dispatch all (shuffle, half, timepoint) tasks to the joblib pool in one flat call, rather
# than one Parallel(...) call per shuffle/half -- avoids repeated pool dispatch overhead and
# lets all n_shuffles x 2 x n_time tasks share the worker pool.
# =============================================================================
n_time = eeg_train.shape[2]
n_vertices_split = fmri_train.shape[1]
n_total_tasks = args.n_shuffles * 2 * n_time
print(f"\nDispatching {n_total_tasks} tasks ({args.n_shuffles} shuffles x 2 halves x {n_time} timepoints) "
      f"to the Joblib pool...")

results = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(compute_shuffle_half)(
        s, h, t, eeg_train, eeg_test, fmri_train, fmri_test,
        shuffle_splits[s]['train'][h], shuffle_splits[s]['test'][h]
    )
    for s in range(args.n_shuffles)
    for h in range(2)
    for t in range(n_time)
)

# =============================================================================
# Assemble into the (n_shuffles, 2, n_time, n_vertices) array and save once at the end.
# =============================================================================
print("Assembling results into (n_shuffles, 2, n_time, n_vertices) array...")
corrs = np.zeros((args.n_shuffles, 2, n_time, n_vertices_split), dtype=np.float32)
for shuffle_idx, half_idx, t, corr in results:
    corrs[shuffle_idx, half_idx, t, :] = corr

np.save(os.path.join(corrs_save_dir, file_name), corrs)

print("Split-half reliability data collection complete!")
print(f"Correlations saved to: {os.path.join(corrs_save_dir, file_name)}, shape={corrs.shape}")

# End time
execution_time = time.time() - start_time
print("Encoding Fusion (split-half) complete!")
print(f"Execution time: {execution_time:.2f} seconds.")