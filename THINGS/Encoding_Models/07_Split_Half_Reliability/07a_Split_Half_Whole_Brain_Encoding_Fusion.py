"""
MEG-fMRI Encoding Fusion, split-half reliability -- whole brain.

Split both the training stimuli and the test stimuli into two disjoint halves, fit a
RidgeCV MEG-to-fMRI encoder separately on each half's training data, predict
that half's test data, and correlate against the actual fMRI -- per
timepoint, per voxel.

Parameters
----------
fmri_subject : int
fmri_split, n_splits : whole-brain voxel-split scheme (as in
    THINGS_Whole_Brain_Encoding_Fusion.py).
n_shuffles : int
    Number of independent random train/test halving shuffles. Default 10.
data_dir, save_dir, n_jobs, tmax : as in the other whole-brain THINGS
    scripts.
"""

import argparse
import os
import random
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits

from utils import load_fmri_wb_data, load_meg_data

start_time = time.time()

seed = 8
np.random.seed(seed)
random.seed(seed)

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)  # 1, 2, or 3
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--n_splits', type=int, default=500)
parser.add_argument('--n_shuffles', type=int, default=10,
                     help='Number of independent random train/test halving shuffles. Must match '
                          'the --n_shuffles used to generate --split_file.')
parser.add_argument('--split_seed', type=int, default=8,
                     help='Seed used when generating the shared test-condition split file. Only '
                          'used to build the default --split_file path below.')
parser.add_argument('--split_file', type=str, default=None,
                     help='Path to the shared (n_shuffles, n_test) test-condition split-label file '
                          'produced by THINGS_00_Generate_Test_Split_Indices.py. If not given, '
                          'defaults to the standard shared_splits path for the observed n_test, '
                          '--n_shuffles, and --split_seed.')
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/'
                             'results/correlations/encoding_fusion_split_half_reliability/whole_brain')
parser.add_argument('--n_jobs', type=int, default=-1)
parser.add_argument('--tmax', type=float, default=0.6)
args = parser.parse_args()

print('>>> MEG-fMRI Encoding Fusion -- Split-Half Reliability (Whole-Brain) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

eps = 1e-8

# =============================================================================
# Loading this run's fMRI and MEG data
# =============================================================================
fmri_train_full, fmri_test_full = load_fmri_wb_data( args.fmri_subject)
meg_train, meg_test = load_meg_data(tmax=args.tmax)
n_time = meg_train.shape[2]

print(f"\n[fMRI sub-{args.fmri_subject:02d}] full-brain train {fmri_train_full.shape}, "
      f"test {fmri_test_full.shape}")
print(f"[MEG] train {meg_train.shape}, test {meg_test.shape}, {n_time} timepoints")

# Same shared voxel-split scheme as THINGS_Whole_Brain_Encoding_Fusion.py.
fmri_train_splits = np.array_split(fmri_train_full, args.n_splits, axis=1)
fmri_test_splits = np.array_split(fmri_test_full, args.n_splits, axis=1)
fmri_train = fmri_train_splits[args.fmri_split - 1]
fmri_test = fmri_test_splits[args.fmri_split - 1]
n_voxels = fmri_train.shape[1]
print(f"Split {args.fmri_split}/{args.n_splits}: {n_voxels} voxels "
      f"(train {fmri_train.shape}, test {fmri_test.shape})")


# =============================================================================
# Save path
# =============================================================================
save_dir = os.path.join(args.save_dir, f'fmri_sub-{args.fmri_subject:02d}')
os.makedirs(save_dir, exist_ok=True)
file_name = f'fmri_split-{args.fmri_split:03d}.npy'

# =============================================================================
# Build the n_shuffles x 2 disjoint train/test index halves up front.
#
# TRAIN halves: independently randomized per shuffle via this script's own


# TEST halves: loaded from the shared split file produced by
# Generate_Test_Split_Indices.py
# =============================================================================
n_train = meg_train.shape[0]
n_test = meg_test.shape[0]
n_train_half1 = n_train // 2

split_file = args.split_file
if split_file is None:
    split_file = os.path.join(
        '/scratch/jeffreykatab/Projects/fusion/THINGS/shared_splits',
        f'test_split_shuffles_ntest-{n_test}_nshuffles-{args.n_shuffles}_seed-{args.split_seed}.npy'
    )

if not os.path.exists(split_file):
    raise FileNotFoundError(
        f"Shared test-condition split file not found at: {split_file}\n"
        f"Run THINGS_00_Generate_Test_Split_Indices.py --n_test {n_test} --n_shuffles "
        f"{args.n_shuffles} --seed {args.split_seed} first, so this script and any future THINGS "
        f"RSA split-half script use identical test-condition splits."
    )

split_labels = np.load(split_file)  # (n_shuffles, n_test), 0 = half 1, 1 = half 2
if split_labels.shape != (args.n_shuffles, n_test):
    raise ValueError(
        f"Loaded split file shape {split_labels.shape} doesn't match expected "
        f"({args.n_shuffles}, {n_test}). Regenerate it with matching --n_test/--n_shuffles."
    )

print(f"\nTrain halves: {n_train_half1} / {n_train - n_train_half1} stimuli (n_train={n_train}), "
      f"independently randomized per shuffle.")
print(f"Test halves loaded from shared split file: {split_file}")

rng = np.random.default_rng(seed)
shuffle_splits = []  # shuffle_splits[s] = {'train': [idx_half1, idx_half2], 'test': [idx_half1, idx_half2]}
for s in range(args.n_shuffles):
    train_perm = rng.permutation(n_train)
    test_half1_idx = np.where(split_labels[s] == 0)[0]
    test_half2_idx = np.where(split_labels[s] == 1)[0]
    print(f"  Shuffle {s}: test halves = {len(test_half1_idx)} / {len(test_half2_idx)} stimuli")
    shuffle_splits.append({
        'train': [train_perm[:n_train_half1], train_perm[n_train_half1:]],
        'test': [test_half1_idx, test_half2_idx],
    })

# =============================================================================
# Worker: fit + evaluate the encoding model for a single (shuffle, half,
# timepoint) triplet. Receives the full meg_train/meg_test/fmri_train/fmri_test arrays
# =============================================================================
alphas = np.logspace(-6, 10, 17)  # matches THINGS_utils.DEFAULT_ALPHAS


def compute_shuffle_half(shuffle_idx, half_idx, t, meg_train, meg_test, fmri_train, fmri_test,
                          train_idx, test_idx):
    with threadpool_limits(limits=1):
        meg_train_half = meg_train[train_idx, :, t]
        fmri_train_half = fmri_train[train_idx, :]
        meg_test_half = meg_test[test_idx, :, t]
        fmri_test_half = fmri_test[test_idx, :]

        # Standardize the test half using its own mean/std, not that of the full
        # 100-stimulus set
        fmri_test_half_z = (fmri_test_half - fmri_test_half.mean(0)) / (fmri_test_half.std(0) + eps)

        meg2fmri = RidgeCV(alphas=alphas, alpha_per_target=True)
        meg2fmri.fit(meg_train_half, fmri_train_half)

        pred_fmri = meg_test_half @ meg2fmri.coef_.T + meg2fmri.intercept_
        pred_fmri_z = (pred_fmri - pred_fmri.mean(0)) / (pred_fmri.std(0) + eps)
        corr = (pred_fmri_z * fmri_test_half_z).mean(axis=0)

    return shuffle_idx, half_idx, t, corr.astype(np.float32)


# =============================================================================
# Dispatch all (shuffle, half, timepoint) tasks in one joblib call.
# =============================================================================
n_total_tasks = args.n_shuffles * 2 * n_time
print(f"\nDispatching {n_total_tasks} tasks ({args.n_shuffles} shuffles x 2 halves x "
      f"{n_time} timepoints) to the joblib pool...")

results = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(compute_shuffle_half)(
        s, h, t, meg_train, meg_test, fmri_train, fmri_test,
        shuffle_splits[s]['train'][h], shuffle_splits[s]['test'][h]
    )
    for s in range(args.n_shuffles)
    for h in range(2)
    for t in range(n_time)
)

# =============================================================================
# Assemble into the (n_shuffles, 2, n_time, n_voxels) array and save once.
# =============================================================================
print("Assembling results into (n_shuffles, 2, n_time, n_voxels) array...")
corrs = np.zeros((args.n_shuffles, 2, n_time, n_voxels), dtype=np.float32)
for shuffle_idx, half_idx, t, corr in results:
    corrs[shuffle_idx, half_idx, t, :] = corr

save_path = os.path.join(save_dir, file_name)
np.save(save_path, corrs)
print(f"\nSaved: {save_path} (shape {corrs.shape})")

print(f"\nExecution complete! Total time: {time.time() - start_time:.2f} seconds.")