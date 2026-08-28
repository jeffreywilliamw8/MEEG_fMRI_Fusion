"""
Generates the canonical random test-condition split-half assignments to be
shared by the encoding-fusion split-half reliability analysis
 and the RSA split-half reliability analysis.


Run this ONCE, before launching either analysis's job batch. Both
downstream scripts should load the single file this produces rather than
generating their own random test-set partitions -- that's what guarantees
they operate on IDENTICAL condition splits per shuffle.

Output: an (n_shuffles, n_test) uint8 array, `split_labels`, where
split_labels[s, i] = 0 means test stimulus i belongs to half 1 for shuffle
s, and 1 means it belongs to half 2. No stimulus is ever dropped -- for odd
n_test, half 2 simply gets 1 extra stimulus. With the default n_test=100
this splits evenly (50/50) every shuffle.
"""

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_test', type=int, default=100,
                     help='Number of held-out test stimuli. This is the shared 100-stimulus '
                          'THINGS test set, identical across all 3 fMRI subjects, used by the '
                          'encoding-fusion test set (meg_test / fmri_test) and by any future '
                          'RSA analysis over the same test stimuli.')
parser.add_argument('--n_shuffles', type=int, default=10,
                     help='Must match --n_shuffles used in both the encoding and (future) RSA '
                          'split-half scripts that consume this file.')
parser.add_argument('--seed', type=int, default=8)
args = parser.parse_args()

save_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/shared_splits'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(
    save_dir, f'test_split_shuffles_ntest-{args.n_test}_nshuffles-{args.n_shuffles}_seed-{args.seed}.npy'
)

print(">>> Generating canonical THINGS test-condition split-half assignments <<<")
print(f"n_test={args.n_test} | n_shuffles={args.n_shuffles} | seed={args.seed}")

rng = np.random.default_rng(args.seed)
n_half1 = args.n_test // 2 

split_labels = np.zeros((args.n_shuffles, args.n_test), dtype=np.uint8)
for s in range(args.n_shuffles):
    perm = rng.permutation(args.n_test)
    half2_idx = perm[n_half1:]
    split_labels[s, half2_idx] = 1

np.save(save_path, split_labels)

print(f"\nSaved to: {save_path}")
print(f"Shape: {split_labels.shape} (n_shuffles, n_test)")
for s in range(args.n_shuffles):
    n_h1 = int(np.sum(split_labels[s] == 0))
    n_h2 = int(np.sum(split_labels[s] == 1))
    print(f"  Shuffle {s}: half1={n_h1} stimuli, half2={n_h2} stimuli")