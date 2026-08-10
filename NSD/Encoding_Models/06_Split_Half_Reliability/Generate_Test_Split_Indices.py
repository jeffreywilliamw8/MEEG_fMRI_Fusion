"""
Generates the canonical random test-condition split-half assignments shared by both the
encoding-fusion split-half reliability analysis and the RSA split-half reliability analysis.

Run this once, before launching either analysis's job batch. Both downstream scripts load the
single file this produces rather than generating their own random test-set partitions, which is
what guarantees they operate on identical condition splits per shuffle -- this can't be achieved
reliably by just using the same seed in each script independently

Output: an (n_shuffles, n_test) uint8 array, `split_labels`, where split_labels[s, i] = 0 means
test condition i belongs to half 1 for shuffle s, and 1 means it belongs to half 2. No condition
is ever dropped -- for odd n_test, half 2 simply gets 1 extra condition (e.g. 515 -> 257 / 258).
"""

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_test', type=int, default=515,
                     help='Number of held-out test conditions/stimuli. These are the same 515 '
                          'shared NSD test stimuli used identically by the encoding-fusion test '
                          'set (eeg_test / fmri_test) and by the RSA full RDMs (n_stim).')
parser.add_argument('--n_shuffles', type=int, default=100,
                     help='Must match --n_shuffles used in both the encoding and RSA split-half '
                          'scripts that consume this file.')
parser.add_argument('--seed', type=int, default=8)
args = parser.parse_args()

save_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/shared_splits'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(
    save_dir, f'test_split_shuffles_ntest-{args.n_test}_nshuffles-{args.n_shuffles}_seed-{args.seed}.npy'
)

print(">>> Generating canonical test-condition split-half assignments <<<")
print(f"n_test={args.n_test} | n_shuffles={args.n_shuffles} | seed={args.seed}")

rng = np.random.default_rng(args.seed)
n_half1 = args.n_test // 2  # half 2 gets the remainder, so nothing is ever dropped

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
    print(f"  Shuffle {s}: half1={n_h1} conditions, half2={n_h2} conditions")