"""
THINGS-MEG pairwise decoding-accuracy RDM, one RDMs time series per MEG subject.
R raw repetitions are averaged into pseudo-trials, a StratifiedKFold split and label vector are
precomputed once (identical for every pair/timepoint), and one LinearSVC
instance plus feature buffer is reused across all pairs per timepoint.

With 12 raw repetitions and --reps_per_pseudo 4, this yields 3 pseudo-trials
per stimulus (used as-is, no shuffling -- a single pseudo-trial assignment).

Parameters
----------
meg_subject : MEG participant (1-4).
reps_per_pseudo : raw repetitions averaged into each pseudo-trial.
n_jobs : parallel workers across timepoints.
"""

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import random
import time

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from berg import BERG

from utils import load_meg_test_trials

start_time = time.time()
seed = 8
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--meg_subject', type=int, default=1)
parser.add_argument('--reps_per_pseudo', type=int, default=4)
parser.add_argument('--n_jobs', type=int, default=-1)
parser.add_argument('--berg_dir', type=str, default='/scratch/jeffreykatab/berg')
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/meg_rdms')
parser.add_argument('--tmax', type=float, default=0.8)
args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)

print(f'>>> THINGS-MEG Decoding Accuracy RDM (P{args.meg_subject}) <<<')

berg = BERG(berg_dir=args.berg_dir)
fmri1_dict = np.load(os.path.join(args.data_dir, 'fmri_sub-01.npy'), allow_pickle=True).item()
canonical_test_stimuli = list(fmri1_dict['test_stimuli'])
del fmri1_dict

meg_data, times = load_meg_test_trials(berg, args.berg_dir, args.meg_subject,
                                        canonical_test_stimuli, tmax=args.tmax)
n_stim, n_trials, n_chan, n_time = meg_data.shape
print(f"MEG data shape: {meg_data.shape} (Stimuli, Trials, Channels, Time)")

n_pseudo = n_trials // args.reps_per_pseudo
pseudo_data = meg_data.reshape(
    n_stim, n_pseudo, args.reps_per_pseudo, n_chan, n_time
).mean(axis=2)
print(f"Pseudo-trials: {pseudo_data.shape} (Stimuli, Pseudo-trials, Channels, Time)")

rows, cols = np.triu_indices(n_stim, k=1)
n_pairs = len(rows)
print(f"Number of stimuli: {n_stim} -> {n_pairs} pairs")

y = np.concatenate([np.zeros(n_pseudo), np.ones(n_pseudo)]).astype(np.int8)
cv_splits = list(
    StratifiedKFold(n_splits=n_pseudo, shuffle=True, random_state=seed).split(np.zeros_like(y), y)
)


def decode_single_timepoint(t, pseudo_data, rows, cols, y, cv_splits, n_pseudo, n_pairs, n_chan):
    current_data = pseudo_data[:, :, :, t]
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


parallel_outputs = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(decode_single_timepoint)(t, pseudo_data, rows, cols, y, cv_splits, n_pseudo, n_pairs, n_chan)
    for t in range(n_time)
)

rdms = np.zeros((n_time, n_pairs), dtype=np.float32)
for t, timepoint_rdm in parallel_outputs:
    rdms[t] = timepoint_rdm
print(f"Final RDM shape: {rdms.shape}")

save_path = os.path.join(args.save_dir, f"decoding_accuracy_rdm_meg_sub-P{args.meg_subject}.npy")
np.save(save_path, rdms)
print(f"Saved: {save_path}")
print(f"Done! Total Time: {time.time() - start_time:.2f} seconds.")