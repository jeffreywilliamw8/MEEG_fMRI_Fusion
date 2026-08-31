"""
THINGS-MEG crossnobis RDM, one RDM time series per MEG subject, all subjects
processed in a single run. Pseudo-trials split the raw repetitions into
n_pseudo cross-validation folds, crossnobis distance is computed per timepoint
with rsatoolbox (Ledoit-Wolf shrinkage noise precision), and the pseudo-trial
assignment is redrawn over n_shuffles random shuffles, with the final RDM
being the average across shuffles.

Parameters
----------
meg_subjects : MEG participants to process.
n_pseudo : number of pseudo-trial CV folds the repetitions are split into.
n_shuffles : number of random repetition-to-pseudo-trial reassignments averaged over.
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
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
import rsatoolbox
from rsatoolbox.data import Dataset
from berg import BERG

from utils import load_meg_test_trials

start_time = time.time()
seed = 8
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--meg_subjects', type=int, nargs='+', default=[1, 2, 3, 4])
parser.add_argument('--n_pseudo', type=int, default=2)
parser.add_argument('--n_shuffles', type=int, default=100)
parser.add_argument('--berg_dir', type=str, default='/scratch/jeffreykatab/berg')
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/crossnobis_rdms')
parser.add_argument('--tmax', type=float, default=0.8)
args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)

print('>>> THINGS-MEG Crossnobis RDM <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

berg = BERG(berg_dir=args.berg_dir)
fmri1_dict = np.load(os.path.join(args.data_dir, 'fmri_sub-01.npy'), allow_pickle=True).item()
canonical_test_stimuli = list(fmri1_dict['test_stimuli'])
del fmri1_dict


def compute_crossnobis_single_timepoint(t, pseudo_data, conds, cv_folds, n_stim, n_pseudo, n_chan):
    current_data = pseudo_data[:, :, :, t]
    measurements = current_data.reshape(n_stim * n_pseudo, n_chan)

    condition_means = current_data.mean(axis=1)
    residuals = (current_data - condition_means[:, None, :]).reshape(n_stim * n_pseudo, n_chan)
    lw = LedoitWolf().fit(residuals)
    precision = np.linalg.inv(lw.covariance_)

    dataset = Dataset(measurements=measurements, obs_descriptors={'conds': conds, 'cv_desc': cv_folds})
    rdm_obj = rsatoolbox.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds',
                                       cv_descriptor='cv_desc', noise=precision)
    return rdm_obj.dissimilarities[0].astype(np.float32)


for msub in args.meg_subjects:

    print(f"\n>>> MEG P{msub} <<<")

    meg_data, times = load_meg_test_trials(berg, args.berg_dir, msub,
                                            canonical_test_stimuli, tmax=args.tmax)
    n_stim, n_trials, n_chan, n_time = meg_data.shape
    print(f"MEG data shape: {meg_data.shape} (Stimuli, Trials, Channels, Time)")

    n_pairs_expected = n_stim * (n_stim - 1) // 2
    n_pseudo = args.n_pseudo
    trials_per_pseudo = n_trials // n_pseudo
    usable_trials = trials_per_pseudo * n_pseudo

    conds = np.repeat(np.arange(n_stim), n_pseudo)
    cv_folds = np.tile(np.arange(n_pseudo), n_stim)

    rng = np.random.default_rng(seed)
    rdm_sum = np.zeros((n_time, n_pairs_expected), dtype=np.float64)

    for shuffle_idx in tqdm(range(args.n_shuffles), desc=f"P{msub} pseudo-trial shuffles"):
        trial_order = rng.permutation(n_trials)[:usable_trials]
        meg_data_shuffled = meg_data[:, trial_order, :, :]
        pseudo_data = meg_data_shuffled.reshape(
            n_stim, n_pseudo, trials_per_pseudo, n_chan, n_time
        ).mean(axis=2)

        outputs = [
            compute_crossnobis_single_timepoint(
                t, pseudo_data, conds, cv_folds, n_stim, n_pseudo, n_chan
            )
            for t in range(n_time)
        ]
        rdm_sum += np.stack(outputs, axis=0).astype(np.float32)

    rdms = (rdm_sum / args.n_shuffles).astype(np.float32)
    print(f"Final RDM shape: {rdms.shape}")

    save_path = os.path.join(args.save_dir, f"crossnobis_rdm_meg_sub-P{msub}.npy")
    np.save(save_path, rdms)
    print(f"Saved: {save_path}")

    del meg_data, rdm_sum, rdms

print(f"\nDone! Total Time: {time.time() - start_time:.2f} seconds.")