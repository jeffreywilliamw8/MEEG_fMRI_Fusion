"""
Computes THINGS-MEG pairwise Pearson-correlation-distance RDMs
for each MEG participant and save the corresponding RDM time series

-----------------------------
Parameters
----------
meg_subjects : list of int
    Which MEG subjects to compute RDMs for. Default [1, 2, 3, 4].
berg_dir : str
    Path to the BERG installation/data directory.
data_dir : str
    Source data directory
save_dir : str
    Directory the per-subject meg_rdms_sub-PX.npy files are saved to.

tmax : float
    Upper time bound (seconds post stimulus onset) the MEG data is
    truncated to. Default 0.8
"""

import os
import argparse

import numpy as np
import h5py
from tqdm import tqdm
from berg import BERG  # NOTE: adjust this import to match your BERG installation

parser = argparse.ArgumentParser()
parser.add_argument('--meg_subjects', type=int, nargs='+', default=[1, 2, 3, 4])
parser.add_argument('--berg_dir', type=str, default='/scratch/jeffreykatab/berg')
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--tmax', type=float, default=0.8)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

print('>>> Computing THINGS-MEG per-subject RDM time series <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

berg = BERG(berg_dir=args.berg_dir)

# =============================================================================
# uniform 100-stimulus test order -- reused from the existing precomputed
# fmri_sub-01.npy rather than re-derived, so these RDMs align stimulus-for-
# stimulus with the fMRI/DNN RDMs they'll later be compared against.
# =============================================================================
fmri1_dict = np.load(os.path.join(args.data_dir, 'fmri_sub-01.npy'), allow_pickle=True).item()
uniform_test_stimuli = list(fmri1_dict['test_stimuli'])
del fmri1_dict
n_stimuli = len(uniform_test_stimuli)
print(f"\nuniform test set: {n_stimuli} stimuli (from fmri_sub-01.npy)")

def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)  # k=1 excludes diagonal

# =============================================================================
# Loop across MEG subjects: load raw test responses, repetition-average in
# uniform stimulus order, compute a Pearson-distance RDM per timepoint.
# =============================================================================
for msub in tqdm(args.meg_subjects, desc='MEG subjects'):

    metadata_meg = berg.get_model_metadata('meg-things_meg_1-vit_b_32', subject=msub)

    # Time point selection (first tmax seconds post stimulus onset) 
    times = metadata_meg['meg']['times']
    time_idx = np.where(times <= args.tmax)[0]
    times = times[times <= args.tmax]

    # Raw test responses (repetitions not yet averaged).
    meg_test_file = os.path.join(args.berg_dir, 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_split-test.h5')
    meg_test_all = h5py.File(meg_test_file, 'r')['neural_data']
    meg_test_all = meg_test_all[:, :, time_idx].astype(np.float32)

    # Average repetitions of each uniform test stimulus, in uniform order
    # for each subject's own channels
    test_stimuli_meg = metadata_meg['encoding_model']['test_stimuli']
    meg_test_sub = []
    for stim in uniform_test_stimuli:
        idx = [i for i, x in enumerate(test_stimuli_meg) if x == stim]
        meg_test_sub.append(meg_test_all[idx].mean(0))
    meg_test_sub = np.array(meg_test_sub)  # (n_stimuli, n_channels, n_time)
    n_channels = meg_test_sub.shape[1]
    n_time = meg_test_sub.shape[2]
    print(f"\n[MEG P{msub}] repetition-averaged test: {meg_test_sub.shape} "
          f"({n_channels} channels, {n_time} timepoints)")

    # =========================================================================
    # Pearson-correlation-distance RDM per timepoint: 1 - corrcoef of the
    # (n_stimuli, n_channels) data matrix at that timepoint.
    # =========================================================================
    rdms = np.zeros((n_time, (n_stimuli*(n_stimuli-1)//2)), dtype=np.float32) # Shape = (n_time points, n unique pairwise distances)
    for t in range(n_time):
        pattern_matrix = meg_test_sub[:, :, t]  # (n_stimuli, n_channels)
        rdm = 1.0 - np.corrcoef(pattern_matrix)
        rdms[t] = flatten_rdm(rdm) # Storing the upper triangular RDM


    save_path = os.path.join(args.save_dir, f"pearsonr_rdm_meg_sub-P{msub}.npy")
    np.save(save_path, rdms)
    print(f"Saved: {save_path} (rdms shape {rdms.shape})")

    del meg_test_all, meg_test_sub, rdms

print('\nDone.')