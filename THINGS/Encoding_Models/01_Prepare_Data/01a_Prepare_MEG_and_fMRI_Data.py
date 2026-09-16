"""
One-time pre-computation of THINGS-fMRI and THINGS-MEG data into .npy files.

-----------------------------
Parameters:
- berg_dir: Path to the BERG installation/data directory, used to fetch
  model metadata (stimulus lists, timing) and to locate the raw HDF5 files.
- fmri_subjects: List of THINGS-fMRI subjects to process (1, 2, 3).
- meg_subjects: List of THINGS-MEG subjects whose sensor data is
  concatenated along the channel axis into the combined meg_train/meg_test
  arrays.
- save_dir: Directory where the per-subject fmri_sub-XX.npy files and the
  meg_train.npy / meg_test.npy files are saved.
- tmax: Upper time bound (seconds post stimulus onset) the MEG data is
  truncated to.
"""
import os
import argparse
import numpy as np
import h5py
from tqdm import tqdm
from berg import BERG
import time


start_time = time.time()


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--berg_dir', type=str, default='/scratch/giffordale95/projects/brain-encoding-response-generator')
parser.add_argument('--data_dir', type=str, default='/scratch/jeffreykatab/berg')
parser.add_argument('--fmri_subjects', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--meg_subjects', type=int, nargs='+', default=[1, 2, 3, 4])
parser.add_argument('--save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--tmax', type=float, default=1.3)  # seconds post stimulus onset -- a ceiling; downstream code selects its own end time within it
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

print('>>> Pre-computing THINGS-fMRI and THINGS-MEG data (aligned) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


berg = BERG(berg_dir=args.berg_dir)


def require_file(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(f" Missing {what}: {path}\n")
    return path


# =============================================================================
# 1. Preparing fMRI data (train + test), per subject
# =============================================================================
fmri1_test_stimuli_unique = None
train_stimuli_fmri1 = None

FMRI_TRAIN_DIR = args.berg_dir   
                                  
FMRI_TEST_DIR = args.data_dir

for fsub in tqdm(args.fmri_subjects, desc='fMRI subjects'):

    metadata_fmri = berg.get_model_metadata(
        'fmri-things_fmri_1-vit_b_32',
        subject=fsub
    )

    fmri_train_file = require_file(os.path.join(FMRI_TRAIN_DIR, 'model_training_datasets',
        'train_dataset-things_fmri_1', f'fmri_sub-{fsub:02d}_split-train.h5'),
        f'fMRI sub-{fsub:02d} train file')
    fmri_train = h5py.File(fmri_train_file, 'r')['neural_data'][:]  # whole brain, all voxels

    fmri_test_file = require_file(os.path.join(FMRI_TEST_DIR, 'model_training_datasets',
        'train_dataset-things_fmri_1', f'fmri_sub-{fsub:02d}_split-test.h5'),
        f'fMRI sub-{fsub:02d} test file')
    fmri_test_all = h5py.File(fmri_test_file, 'r')['neural_data']

    train_stimuli_fmri = metadata_fmri['encoding_model']['train_stimuli']
    test_stimuli_fmri = metadata_fmri['encoding_model']['test_stimuli']
    unique_test_stimuli = np.unique(test_stimuli_fmri)

    if fsub == args.fmri_subjects[0]:
        fmri1_test_stimuli_unique = unique_test_stimuli
        train_stimuli_fmri1 = train_stimuli_fmri

    # Average the fMRI responses across repetitions of the same test stimulus.
    fmri_test = []
    for stim in unique_test_stimuli:
        idx = np.where(test_stimuli_fmri == stim)[0]
        fmri_test.append(fmri_test_all[idx].mean(0))
    fmri_test = np.array(fmri_test)

    print(f"\n[fMRI sub-{fsub:02d}] train {fmri_train.shape}, test {fmri_test.shape}")

    # ROI voxel indices and whole-brain noise ceiling, needed for
    # load_fmri_roi_data() in utils.py
    roi_dict = metadata_fmri['roi']
    noise_ceiling_testset = metadata_fmri['encoding_model']['noise_ceiling_testset']

    save_path = os.path.join(args.save_dir, f'fmri_sub-{fsub:02d}.npy')
    np.save(save_path, {
        'fmri_train': fmri_train,
        'train_stimuli': train_stimuli_fmri,
        'fmri_test': fmri_test,
        'test_stimuli': unique_test_stimuli,
        'roi': roi_dict,
        'noise_ceiling_testset': noise_ceiling_testset,
    })
    print(f"Saved: {save_path}")

    del fmri_train, fmri_test_all, fmri_test

# =============================================================================
# 2. MEG: sensors concatenated across subjects along the channel axis,
#    computed once, saved as 2 files (train, test). Aligned to fMRI subject 1's stimulus order.
# =============================================================================
meg_train, meg_test = None, None
meg_times = None

for ms, msub in enumerate(tqdm(args.meg_subjects, desc='MEG subjects')):

    metadata_meg = berg.get_model_metadata(
        'meg-things_meg_1-vit_b_32',
        subject=msub
    )

    # Time point selection (first tmax seconds post stimulus onset)
    times = metadata_meg['meg']['times']
    time_idx = np.zeros(len(times), dtype=int)
    time_idx[times <= args.tmax] = 1
    time_idx = np.where(time_idx == 1)[0]
    times = times[times <= args.tmax]
    if ms == 0:
        meg_times = times

    # Loading the MEG responses
    meg_train_file = require_file(os.path.join(args.data_dir, 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_all_training_splits.h5'),
        f'MEG P{msub} train file')
    meg_train_sub = h5py.File(meg_train_file, 'r')['neural_data']

    meg_test_file = require_file(os.path.join(args.data_dir, 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_split-test.h5'),
        f'MEG P{msub} test file')
    meg_test_all = h5py.File(meg_test_file, 'r')['neural_data']
    meg_test_all = meg_test_all[:, :, time_idx].astype(np.float32)

    # Averaging the MEG test responses across repetitions, for the fMRI
    # subject-1 unique test stimuli
    test_stimuli_meg = metadata_meg['encoding_model']['test_stimuli']
    meg_test_sub = []
    for stim in fmri1_test_stimuli_unique:
        idx = [i for i, x in enumerate(test_stimuli_meg) if x == stim]
        meg_test_sub.append(meg_test_all[idx].mean(0))
    meg_test_sub = np.array(meg_test_sub)

    # MEG train
    train_stimuli_meg = metadata_meg['encoding_model']['all_training_splits']['train_stimuli']
    idx_meg = []
    for stim in train_stimuli_fmri1:
        idx_meg.append(train_stimuli_meg.index(stim))
    idx_meg = np.array(idx_meg)
    meg_train_sub = meg_train_sub[:, :, time_idx][idx_meg].astype(np.float32)

    print(f"[MEG P{msub}] train {meg_train_sub.shape}, test {meg_test_sub.shape}")

    # Appending the MEG sensor responses across subjects (channel axis)
    if ms == 0:
        meg_train = meg_train_sub
        meg_test = meg_test_sub
    else:
        meg_train = np.append(meg_train, meg_train_sub, 1)
        meg_test = np.append(meg_test, meg_test_sub, 1)

    del meg_train_sub, meg_test_all, meg_test_sub

print(f"\n[MEG, all subjects concatenated] train {meg_train.shape}, test {meg_test.shape}")

meg_train_path = os.path.join(args.save_dir, 'meg_train.npy')
np.save(meg_train_path, {
    'meg_train': meg_train,
    'times': meg_times,
    'meg_subjects': args.meg_subjects,
    'train_stimuli': train_stimuli_fmri1,  # exact order meg_train's rows are in
})
print(f"Saved: {meg_train_path}")

meg_test_path = os.path.join(args.save_dir, 'meg_test.npy')
np.save(meg_test_path, {
    'meg_test': meg_test,
    'times': meg_times,
    'meg_subjects': args.meg_subjects,
    'test_stimuli': fmri1_test_stimuli_unique,  # exact order meg_test's rows are in
})
print(f"Saved: {meg_test_path}")

print(f"\nExecution complete! Total time: {time.time() - start_time:.2f} seconds.")