import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
import random
from sklearn.linear_model import RidgeCV
import time
import gc

#=============================================================================
# Because each fMRI Participant has a different number of voxels,
# And for an efficient use of computational resources,
# We write a different version of this script for each participant.
# This script is for fMRI Subject 03.
#=============================================================================

# Start time
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--berg_dir', default='/scratch/jeffreykatab/Code/Encoding_Models/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print(f'>>> MEG-fMRI Encoding Fusion (Whole-Brain) <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 8
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Loading the fMRI data
# =============================================================================
# Loading the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-things_fmri_1-vit_b_32',
    subject=3
    )

# Loading the fMRI responses
fmri_train_file = os.path.join('/scratch/jeffreykatab/berg', 'model_training_datasets',
    'train_dataset-things_fmri_1',
    'fmri_sub-03_split-train.h5')
fmri_train = h5py.File(fmri_train_file, 'r')['neural_data']


fmri_test_file = os.path.join('/scratch/jeffreykatab/berg', 'model_training_datasets',
    'train_dataset-things_fmri_1',
    'fmri_sub-03_split-test.h5')
fmri_test_all = h5py.File(fmri_test_file, 'r')['neural_data']

# Getting the image files names
train_stimuli_fmri = metadata_fmri['encoding_model']['train_stimuli']
test_stimuli_fmri = metadata_fmri['encoding_model']['test_stimuli']
unique_test_stimuli = np.unique(test_stimuli_fmri)

# Average the fMRI responses across repetitions of the same test stimulus
fmri_test = []
for stim in tqdm(unique_test_stimuli):
    idx = np.where(test_stimuli_fmri == stim)[0]
    fmri_test.append(fmri_test_all[idx].mean(0))
fmri_test = np.array(fmri_test)


# Subject 03 has 189,164 voxels voxels. We split them into 524 splits of 361 voxels each.
fmri_train = fmri_train[:, 361*(args.fmri_split - 1):361*args.fmri_split] # Each split has 361 voxels
fmri_test = fmri_test[:, 361*(args.fmri_split - 1):361*args.fmri_split] # Each split has 361 voxels

# Center and normalize the test fMRI responses (for later correlation)
eps = 1e-8
fmri_test_z = (fmri_test - fmri_test.mean(0)) / (fmri_test.std(0) + eps)
print("Shape of the fMRI data (train, test): ", fmri_train.shape, fmri_test.shape)

# ==============================================================================================
# Loading the MEG data
# For the whole-brain analysis, we append the MEG sensors responses across subjects by default.
# ==============================================================================================
# Loop across subjects
for ms, msub in enumerate(tqdm(args.meg_subjects)):

    # Loading the MEG metadata
    metadata_meg = berg.get_model_metadata(
        'meg-things_meg_1-vit_b_32',
        subject=msub
    )

    # Time point selection
    tmax = 0.8 # We limit our analysis to the first 800 ms after stimulus onset
    times = metadata_meg['meg']['times']
    time_idx = np.zeros(len(times), dtype=int)
    time_idx[times <= tmax] = 1
    time_idx = np.where(time_idx == 1)[0]
    times = times[times <= tmax]

    # Loading the MEG responses
    meg_train_file = os.path.join('/scratch/jeffreykatab/berg', 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_all_training_splits.h5')
    meg_train_sub = h5py.File(meg_train_file, 'r')['neural_data']


    meg_test_file = os.path.join('/scratch/jeffreykatab/berg', 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_split-test.h5')
    meg_test_all = h5py.File(meg_test_file, 'r')['neural_data']
    meg_test_all = meg_test_all[:,:,time_idx].astype(np.float32)

    # Averaging the MEG responses across repetitions for the images shared with the fMRI data
    test_stimuli_meg = metadata_meg['encoding_model']['test_stimuli']
    meg_test_sub = []
    for stim in unique_test_stimuli:
        idx = [i for i, x in enumerate(test_stimuli_meg) if x == stim]
        meg_test_sub.append(meg_test_all[idx].mean(0))
    meg_test_sub = np.array(meg_test_sub)
    
    

    # Getting the MEG responses for the images shared with the fMRI
    train_stimuli_meg = metadata_meg['encoding_model']['all_training_splits']['train_stimuli']
    idx_meg = []
    for stim in train_stimuli_fmri:
        idx_meg.append(train_stimuli_meg.index(stim))
    idx_meg = np.array(idx_meg)
    meg_train_sub = meg_train_sub[:,:,time_idx][idx_meg].astype(np.float32)

    # Appending the MEG sensor responses across subjects
    if ms == 0:
        meg_train = meg_train_sub
        meg_test = meg_test_sub
    else:
        meg_train = np.append(meg_train, meg_train_sub, 1)
        meg_test = np.append(meg_test, meg_test_sub, 1)
    del meg_train_sub, meg_test_all, meg_test_sub

print("Shape of the MEG data (train, test): ", meg_train.shape, meg_test.shape) 

# =============================================================================
# Training and testing the encoding fusion models
# =============================================================================
alphas = np.logspace(-6, 10, 17)
n_voxels = fmri_test.shape[1]
n_time_points = len(times)
correlations = np.zeros((n_time_points, n_voxels), dtype=np.float32)

print("Starting encoding fusion...")
for t in tqdm(range(len(times))):

    # Train the encoding fusion models
    meg2fmri = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    meg2fmri.fit(meg_train[:,:,t], fmri_train)

    # Generate the t-fMRI responses for the test images with in vivo MEG
    t_fmri = meg2fmri.predict(meg_test[:,:,t])

    # Center and normalize the t-fMRI responses
    t_fmri_z = (t_fmri - t_fmri.mean(0)) /  (t_fmri.std(0) + eps)

    # Correlate the t-fMRI test responses with the fMRI test responses
    correlations[t,:] = np.diag(t_fmri_z.T @ fmri_test_z) / len(t_fmri_z)

    # Delete unused variables
    del t_fmri, t_fmri_z, meg2fmri
    gc.collect()
del fmri_test, fmri_test_z, meg_train, meg_test
print("Encoding fusion complete!")

# =============================================================================
# Saving the encoding fusion correlations
# =============================================================================
save_dir = '/home/jeffreykatabo/Projects/fusion/THINGS/Encoding_Models/results/correlations/encoding_fusion_wb/fmri_sub-03'
os.makedirs(save_dir, exist_ok=True)

# Save the correlations
file_name = f'fmri_split-{args.fmri_split:03d}.npy'
np.save(os.path.join(save_dir, file_name), correlations)

# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")
