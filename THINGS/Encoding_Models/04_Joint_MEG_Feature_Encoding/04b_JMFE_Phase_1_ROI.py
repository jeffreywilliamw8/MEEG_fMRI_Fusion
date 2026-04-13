import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
import random
from sklearn.linear_model import RidgeCV
import time

# Start time
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--berg_dir', default='/scratch/jeffreykatab/Code/Encoding_Models/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print(f'>>> Whole-Brain MEG-fMRI Fusion (Training) <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Loading the fMRI data
# =============================================================================
# Load the fMRI responses
fmri_dir = os.path.join('/scratch/jeffreykatab/berg', 'model_training_datasets',
    'train_dataset-things_fmri_1',
    f'fmri_sub-{args.fmri_subject:02d}_split-train.h5')
fmri_train = h5py.File(fmri_dir, 'r')['neural_data']

# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-things_fmri_1-vit_b_32',
    subject=args.fmri_subject)

roi_idx = metadata_fmri['roi'][args.roi]
# We use half of the training samples for phase 1
fmri_train = fmri_train[4320:, roi_idx].astype(np.float32) # Shape: (4320, n_voxels)

# To run this analysis using the other half of the data (for later cross-validation in the GC analysis),
# we also use the [4320:] partition
print("Shape of the fMRI data: {}".format(fmri_train.shape))


# Get the image files names
train_stimuli_fmri = metadata_fmri['encoding_model']['train_stimuli']


# =============================================================================
# Loading the MEG data
# =============================================================================
# Loop across subjects
for ms, msub in enumerate(tqdm(args.meg_subjects)):

    # Load the MEG metadata
    metadata_meg = berg.get_model_metadata(
        'meg-things_meg_1-vit_b_32',
        subject=msub
    )

    # Time point selection
    tmax = 0.8
    times = metadata_meg['meg']['times']
    time_idx = np.zeros(len(times), dtype=int)
    time_idx[times <= tmax] = 1
    time_idx = np.where(time_idx == 1)[0]
    times = times[times <= tmax]

    # Load the MEG responses
    meg_train_dir = os.path.join('/scratch/jeffreykatab/berg', 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_all_training_splits.h5')
    
    # meg_P{msub}_all_training_splits.h5 or meg_P{msub}_split-train.h5
    meg_train_sub = h5py.File(meg_train_dir, 'r')['neural_data']

    # Get the MEG responses for the images shared with the fMRI
    train_stimuli_meg = metadata_meg['encoding_model']['all_training_splits']\
        ['train_stimuli']
    idx_meg = []
    for stim in train_stimuli_fmri:
        idx_meg.append(train_stimuli_meg.index(stim))
    idx_meg = np.array(idx_meg)
    meg_train_sub = meg_train_sub[:,:,time_idx][idx_meg].astype(np.float32)

    # Append the MEG sensor responses across subjects
    if ms == 0:
        meg_train = meg_train_sub
    else:
        meg_train = np.append(meg_train, meg_train_sub, 1)
    del meg_train_sub

meg_train = meg_train[4320:,:,:] # Shape: (4320, n_sensors_across_subjects, n_time_points)
# =============================================================================
# Train the encoding fusion models
# =============================================================================
reg_param = {}
reg_param['coef_'] = []
reg_param['intercept_'] = []
reg_param['alpha_'] = []
reg_param['n_features_in_'] = []

# Loop across MEG time points
alphas = np.logspace(-6, 10, 17)

print("Shape of the MEG train data: ", meg_train.shape)
print("Shape of the fMRI train data: ", fmri_train.shape)

print("Starting training")
for t in tqdm(range(len(times))):

    # Train the encoding fusion models
    reg = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    reg.fit(meg_train[:,:,t], fmri_train)
    # Store the encoding fusion model weights
    reg_param['coef_'].append(reg.coef_.astype(np.float32))
    reg_param['intercept_'].append(reg.intercept_.astype(np.float32))
    reg_param['alpha_'].append(reg.alpha_.astype(np.float32))
    reg_param['n_features_in_'].append(reg.n_features_in_)

print("Training complete!")

# =============================================================================
# Saving the encoding fusion model weights
# =============================================================================
# Creating the encoding fusion model weight save directory
# If using the [:4320], save to 'half_1'; if using the [4320:] partition, save to 'half_2'
# By default, 'half_1' will be used, but the analysis must also be run on half 2 to have 2 independent
# sets of MEG-fMRI model weights
save_dir_weights = os.path.join('/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models', 'jmfe_phase_1', 'roi', 'half_2', f'fmri_sub-{args.fmri_subject:02d}')
os.makedirs(save_dir_weights, exist_ok=True)

# Saving the weights
file_name = f'{args.roi}.npy'
np.save(os.path.join(save_dir_weights, file_name), reg_param)


# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")
