import numpy as np
import os
import random
import argparse
from sklearn.linear_model import RidgeCV
import time
import tqdm
from utils import load_fmri_hemi_data, get_eeg_times
import h5py

# Start time
start_time = time.time()

# Random seed for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

#=====================================================
# Input arguments
#======================================================

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh') # lh -> left hemisphere, rh -> right hemisphere
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--cv_split', type=str, default='even') # Even/odd cross-validation split

args = parser.parse_args()

print(f'>>> Joint EEG-Feature Encoding Fusion Phase 1 (Whole Brain) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#=====================================================
# Loading the EEG responses 
#======================================================
data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
eeg_train = np.load(os.path.join(data_path, f'eeg_train_sub-{args.subject:02d}_trial_avg-{args.cv_split}.npy'), allow_pickle=True).item()['eeg_train'].astype(np.float32) # Shape: (9000, 160, 359)
print('Shape of the EEG data (train):', eeg_train.shape)
# Get the time points
times = get_eeg_times()
# =============================================================================
# Load the fMRI responses
# =============================================================================
fmri_train, _ = load_fmri_hemi_data(args.subject, args.hemisphere) # Shape: (9000, 7820), (515, 7820)
fmri_train = fmri_train[:, 7802*(args.fmri_split - 1):7802*args.fmri_split]
print('Shape of the fMRI data (train):', fmri_train.shape)

#=============================================================================
# Preparing to save the correlations and regression weights
#=============================================================================    
file_name = f'fmri_split-{args.fmri_split}_cv_split-{args.cv_split}.npy'

weights_save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/joint_eeg_feature_encoding/wb/phase_1/subject-{args.subject}/hemi-{args.hemisphere}'
if os.path.isdir(weights_save_dir) == False:
    os.makedirs(weights_save_dir)


#============================================================================
# Fitting a linear model that predicts the responses of a group of vertices
# using all EEG channels at each time point
#============================================================================
alphas = np.logspace(-6, 3, 20) # List of alphas for Ridge regression
weights = {}
weights['coef_'] = []
weights['intercept_'] = []

print("Starting encoding fusion...")
for t in tqdm.tqdm(range(len(times))):
    eeg2fmri = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    eeg2fmri.fit(eeg_train[:, :, t], fmri_train)

    # Storing the encoding fusion model weights
    weights['coef_'].append(eeg2fmri.coef_.astype(np.float32))
    weights['intercept_'].append(eeg2fmri.intercept_.astype(np.float32))
    np.save(os.path.join(weights_save_dir, file_name), weights)
print("Encoding fusion complete!")



# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")