import numpy as np
import os
import random
import argparse
from sklearn.linear_model import RidgeCV
import time
import tqdm
from utils import load_fmri_data

# Start time
start_time = time.time()

# Random seed for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

#=================================================
# Input arguments
#=================================================

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=str, default='01')
parser.add_argument('--hemisphere', type=str, default='left')
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--eeg_frequency', type=int, default=100)
parser.add_argument('--eeg_channel_policy', type=str, default='append', 
                    help='Policy for handling EEG channels: "append" to use all channels, "average" to average across subjects',
                    choices=['append', 'average'])
args = parser.parse_args()

print(f'>>> Joint EEG-Features Encoding Fusion Phase 1 (Whole Brain) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#=================================================
# Loading the EEG data (even repeats for phase 1)
#=================================================
eeg_path = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/prepared_eeg/{args.eeg_frequency}_Hz'
if args.eeg_channel_policy == 'append':
    eeg_train = np.load(os.path.join(eeg_path, 'eeg_train_zsc_even_channel_policy-append.npy')) # Shape: (1000, 762, 185) or (1000, 762, 370)

elif args.eeg_channel_policy == 'average':
    eeg_train = np.load(os.path.join(eeg_path, 'eeg_train_zsc_even_channel_policy-average.npy')) # Shape: (1000, 127, 185) or (1000, 127, 370)

print('Shape of the EEG data (train):', eeg_train.shape)
#=================================================
# Loading the fMRI data
#=================================================
fmri_train, _ = load_fmri_data(args.fmri_subject, args.hemisphere, roi='WB')

# For computational efficiency and parallelization, we fit the model on subsets of the whole-brain vertices.
# Each subset contains 7802 vertices, and we iterate through 21 subsets to cover all 163842 vertices of each hemisphere.
fmri_train = fmri_train[:, 7802*(args.fmri_split - 1):7802*args.fmri_split]
print("Shape of the fMRI data (train):", fmri_train.shape)
#===========================================================================
# Fitting a linear model that predicts the responses of a group of vertices
# using all channels at each time point
#===========================================================================
alphas = np.logspace(-1, 1, 20) # List of alphas for Ridge regression
reg_param = {}
reg_param['coef_'] = []
reg_param['intercept_'] = []
print("Starting training")
for t in tqdm.tqdm(range(eeg_train.shape[2])):
    eeg2fmri = RidgeCV(alphas=alphas, store_cv_results=True)
    eeg2fmri.fit(eeg_train[:, :, t], fmri_train)
    # Store the encoding fusion model weights
    reg_param['coef_'].append(eeg2fmri.coef_.astype(np.float32))
    reg_param['intercept_'].append(eeg2fmri.intercept_.astype(np.float32))
print("Training complete!")

#=============================================================================
# Saving the correlations and regression weights
#==============================================================================    
save_dir = f'/scratch/jeffreykatab/Code/Encoding_Models/regression_weights/jefe_phase_1_wb/{args.eeg_frequency}_Hz/fmri_sub-{args.fmri_subject}/{args.hemisphere}_hemisphere'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = f'fmri_split-{args.fmri_split}.npy'

np.save(os.path.join(save_dir, file_name), reg_param)

# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")
