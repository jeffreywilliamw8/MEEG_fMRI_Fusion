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

######################################################
# Input arguments
######################################################

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=str, default='01')
parser.add_argument('--hemisphere', type=str, default='left')
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--eeg_frequency', type=int, default=100)
parser.add_argument('--eeg_channel_policy', type=str, default='average', 
                    help='Policy for handling EEG channels: "append" to use all channels, "average" to average across subjects',
                    choices=['append', 'average'])
args = parser.parse_args()

print(f'>>> EEG-fMRI Encoding Fusion (Whole Brain) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#################################################
# Loading the EEG data
#################################################
eeg_path = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/prepared_eeg/{args.eeg_frequency}_Hz'
if args.eeg_channel_policy == 'append':
    eeg_train = np.load(os.path.join(eeg_path, 'concat_eeg_train_762.npy')) # Shape: (1000, 762, 185) or (1000, 762, 370)
    eeg_test = np.load(os.path.join(eeg_path, 'concat_eeg_test_762.npy')) # Shape: (102, 762, 185) or (102, 762, 370)

elif args.eeg_channel_policy == 'average':
    eeg_train = np.load(os.path.join(eeg_path, 'eeg_train_subject_avg.npy')) # Shape: (1000, 127, 185) or (1000, 127, 370)
    eeg_test = np.load(os.path.join(eeg_path, 'eeg_test_subject_avg.npy')) # Shape: (102, 127, 185) or (102, 127, 370)

print('Shape of the EEG data (train, test):', eeg_train.shape, eeg_test.shape)
#################################
# Loading the fMRI data
################################
fmri_train, fmri_test = load_fmri_data(args.fmri_subject, args.hemisphere, roi='WB')

# For computational efficiency and parallelization, we fit the model on subsets of the whole-brain vertices.
# Each subset contains 7802 vertices, and we iterate through 21 subsets to cover all 163842 vertices of each hemisphere.
fmri_train = fmri_train[:, 7802*(args.fmri_split - 1):7802*args.fmri_split]
fmri_test = fmri_test[:, 7802*(args.fmri_split - 1):7802*args.fmri_split]
print("Shape of the fMRI data (train, test):", fmri_train.shape, fmri_test.shape)
######################################################################################################
# Fitting a linear model that predicts the responses of a group of vertices
# using all channels at each time point
######################################################################################################
corrs = []
alphas = np.logspace(-1, 1, 20) # List of alphas for Ridge regression
print("Starting training...")
for t in tqdm.tqdm(range(eeg_train.shape[2])):
    eeg2fmri = RidgeCV(alphas=alphas, store_cv_results=True)
    eeg2fmri.fit(eeg_train[:, :, t], fmri_train)
    pred_fmri = eeg2fmri.predict(eeg_test[:, :, t])
    corrs.append([np.corrcoef(pred_fmri[:, j], fmri_test[:, j])[0, 1] for j in range(fmri_test.shape[1])]) # Appending a list of 7802 correlation coefficients at each time point
print("Training complete!")
corrs = np.array(corrs, dtype=np.float32) # Shape: (n_timepoints, n_vertices)

#############################################################################
# Saving the correlations and regression weights
#############################################################################    
save_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/eeg2fmri_wb/{args.eeg_frequency}_Hz/eeg_channel_policy_{args.eeg_channel_policy}/fmri_sub-{args.fmri_subject}/{args.hemisphere}_hemisphere'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = f'fmri_split-{args.fmri_split}.npy'

np.save(os.path.join(save_dir, file_name), corrs)

# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")
