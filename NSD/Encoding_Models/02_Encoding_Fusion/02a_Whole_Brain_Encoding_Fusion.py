import numpy as np
import os
import random
import argparse
from sklearn.linear_model import RidgeCV
import time
import tqdm
from utils import load_fmri_hemi_data

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
args = parser.parse_args()

print(f'>>> EEG-fMRI Encoding Fusion (Whole-brain) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#=====================================================
# Loading the EEG responses 
#======================================================
data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
eeg_train = np.load(os.path.join(data_path, f'eeg_train_sub-{args.subject:02d}_trial_avg-all.npy'), allow_pickle=True).item()['eeg_train'] # Shape: (9000, 160, 359)
eeg_test = np.load(os.path.join(data_path, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True).item()['eeg_test'] # Shape: (515, 30, 160, 359)
eeg_test = np.mean(eeg_test, axis=1) # Averaging across repeats to get shape (515, 160, 359)
print('Shape of the EEG data (train, test):', eeg_train.shape, eeg_test.shape)
# =============================================================================
# Load the fMRI responses
# =============================================================================
fmri_train, fmri_test = load_fmri_hemi_data(args.subject, args.hemisphere) # Shape: (9000, 163842), (515, 163842)
fmri_train = fmri_train[:, 7802*(args.fmri_split - 1):7802*args.fmri_split]
fmri_test = fmri_test[:, 7802*(args.fmri_split - 1):7802*args.fmri_split]
fmri_test_z = (fmri_test - fmri_test.mean(0)) /  (fmri_test.std(0) + 1e-8)

print("Shape of the fMRI data (train, test):", fmri_train.shape, fmri_test.shape)

#=============================================================================
# Preparing to save the correlations
#=============================================================================    
file_name = f'fmri_split-{args.fmri_split:02d}.npy'
corrs_save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion/whole_brain/subject-{args.subject}/hemisphere-{args.hemisphere}'
if os.path.isdir(corrs_save_dir) == False:
	os.makedirs(corrs_save_dir)
corrs = []

#============================================================================
# Fitting a linear model that predicts the responses of a group of vertices
# using all EEG channels at each time point
#============================================================================
alphas = np.logspace(-6, 10, 20) # List of alphas for Ridge regression
print("Starting training...")
for t in tqdm.tqdm(range(eeg_train.shape[2])):
    eeg2fmri = RidgeCV(alphas=alphas, store_cv_results=True)
    eeg2fmri.fit(eeg_train[:, :, t], fmri_train)

    # Correlation between the predicted and actual fMRI responses on the training set
    #pred_fmri = eeg2fmri.predict(eeg_test[:, :, t])
    pred_fmri = eeg_test[:, :, t]@eeg2fmri.coef_.T + eeg2fmri.intercept_
    pred_fmri_z = (pred_fmri - pred_fmri.mean(0)) / (pred_fmri.std(0) + 1e-8)
    corr = np.diag(pred_fmri_z.T @ fmri_test_z) / len(pred_fmri_z)
    corrs.append(corr)
    np.save(os.path.join(corrs_save_dir, file_name), np.array(corrs, dtype=np.float32))
	
print("Training complete!")
corrs = np.array(corrs, dtype=np.float32)

# End time
end_time = time.time()
execution_time = end_time - start_time

print("Encoding Fusion complete!")
print(f"Execution time: {execution_time:.2f} seconds.")