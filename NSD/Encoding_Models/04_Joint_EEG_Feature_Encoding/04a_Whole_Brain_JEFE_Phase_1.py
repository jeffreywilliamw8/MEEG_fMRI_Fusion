"""
This script performs the first phase of the stimulus feature encoding fusion (termed Joint EEG-Feature Encoding -JEFE- in the original McMahon paper)
The first phase consists of training encoding models to predict fMRI responses from EEG responses at each time point,
but using only EEG data averaged across either the even or odd half of total repeats. After training, the model weights are saved for later use in the
 second phase of the JEFE analysis, where the trained models are used to predict fMRI responses from EEG data in the held-out half of repeats, and then 
 the predicted fMRI responses are used to train a second model that predicts the predicted fMRI responses from stimulus features (DNN features).

Parameters
----------
subject : int
    The number of the NSD participant pair for which the encoding fusion is performed.
hemisphere : str
    The hemisphere of the brain to analyze ('lh' for left hemisphere, 'rh' for right hemisphere).
fmri_split : int
    The split index for fMRI vertices. The fMRI data is split into 21 splits, each containing 7802 vertices, to cover all 163842 fsaverage vertices per hemisphere
cv_split : str
    The half of repeats to use for training the EEG-to-fMRI encoding model ('even' or 'odd'). The other half will be used for testing in phase 2 of the JEFE analysis.

"""
import numpy as np
import os
import random
import argparse
from sklearn.linear_model import RidgeCV
from joblib import Parallel, delayed
import time
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
parser.add_argument('--n_jobs', type=int, default=-1) # Number of parallel workers across time points (-1 = all available cores)

args = parser.parse_args()

print(f'>>> Joint EEG-Feature Encoding Fusion Phase 1 (Whole Brain) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#=====================================================
# Loading the EEG responses
#======================================================
data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
eeg_train = np.load(os.path.join(data_path, f'eeg_train_sub-{args.subject:02d}_trial_avg-{args.cv_split}.npy'), allow_pickle=True).item()['eeg_train'].astype(np.float32) # Shape: (~9000, 160, 359)
print('Shape of the EEG data (train):', eeg_train.shape)
# Get the time points
times = get_eeg_times()
# =============================================================================
# Load the fMRI responses
# =============================================================================
fmri_train, _ = load_fmri_hemi_data(args.subject, args.hemisphere) # Shape: (~9000, 7820), (515, 7820)
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
# using all EEG channels at each time point -- parallelized across time points
# with joblib, since each time point's RidgeCV fit is fully independent of the
# others (same fmri_train target, different eeg_train[:, :, t] input).
#============================================================================
alphas = np.logspace(-6, 10, 20) # List of alphas for Ridge regression


def fit_timepoint(t):
    """Fit one time point's encoding model and return its (coef_, intercept_), both float32."""
    eeg2fmri = RidgeCV(alphas=alphas, alpha_per_target=True)
    eeg2fmri.fit(eeg_train[:, :, t], fmri_train)
    return eeg2fmri.coef_.astype(np.float32), eeg2fmri.intercept_.astype(np.float32)


print(f"Starting encoding fusion (parallelized across {len(times)} time points, n_jobs={args.n_jobs})...")
results = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(fit_timepoint)(t) for t in range(len(times))
)

weights = {
    'coef_': [r[0] for r in results],
    'intercept_': [r[1] for r in results],
}

np.save(os.path.join(weights_save_dir, file_name), weights)
print("Encoding fusion complete!")



# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")