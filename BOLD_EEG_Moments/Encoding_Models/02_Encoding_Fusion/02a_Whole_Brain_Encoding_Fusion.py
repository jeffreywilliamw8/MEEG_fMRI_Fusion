"""
This script implements EEG-fMRI encoding fusion, by predicting whole-brain fMRI responses from EEG responses using Ridge regression. 
The script implements 2 forms of parallelization for efficient computing:
- using sbatch to run multiple jobs in parallel on the HPC, each job handling a different combination of the input arguments
- using joblib to parallelize the training and evaluation of the encoding models across timepoints within each execution of the script.

-----------------------------
Parameters:
- subject: BMD fMRI participant ID
- hemisphere: The hemisphere of the brain to analyze.
- fmri_split: The split index for fMRI vertices. The fMRI data is split into 21 splits, each containing 7802 vertices, to cover all 163842 fsaverage vertices per hemisphere
- eeg_temporal_resolution: The temporal resolution of EEG data in ms. Default is 4 ms (250 Hz). The original prepared EEG data has a resolution of 2 ms (500 Hz) -> 1800 time points
    but it can be downsampled to 4 ms (900 time points) for faster processing.
- n_jobs: The number of parallel jobs to run (using joblib). Set to -1 to use all available cores.

"""



import os
# --- Thread caps before importing joblib/sklearn, to prevent core oversubscription ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import random
import argparse
import time
from sklearn.linear_model import RidgeCV
from fracridge import FracRidgeRegressorCV
from utils import load_fmri_data, load_eeg_data
from joblib import Parallel, delayed

# Start time
start_time = time.time()

# Random seed for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

# =====================================================
# Input arguments
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='left')
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--eeg_temporal_resolution', type=int, default=2, choices=[2, 4], help='Temporal resolution of EEG data in ms. ' \
'Default is 4 ms (250 Hz). The original prepared EEG data has ' \
'a resolution of 2 ms (500 Hz) -> 1800 time points, but it can be downsampled to 4 ms (900 time points) for faster processing.')

parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()

print(f'>>> EEG-fMRI Encoding Fusion (Whole-brain) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =====================================================
# Loading the EEG responses
# ======================================================
eeg_train, eeg_test = load_eeg_data() # Default data loading: all 6 EEG subjects (channel-concatenated), responses averaged across all trials
if args.eeg_temporal_resolution == 4:
    eeg_train = eeg_train[:, :, ::2] # Downsample the EEG data to 4 ms resolution (from 2 ms)
    eeg_test = eeg_test[:, :, ::2]
print('Shape of the EEG data (train, test):', eeg_train.shape, eeg_test.shape)

# =============================================================================
# Load the fMRI responses
# =============================================================================
fmri_train, fmri_test = load_fmri_data(args.subject, args.hemisphere, roi='WB') # Load whole-brain fMRI data
fmri_train = fmri_train[:, 7802 * (args.fmri_split - 1):7802 * args.fmri_split] # Split the fMRI data into 1 containing 7802 vertices
fmri_test = fmri_test[:, 7802 * (args.fmri_split - 1):7802 * args.fmri_split]
fmri_test_z = (fmri_test - fmri_test.mean(0)) / (fmri_test.std(0) + 1e-8) # Z-score the fMRI test data for fast correlation computation later
print("Shape of the fMRI data (train, test):", fmri_train.shape, fmri_test.shape)

# =============================================================================
# Preparing save path
# =============================================================================
file_name = f'fmri_split-{args.fmri_split:02d}.npy'
corrs_save_dir = f'/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/encoding_fusion/whole_brain/eeg_temporal_resolution-{args.eeg_temporal_resolution}ms/subject-{args.subject}/hemisphere-{args.hemisphere}'
os.makedirs(corrs_save_dir, exist_ok=True)

# =============================================================================
# Worker: fit + evaluate the encoding model for a single timepoint
# =============================================================================
alphas = np.logspace(-6, 12, 20)  # List of alphas for Ridge regression

def fit_timepoint(t, eeg_train, eeg_test, fmri_train, fmri_test_z):
    #eeg2fmri = FracRidgeRegressorCV(fit_intercept=True)
    eeg2fmri = RidgeCV(alphas=alphas, alpha_per_target=True)
    eeg2fmri.fit(eeg_train[:, :, t], fmri_train)

    t_fmri = eeg_test[:, :, t] @ eeg2fmri.coef_.T + eeg2fmri.intercept_ # predict t-fMRI using matrix multiplication: faster than model.predict
    #t_fmri = eeg2fmri.predict(eeg_test[:, :, t])
    t_fmri_z = (t_fmri - t_fmri.mean(0)) / (t_fmri.std(0) + 1e-8)
    corr = np.diag(t_fmri_z.T @ fmri_test_z) / len(t_fmri_z) # compute correlation between predicted and actual fMRI responses for each vertex using matrix multiplication: faster than np.corrcoef

    return t, corr.astype(np.float32)


# =============================================================================
# Dispatch all timepoints to the joblib pool
# =============================================================================
n_time = eeg_train.shape[2]
print("Starting training (parallel)...")

results = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(fit_timepoint)(t, eeg_train, eeg_test, fmri_train, fmri_test_z)
    for t in range(n_time)
)


corrs = np.array([r[1] for r in results], dtype=np.float32)
np.save(os.path.join(corrs_save_dir, file_name), corrs) # Save the correlation coefficients for all timepoints to disk



print("Training complete!")
print(f"Correlations saved to: {os.path.join(corrs_save_dir, file_name)}, shape={corrs.shape}")

# End time
execution_time = time.time() - start_time
print("Encoding Fusion complete!")
print(f"Execution time: {execution_time:.2f} seconds.")