"""
This script implements EEG-fMRI encoding fusion, by predicting fMRI responses or a specified ROI from EEG responses using Ridge regression. 
The script implements 2 forms of parallelization for efficient computing:
- using sbatch to run multiple jobs in parallel on the HPC, each job handling a different combination of the input arguments
- using joblib to parallelize the training and evaluation of the encoding models across timepoints within each execution of the script.

Parameters:
- subject: The number of the NSD participant pair for which the encoding fusion is performed.
- hemisphere: The hemisphere of the brain to analyze ('lh' for left hemisphere, 'rh' for right hemisphere).
- roi: The region of interest (ROI) to analyze. The fMRI data is filtered to include only vertices within the specified ROI and above 0.2 NCSNR threshold.
- n_jobs: The number of parallel jobs to run (using joblib). Set to -1 to use all available cores.

"""


import os
# --- Thread caps BEFORE importing joblib/sklearn, to prevent core oversubscription ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import random
import argparse
import time
from utils import load_fmri_roi_data, get_eeg_times
from sklearn.linear_model import RidgeCV
from fracridge import FracRidgeRegressorCV
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
parser.add_argument('--hemisphere', type=str, default='lh')  # lh -> left hemisphere, rh -> right hemisphere
parser.add_argument('--roi', type=str, default='V1v')
parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()

print(f'>>> EEG-fMRI Encoding Fusion (ROI)<<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =====================================================
# Loading the EEG responses
# ======================================================
data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
eeg_train = np.load(
    os.path.join(data_path, f'eeg_train_sub-{args.subject:02d}_trial_avg-all.npy'), allow_pickle=True
).item()['eeg_train'].astype(np.float32)  # Shape: (~9000, 160, 359)
eeg_test = np.load(
    os.path.join(data_path, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True
).item()['eeg_test']  # Shape: (515, 30, 160, 359)
eeg_test = np.mean(eeg_test, axis=1, dtype=np.float32)  # (515, 160, 359)
print('Shape of the EEG data (train, test):', eeg_train.shape, eeg_test.shape)

times = get_eeg_times()
n_time = len(times)

# =============================================================================
# Load the fMRI responses
# =============================================================================
fmri_train, fmri_test = load_fmri_roi_data(args.subject, args.hemisphere, args.roi, nc_threshold=0.20)
fmri_test_z = (fmri_test - fmri_test.mean(0)) / (fmri_test.std(0) + 1e-8)
print('Shape of the fMRI data (train, test):', fmri_train.shape, fmri_test.shape)

if fmri_train.shape[1] > 0: # Check if there are any vertices above the noise ceiling threshold in the specified ROI
    # =============================================================================
    # Preparing save paths
    # =============================================================================
    file_name = f'{args.roi}_{args.hemisphere}.npy'

    weights_save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/encoding_fusion/roi/fracridge/subject-{args.subject}'
    os.makedirs(weights_save_dir, exist_ok=True)

    corrs_save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion/roi/fracridge/subject-{args.subject}'
    os.makedirs(corrs_save_dir, exist_ok=True)

    # =============================================================================
    # Worker: fit + evaluate the encoding model for a single timepoint
    # =============================================================================
    alphas = np.logspace(-6, 10, 20)  # List of alphas for Ridge regression
    def fit_timepoint(t, eeg_train, eeg_test, fmri_train, fmri_test_z):
        #eeg2fmri = FracRidgeRegressorCV(fit_intercept=True)
        eeg2fmri = RidgeCV(alphas=alphas, alpha_per_target=True)
        eeg2fmri.fit(eeg_train[:, :, t], fmri_train)

        coef_ = eeg2fmri.coef_.astype(np.float32)
        intercept_ = eeg2fmri.intercept_.astype(np.float32)

        t_fmri = eeg_test[:, :, t] @ eeg2fmri.coef_.T + eeg2fmri.intercept_ # predict t-fMRI using matrix multiplication: faster than model.predict
        t_fmri_z = (t_fmri - t_fmri.mean(0)) / (t_fmri.std(0) + 1e-8)
        corr = np.diag(t_fmri_z.T @ fmri_test_z) / len(t_fmri_z) # compute correlation between predicted and actual fMRI responses for each vertex using matrix multiplication: faster than np.corrcoef

        return t, coef_, intercept_, corr.astype(np.float32)

    # =============================================================================
    # Dispatch all timepoints to the joblib pool
    # =============================================================================
    print("Starting encoding fusion (parallel)...")
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fit_timepoint)(t, eeg_train, eeg_test, fmri_train, fmri_test_z)
        for t in range(n_time)
    )

    weights = {
        'coef_': [r[1] for r in results],
        'intercept_': [r[2] for r in results],
    }
    corrs = np.array([r[3] for r in results], dtype=np.float32)
    np.save(os.path.join(weights_save_dir, file_name), weights)
    np.save(os.path.join(corrs_save_dir, file_name), corrs)

    print("Encoding fusion complete!")
    print(f"Weights saved to: {os.path.join(weights_save_dir, file_name)}")
    print(f"Correlations saved to: {os.path.join(corrs_save_dir, file_name)}, shape={corrs.shape}")

else:
    print("No vertices above noise ceiling threshold found in this ROI. Terminating...")

# End time
execution_time = time.time() - start_time
print("Encoding Fusion complete!")
print(f"Execution time: {execution_time:.2f} seconds.")