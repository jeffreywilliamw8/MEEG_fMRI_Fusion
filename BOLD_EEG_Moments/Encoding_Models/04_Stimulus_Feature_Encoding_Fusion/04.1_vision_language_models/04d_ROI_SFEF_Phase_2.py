"""
This script implements the second phase of the Stimulus feature encoding fusion Encoding Fusion (SFEF) for
the a specific Region of Interest (ROI), using either vision DNN features, language model features, or both. The second phase consists of using the pre-trained
EEG-to-fMRI encoding models from phase 1 to predict fMRI responses from EEG data in the held-out half of repeats,
and then training a second model to predict the predicted fMRI responses from DNN features.
The correlation coefficients between the predicted fMRI responses and the actual fMRI responses are computed and saved for each time point.
The encoding models' weights are also saved for the partial correlation analysis.

Parameters
----------
subject : int
    The number of the BMD participant pair for which the encoding fusion is performed.
hemisphere : str
    The hemisphere of the brain to analyze ('lh' for left hemisphere, 'rh' for right hemisphere).
roi : str
    The region of interest (ROI) to analyze (e.g., 'V1v', 'V1d', etc.).
cv_split : str
    The half of repeats to use for training the EEG-to-fMRI encoding model ('even' or 'odd'). The other half will be used for testing in phase 2 of the stimulus feature encoding fusion analysis.
dnn_type : str
    The type of DNN features to use for the stimulus encoding fusion: "vdnn" for vision DNN features, "llm" for language model features, or "both" for using both types of features (via concatenation).
n_jobs : int
    Number of parallel workers used across time points (-1 = all available cores).
"""


import numpy as np
import os
import random
import argparse
from sklearn.linear_model import LinearRegression, RidgeCV
import tqdm
from utils import load_fmri_data, load_eeg_data
import time

# --- Set Thread Caps BEFORE importing joblib to prevent core thrashing ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from joblib import Parallel, delayed

# Start time
start_time = time.time()

# Random seed for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

#======================================
# Input arguments
#======================================

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='left')
parser.add_argument('--roi', type=str, default='V1v')


parser.add_argument('--eeg_temporal_resolution', type=int, default=2, choices=[2, 4], help='Temporal resolution of EEG data in ms. ' \
'Default is 4 ms (250 Hz). The original prepared EEG data has ' \
'a resolution of 2 ms (500 Hz) -> 1800 time points, but it can be downsampled to 4 ms (900 time points) for faster processing.')

parser.add_argument('--cv_split', type=str, default='odd') # Even/odd cross-validation split
parser.add_argument('--dnn_type', type=str, default='vdnn', choices=['vdnn', 'llm'],
                    help='Type of DNN features to use for the stimulus feature encoding fusion: "vdnn" for vision DNN features, "llm" for language model features.')
parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()

print(f'>>> Stimulus Feature Encoding Fusion Phase 2 (ROI-wise) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#=====================================================
# Loading the EEG responses
#======================================================
cv_dict = {
        'even': 'odd', # If 'even' was used for phase 1, 'odd' weights will be used for phase 2, and vice-versa
        'odd': 'even'
    }

eeg_train, _ = load_eeg_data(trial_average=args.cv_split) # all 6 EEG subjects (channel-concatenated), responses averaged across even/odd trials


if args.eeg_temporal_resolution == 4:
    eeg_train = eeg_train[:, :, ::2] # Downsample the EEG data to 4 ms resolution (from 2 ms)
print('Shape of the EEG data (train):', eeg_train.shape)


# =============================================================================
# Load the fMRI responses (only the test set are necessary for phase 2)
# =============================================================================
_, fmri_test = load_fmri_data(args.subject, args.hemisphere, args.roi, threshold=20.0) # Shape: (1000, n_vertices)
print('Shape of the fMRI data (test):', fmri_test.shape)
if fmri_test.shape[1]>0:
    #fmri_test_z = (fmri_test - fmri_test.mean(0)) / (fmri_test.std(0) + 1e-8) # z-score the fMRI test responses for correlation computation

    #=======================================================================
    # Loading the pre-trained EEG-to-fMRI encoder's weights (from phase 1)
    #=======================================================================

    phase_1_weights_path = f'/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/regression_weights/stimulus_feature_encoding_fusion/roi/eeg_temporal_resolution-{args.eeg_temporal_resolution}ms/phase_1/subject-{args.subject}'
    phase_1_weights = np.load(os.path.join(phase_1_weights_path, f'{args.roi}_{args.hemisphere}_cv_split-{cv_dict[args.cv_split]}.npy'), allow_pickle=True).item()
    print("Loaded pre-trained EEG-to-fMRI encoder's weights")

    #=======================================================================
    # Loading the features data
    #=======================================================================
    features_dir = '/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/stimulus_features'
    if args.dnn_type == 'vdnn':
        features_train = np.load(os.path.join(features_dir, "vision_features_train.npy"))
        features_test = np.load(os.path.join(features_dir, "vision_features_test.npy"))
    elif args.dnn_type == 'llm':
        features_train = np.load(os.path.join(features_dir, "language_features_train.npy"))
        features_test = np.load(os.path.join(features_dir, "language_features_test.npy"))
        features_train = np.mean(features_train, axis=1)  # Average across descriptions for each stimulus
        features_test = np.mean(features_test, axis=1)  # Average across descriptions for each stimulus

    print("Shape of the features data (train, test):", features_train.shape, features_test.shape) # Should be (1000, 900), (102, 900) for visual features and (1000, 277), (102, 277) for language model features

    #=========================================================================
    # Settings for saving the correlation coefficients and regression weights
    # The weights will be used for variance partitioning
    #=========================================================================
    file_name = f'{args.roi}_{args.hemisphere}_cv_split-{args.cv_split}.npy'

    corrs_save_dir = f'/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/stimulus_feature_encoding_fusion/roi/eeg_temporal_resolution-{args.eeg_temporal_resolution}ms/phase_2/vision_language_models/dnn_type-{args.dnn_type}/subject-{args.subject}'
    if os.path.isdir(corrs_save_dir) == False:
        os.makedirs(corrs_save_dir)


    weights_save_dir = f'/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/regression_weights/stimulus_feature_encoding_fusion/roi/eeg_temporal_resolution-{args.eeg_temporal_resolution}ms/phase_2/vision_language_models/dnn_type-{args.dnn_type}/subject-{args.subject}'
    if os.path.isdir(weights_save_dir) == False:
        os.makedirs(weights_save_dir)


    #===========================================================================
    # Joint EEG-Feature Encoding Fusion: Predict fMRI from EEG (t-fMRI),
    # and then train model to predict t-fMRI from the features
    # Testing is done using the test fMRI responses
    #===========================================================================
    n_time = eeg_train.shape[2]
    encoding_models_weights = {}
    encoding_models_weights['coef_'] = [None] * n_time
    encoding_models_weights['intercept_'] = [None] * n_time
    correlations = []
    alphas = np.logspace(-6, 3, 20) # List of alphas for Ridge regression

    def fit_timepoint(t, eeg_t, phase_1_coef_t, phase_1_intercept_t, features_train, alphas):
        """Reconstruct the pre-trained phase-1 EEG-to-fMRI encoder for one time point, predict
        t-fMRI, then fit the phase-2 features-to-t-fMRI RidgeCV encoding model."""
        t_fmri = eeg_t @ phase_1_coef_t.T + phase_1_intercept_t # predict t-fMRI using matrix multiplication: faster than model.predict

        # Fitting a new linear regression model using the predicted t-fMRI as target
        encoding_model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
        encoding_model.fit(features_train, t_fmri)

        coef_ = encoding_model.coef_.astype(np.float32)
        intercept_ = encoding_model.intercept_.astype(np.float32)

        # Evaluating the encoding model and computing the correlation coefficients
        #ft_fmri = features_test @ encoding_model.coef_.T + encoding_model.intercept_ # predict ft-fMRI using matrix multiplication: faster than model.predict
        #ft_fmri_z = (ft_fmri - ft_fmri.mean(0)) / (ft_fmri.std(0) + 1e-8)
        #corrs = np.diag(ft_fmri_z.T @ fmri_test_z) / len(ft_fmri_z) # compute correlation between predicted and actual fMRI responses for each vertex using matrix multiplication: faster than np.corrcoef

        return t, coef_, intercept_

    print("Starting Stimulus Feature Encoding Fusion...")
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(fit_timepoint)(
            t, eeg_train[:, :, t], phase_1_weights['coef_'][t], phase_1_weights['intercept_'][t],
            features_train, alphas
        )
        for t in range(n_time)
    )

    # Storing the encoding fusion model weights
    for t, coef_, intercept_ in results:
        encoding_models_weights['coef_'][t] = coef_
        encoding_models_weights['intercept_'][t] = intercept_
    print("Stimulus Feature Encoding Fusion complete!")

    # Saving the correlation coefficients and regression weights to disk
    #np.save(os.path.join(corrs_save_dir, file_name), np.array(correlations, dtype=np.float32))
    np.save(os.path.join(weights_save_dir, file_name), encoding_models_weights)
    #print(f"Correlations saved to: {os.path.join(corrs_save_dir, file_name)}, shape={np.array(correlations).shape}")
    print(f"Regression weights saved to: {os.path.join(weights_save_dir, file_name)}")

else:
     print("No vertices above noise ceiling threshold found in this ROI. Terminating...")


# End time
end_time = time.time()
execution_time = end_time - start_time

print("JEFE Phase 2 complete!")
print(f"Execution time: {execution_time:.2f} seconds.")