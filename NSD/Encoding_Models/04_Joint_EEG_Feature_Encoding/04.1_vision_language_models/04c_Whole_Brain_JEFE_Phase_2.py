"""
This script implements the second phase of the Stimulus feature encoding fusion/Joint EEG-Feature Encoding Fusion (JEFE) for
the whole brain, using either vision DNN features, language model features, or both. The second phase consists of using the pre-trained
EEG-to-fMRI encoding models from phase 1 to predict fMRI responses from EEG data in the held-out half of repeats,
and then training a second model to predict the predicted fMRI responses from DNN features.
The correlation coefficients between the predicted fMRI responses and the actual fMRI responses are computed and saved for each time point.
The encoding models' weights are also saved for the partial correlation analysis.



Parameters
----------
subject : int
    The number of the NSD participant pair for which the encoding fusion is performed.
hemisphere : str
    The hemisphere of the brain to analyze ('lh' for left hemisphere, 'rh' for right hemisphere).
fmri_split : int
    The split index for fMRI vertices. The fMRI data is split into 21 splits, each containing 7802 vertices, to cover all 163842 fsaverage vertices per hemisphere
cv_split : str
    The half of repeats to use for training the EEG-to-fMRI encoding model ('even' or 'odd'). The other half will be used for testing in phase 2 of the joint EEG-feature encoding fusion analysis.
dnn_type : str
    The type of DNN features to use for the joint encoding fusion: "vdnn" for vision DNN features, "llm" for language model features, or "both" for using both types of features (via concatenation).
n_jobs : int
    Number of parallel workers used across time points (-1 = all available cores).
"""



import numpy as np
import os
import random
import argparse
from sklearn.linear_model import LinearRegression, RidgeCV
from joblib import Parallel, delayed
import time
from utils import load_fmri_hemi_data

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
parser.add_argument('--hemisphere', type=str, default='lh') # lh -> left hemisphere, rh -> right hemisphere
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--cv_split', type=str, default='odd') # Even/odd cross-validation split
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--dnn_type', type=str, default='vdnn', choices=['vdnn', 'llm', 'both'],
                    help='Type of DNN features to use for the joint encoding fusion: "vdnn" for vision DNN features, "llm" for language model features.')
parser.add_argument('--n_jobs', type=int, default=-1,
                    help='Number of parallel workers across time points (-1 = all available cores).')
args = parser.parse_args()

print(f'>>> Joint EEG-Features Encoding Fusion Phase 2 (Whole-Brain) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#=====================================================
# Loading the EEG responses (odd repeats for phase 2)
#======================================================
data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
cv_dict = {
        'even': 'odd', # If 'even' was used for phase 1, 'odd' will be used for phase 2, and vice-versa
        'odd': 'even'
    }
eeg_train = np.load(os.path.join(data_path, f'eeg_train_sub-{args.subject:02d}_trial_avg-{args.cv_split}.npy'), allow_pickle=True).item()['eeg_train'] # Shape: (~9000, 160, 359)
print('Shape of the EEG data (train):', eeg_train.shape)


# =============================================================================
# Load the fMRI responses (only the test set is necessary for phase 2)
# =============================================================================
_, fmri_test = load_fmri_hemi_data(args.subject, args.hemisphere) # training fMRI responses are not needed for phase 2
fmri_test = fmri_test[:, 7802*(args.fmri_split - 1):7802*args.fmri_split]
fmri_test_z = (fmri_test - fmri_test.mean(0)) / (fmri_test.std(0) + 1e-8) # z-score the fMRI test responses for correlation computation
print('Shape of the fMRI data (test):', fmri_test.shape)

#=======================================================================
# Loading the pre-trained EEG-to-fMRI encoder's weights (from phase 1)
#=======================================================================
phase_1_weights_path = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/joint_eeg_feature_encoding/wb/phase_1/subject-{args.subject}/hemi-{args.hemisphere}'
phase_1_weights = np.load(os.path.join(phase_1_weights_path, f'fmri_split-{args.fmri_split}_cv_split-{cv_dict[args.cv_split]}.npy'), allow_pickle=True).item()
print("Loaded pre-trained EEG-to-fMRI encoder's weights")

#=======================================================================
# Loading the features data
#=======================================================================
if args.dnn_type == 'vdnn':
    features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/vit_b_32'
    features_train = np.load(os.path.join(features_dir, f'fmri_sub-{args.subject:02d}_fmaps.npy'), allow_pickle=True).item()['fmaps_train']
    features_test = np.load(os.path.join(features_dir, f'fmri_sub-{args.subject:02d}_fmaps.npy'), allow_pickle=True).item()['fmaps_test']
elif args.dnn_type == 'llm':
    features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/llms'
    features_train = np.load(os.path.join(features_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_train']
    features_test = np.load(os.path.join(features_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_test']
elif args.dnn_type == 'both':
    vision_features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/vit_b_32'
    vision_features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/vit_b_32'
    vision_features_train = np.load(os.path.join(vision_features_dir, f'fmri_sub-{args.subject:02d}_fmaps.npy'), allow_pickle=True).item()['fmaps_train']
    vision_features_test = np.load(os.path.join(vision_features_dir, f'fmri_sub-{args.subject:02d}_fmaps.npy'), allow_pickle=True).item()['fmaps_test']

    language_features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/llms'
    language_features_train = np.load(os.path.join(language_features_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_train']
    language_features_test = np.load(os.path.join(language_features_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_test']

    features_train = np.concatenate((vision_features_train, language_features_train), axis=1)
    features_test = np.concatenate((vision_features_test, language_features_test), axis=1)

print("Shape of the features data (train, test):", features_train.shape, features_test.shape) # Should be (9000, 250), (515, 250)

#=========================================================================
# Settings for saving the correlation coefficients and regression weights
# The weights will be used for variance partitioning
#=========================================================================
file_name = f'fmri_split-{args.fmri_split}_cv_split-{args.cv_split}.npy'

corrs_save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/jefe_phase_2/wb/vision_language_models/dnn_type-{args.dnn_type}/subject-{args.subject}/hemisphere-{args.hemisphere}'
if os.path.isdir(corrs_save_dir) == False:
    os.makedirs(corrs_save_dir)

weights_save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/jefe_phase_2/wb/vision_language_models/dnn_type-{args.dnn_type}/subject-{args.subject}/hemisphere-{args.hemisphere}'
if os.path.isdir(weights_save_dir) == False:
    os.makedirs(weights_save_dir)

#===========================================================================
# Joint EEG-Feature Encoding Fusion: Predict fMRI from EEG (t-fMRI),
# and then train model to predict t-fMRI from the features
# Testing is done using the test fMRI responses
#
# Parallelized across time points: each t independently (1) reconstructs the phase-1
# EEG-to-fMRI encoder from its stored weights, (2) predicts t-fMRI from EEG, (3) fits a new
# RidgeCV mapping features -> t-fMRI, and (4) computes the per-vertex test-set correlations.
#===========================================================================
alphas = np.logspace(-6, 3, 20) # List of alphas for Ridge regression


def fit_timepoint(t):
    """Run one time point's phase-2 fit and evaluation, returning 
    the encoding model's weights and the vertex-wise correlation coefficients."""

    # Reconstructing the pre-trained phase-1 EEG-to-fMRI encoder for this time point
    t_fmri = eeg_train[:, :, t] @ phase_1_weights['coef_'][t].T + phase_1_weights['intercept_'][t].intercept_ # predict t-fMRI using matrix multiplication: faster than model.predict

    # Fitting a new linear regression model using the predicted t-fMRI as target
    encoding_model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    encoding_model.fit(features_train, t_fmri)

    coef_ = encoding_model.coef_.astype(np.float32)
    intercept_ = encoding_model.intercept_.astype(np.float32)

    # Evaluating the encoding model and computing the correlation coefficients
    pred_t_fmri = features_test @ encoding_model.coef_.T + encoding_model.intercept_ # predict t-fMRI using matrix multiplication: faster than model.predict
    pred_t_fmri_z = (pred_t_fmri - pred_t_fmri.mean(0)) / (pred_t_fmri.std(0) + 1e-8)
    corrs = np.diag(pred_t_fmri_z.T @ fmri_test_z) / len(pred_t_fmri_z) # compute correlation between predicted and actual fMRI responses for each vertex using matrix multiplication: faster than np.corrcoef

    return coef_, intercept_, corrs


print(f"Starting Joint EEG-Feature Encoding Fusion (parallelized across {eeg_train.shape[2]} "
      f"time points, n_jobs={args.n_jobs})...")
results = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(fit_timepoint)(t) for t in range(eeg_train.shape[2])
)

encoding_models_weights = {
    'coef_': [r[0] for r in results],
    'intercept_': [r[1] for r in results],
}
corrs = [r[2] for r in results]

np.save(os.path.join(weights_save_dir, file_name), encoding_models_weights)
np.save(os.path.join(corrs_save_dir, file_name), np.array(corrs, dtype=np.float32))
print("Joint EEG-Feature Encoding Fusion complete!")


# End time
end_time = time.time()
execution_time = end_time - start_time

print("JEFE Phase 2 complete!")
print(f"Execution time: {execution_time:.2f} seconds.")