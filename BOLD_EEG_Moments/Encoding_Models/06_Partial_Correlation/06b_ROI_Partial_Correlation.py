"""
This script computes the unique (partial-correlation) contribution of vision-DNN and
language-model features to the Stimulus Feature Encoding Fusion (SFEF), for a specific ROI, using
the phase-2 encoding models' weights already fit and saved separately for dnn_type='vdnn' and
dnn_type='llm'. For each time point and vertex, both models' predicted fMRI responses are
correlated with the actual (held-out) test fMRI responses and with each other, and the standard
first-order partial correlation formula is used to isolate the variance uniquely explained by each
feature type once the other is controlled for. No total/shared correlation is computed.

Parameters
----------
subject : int
    The number of the BMD participant pair for which the partial correlation is computed.
hemisphere : str
    The hemisphere of the brain to analyze ('lh' for left hemisphere, 'rh' for right hemisphere).
roi : str
    The region of interest (ROI) to analyze (e.g., 'V1v', 'V1d', etc.).
cv_split : str
    The half of repeats the phase-2 weights were fit on ('even' or 'odd') -- must match the
    cv_split used when running Stimulus_Feature_Encoding_Fusion_Phase2_ROI.py for both dnn_types.
eeg_temporal_resolution : int
    Temporal resolution (ms) of the phase-2 weights being loaded -- must match phase 2.
"""

import numpy as np
import os
import argparse
from utils import load_fmri_data
import time

start_time = time.time()

#======================================
# Input arguments
#======================================

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='left')
parser.add_argument('--roi', type=str, default='V1v')
parser.add_argument('--eeg_temporal_resolution', type=int, default=2, choices=[2, 4])
parser.add_argument('--cv_split', type=str, default='odd')
args = parser.parse_args()

print(f'>>> Stimulus Feature Encoding Fusion Phase 2 (ROI-wise) -- Unique (Partial) Correlations <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =============================================================================
# Load the actual (ground-truth) fMRI test responses
# =============================================================================
_, fmri_test = load_fmri_data(args.subject, args.hemisphere, args.roi, threshold=20.0)  # Shape: (102, n_vertices)
print('Shape of the fMRI data (test):', fmri_test.shape)

if fmri_test.shape[1] > 0:
    fmri_test_z = (fmri_test - fmri_test.mean(0)) / (fmri_test.std(0) + 1e-8)

    #=======================================================================
    # Loading the phase-2 vision-DNN and language-model encoding models' weights
    #=======================================================================
    weights_dir = f'/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/regression_weights/stimulus_feature_encoding_fusion/roi/eeg_temporal_resolution-{args.eeg_temporal_resolution}ms/phase_2/vision_language_models'
    file_name = f'{args.roi}_{args.hemisphere}_cv_split-{args.cv_split}.npy'

    vdnn_weights = np.load(os.path.join(weights_dir, 'dnn_type-vdnn', f'subject-{args.subject}', file_name), allow_pickle=True).item()
    llm_weights = np.load(os.path.join(weights_dir, 'dnn_type-llm', f'subject-{args.subject}', file_name), allow_pickle=True).item()
    print("Loaded phase-2 vision-DNN and language-model encoding models' weights")

    n_time = len(vdnn_weights['coef_'])
    assert n_time == len(llm_weights['coef_']), \
        f"Time point mismatch between vdnn ({n_time}) and llm ({len(llm_weights['coef_'])}) phase-2 weights."

    #=======================================================================
    # Loading the DNN features (test set only -- needed to reconstruct each model's predictions)
    #=======================================================================
    features_dir = '/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/stimulus_features'
    features_test_vdnn = np.load(os.path.join(features_dir, "vision_features_test.npy"))
    features_test_llm = np.load(os.path.join(features_dir, "language_features_test.npy"))
    features_test_llm = np.mean(features_test_llm, axis=1)  # Average across descriptions for each stimulus

    print("Shape of the features data (vdnn, llm):", features_test_vdnn.shape, features_test_llm.shape)

    #=========================================================================
    # Settings for saving the unique (partial) correlation coefficients
    #=========================================================================
    corrs_save_dir = f'/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/stimulus_feature_encoding_fusion/roi/eeg_temporal_resolution-{args.eeg_temporal_resolution}ms/phase_2/vision_language_models/partial_correlation/subject-{args.subject}'
    if os.path.isdir(corrs_save_dir) == False:
        os.makedirs(corrs_save_dir)

    #===========================================================================
    # For each time point: reconstruct both models' predicted fMRI responses, correlate each
    # with the actual test fMRI responses and with each other (vertex-wise), then compute the
    # first-order partial correlation isolating each feature type's unique contribution.
    #===========================================================================
    n_vertices = fmri_test.shape[1]
    unique_vdnn = np.zeros((n_time, n_vertices), dtype=np.float32)
    unique_llm = np.zeros((n_time, n_vertices), dtype=np.float32)

    def zscore(x):
        return (x - x.mean(0)) / (x.std(0) + 1e-8)

    def vertex_corr(a_z, b_z):
        """Vertex-wise Pearson correlation between two already z-scored (n_stimuli, n_vertices)
        matrices, via the diag(A.T @ B) / n trick (faster than looping np.corrcoef per vertex)."""
        return np.diag(a_z.T @ b_z) / len(a_z)

    print("Computing unique (partial) correlations...")
    for t in range(n_time):
        ft_fmri_vdnn = features_test_vdnn @ vdnn_weights['coef_'][t].T + vdnn_weights['intercept_'][t]
        ft_fmri_llm = features_test_llm @ llm_weights['coef_'][t].T + llm_weights['intercept_'][t]

        ft_fmri_vdnn_z = zscore(ft_fmri_vdnn)
        ft_fmri_llm_z = zscore(ft_fmri_llm)

        r_x_vdnn = vertex_corr(fmri_test_z, ft_fmri_vdnn_z)
        r_x_llm = vertex_corr(fmri_test_z, ft_fmri_llm_z)
        r_vdnn_llm = vertex_corr(ft_fmri_vdnn_z, ft_fmri_llm_z)

        # First-order partial correlation: unique contribution of one predictor, controlling for the other
        unique_vdnn[t] = (r_x_vdnn - r_x_llm * r_vdnn_llm) / np.sqrt((1 - r_x_llm ** 2) * (1 - r_vdnn_llm ** 2))
        unique_llm[t] = (r_x_llm - r_x_vdnn * r_vdnn_llm) / np.sqrt((1 - r_x_vdnn ** 2) * (1 - r_vdnn_llm ** 2))

    print("Unique (partial) correlation computation complete!")

    # Saving the unique correlation coefficients to disk
    partial_correlations = {'unique_vdnn': unique_vdnn, 'unique_llm': unique_llm}
    np.save(os.path.join(corrs_save_dir, file_name), partial_correlations)
    print(f"Unique (partial) correlations saved to: {os.path.join(corrs_save_dir, file_name)}")

else:
    print("No vertices above noise ceiling threshold found in this ROI. Terminating...")

# End time
end_time = time.time()
execution_time = end_time - start_time

print("SFEF Phase 2 (unique/partial correlations) complete!")
print(f"Execution time: {execution_time:.2f} seconds.")