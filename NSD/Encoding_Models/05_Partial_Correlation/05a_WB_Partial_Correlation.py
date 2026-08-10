"""
This script performs a partial correlation analysis to determine the unique contributions of the VDNN and LLM features to fMRI responses,
controlling for the other modality, using pre-computed regression weights to predict fMRI responses from vision and language features.
The analysis is performed at the whole-brain level

Parameters
----------
subject : int
    The subject ID for the fMRI data.
hemisphere : str
    The hemisphere to analyze ('lh' or 'rh').
cv_split : str
    The cross-validation split of EEG repeats to use ('even' or 'odd').
fmri_split : int
    The split index for fMRI vertices. The fMRI data is split into 21 splits, each containing 7802 vertices, to cover all 163842 fsaverage vertices per hemisphere
n_jobs : int
    The number of parallel jobs to run (-1 for all available cores).

"""


import os
# --- Set Thread Caps BEFORE importing joblib/sklearn to prevent core thrashing ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import random
import argparse
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from utils import load_fmri_hemi_data
import time

# Start time
start_time = time.time()

# Random seed for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh') # lh -> left hemisphere, rh -> right hemisphere
parser.add_argument('--cv_split', type=str, default='odd') # Even/odd cross-validation split
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel workers across time points (-1 = all available cores).')
args = parser.parse_args()

# =============================================================================
# Loading the weights and fMRI data
# =============================================================================
vision_weights_file = os.path.join(f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/jefe_phase_2/wb/vision_language_models/dnn_type-vdnn/subject-{args.subject}/hemisphere-{args.hemisphere}', f'fmri_split-{args.fmri_split}_cv_split-{args.cv_split}.npy')
language_weights_file = os.path.join(f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/jefe_phase_2/wb/vision_language_models/dnn_type-llm/subject-{args.subject}/hemisphere-{args.hemisphere}', f'fmri_split-{args.fmri_split}_cv_split-{args.cv_split}.npy')

vision_weights = np.load(vision_weights_file, allow_pickle=True).item()
language_weights = np.load(language_weights_file, allow_pickle=True).item()

# =============================================================================
# Load the fMRI responses (only the test set are necessary)
# =============================================================================
fmri_train, fmri_test = load_fmri_hemi_data(args.subject, args.hemisphere)
fmri_train = fmri_train[:, 7802*(args.fmri_split - 1):7802*args.fmri_split]
fmri_test = fmri_test[:, 7802*(args.fmri_split - 1):7802*args.fmri_split]
print('Shape of the fMRI data (train, test):', fmri_train.shape, fmri_test.shape)

# =============================================================================
# Loading the visual/language features data
# =============================================================================
vision_features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/vit_b_32'
vision_features = np.load(os.path.join(vision_features_dir, f'fmri_sub-{args.subject:02d}_fmaps.npy'), allow_pickle=True).item()['fmaps_test']

language_features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/llms'
language_features = np.load(os.path.join(language_features_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_test']

print("Shape of the stimulus features (VDNN, LLM): {}, {}".format(vision_features.shape, language_features.shape))

# =============================================================================
# Setup Loop Parameters
# =============================================================================
n_time = len(vision_weights['coef_'])
n_vertices = vision_weights['coef_'][0].shape[0]
n_samples = vision_features.shape[0]

print("Number of time points: ", n_time)
print("Number of vertices: ", n_vertices)
print("Number of samples: ", n_samples)

# =============================================================================
# Per-timepoint worker: identical computation as the original double loop's body for a
# single t, just factored into a function so it can be dispatched to joblib. Time points are
# fully independent of one another, so this is the natural axis to parallelize across.
# Defined at module level (not a nested closure) so it pickles cleanly for joblib's default
# 'loky' (multiprocessing) backend.
# =============================================================================
def compute_timepoint(t):
    vision_t_fmri_model = LinearRegression()
    language_t_fmri_model = LinearRegression()

    vision_t_fmri_model.coef_ = vision_weights['coef_'][t]
    vision_t_fmri_model.intercept_ = vision_weights['intercept_'][t]

    language_t_fmri_model.coef_ = language_weights['coef_'][t]
    language_t_fmri_model.intercept_ = language_weights['intercept_'][t]

    # Predict profiles shape: (n_samples, n_vertices)
    vision_t_fmri = vision_t_fmri_model.predict(vision_features)
    language_t_fmri = language_t_fmri_model.predict(language_features)

    vision_t_fmri = np.expand_dims(vision_t_fmri, 2)
    language_t_fmri = np.expand_dims(language_t_fmri, 2)

    vision_partial_row = np.zeros(n_vertices)
    language_partial_row = np.zeros(n_vertices)

    for vertex in range(n_vertices):
        x_vis = vision_t_fmri[:, vertex]
        x_lang = language_t_fmri[:, vertex]

        y = fmri_test[:, vertex]

        # 1. Vision partial correlation, controlling for language
        residual_1 = y - LinearRegression().fit(x_lang, y).predict(x_lang)  # variance in fMRI unexplained by language
        residual_2 = x_vis - LinearRegression().fit(x_lang, x_vis).predict(x_lang)  # Vision stripped of linear relation with language
        vision_partial_row[vertex] = np.corrcoef(residual_1.flatten(), residual_2.flatten())[1, 0]

        # 2. Language Partial Correlation, controlling for vision
        residual_1 = y - LinearRegression().fit(x_vis, y).predict(x_vis)  # variance in fMRI unexplained by vision
        residual_2 = x_lang - LinearRegression().fit(x_vis, x_lang).predict(x_vis)  # Language stripped of linear relation with vision
        language_partial_row[vertex] = np.corrcoef(residual_1.flatten(), residual_2.flatten())[1, 0]


    return t, vision_partial_row, language_partial_row


print(f"Starting Partial Correlation Analysis (parallelized across {n_time} time points, "
      f"n_jobs={args.n_jobs})...")
results = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(compute_timepoint)(t) for t in range(n_time)
)

# Initialize storage arrays for the 3 target tracks, then assemble by explicit timepoint index
# (robust regardless of the order results come back in, even though joblib.Parallel already
# preserves input order by default).
partial_correlations = {
    "vision_partial_correlation": np.zeros((n_time, n_vertices)),         # Vision controlling for Language
    "language_partial_correlation": np.zeros((n_time, n_vertices))           # Language controlling for Vision
}

for t, vision_row, language_row, total_row in results:
    partial_correlations["vision_partial_correlation"][t] = vision_row
    partial_correlations["language_partial_correlation"][t] = language_row

print("Partial correlation calculation complete!")

# =============================================================================
# Saving the metrics
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/partial_correlation/wb/subject-{args.subject}/hemisphere-{args.hemisphere}'
os.makedirs(save_dir, exist_ok=True)

file_name = f'fmri_split-{args.fmri_split}_cv_split-{args.cv_split}.npy'
np.save(os.path.join(save_dir, file_name), partial_correlations)

# End time
end_time = time.time()
print(f"Execution complete! Total Time: {end_time - start_time:.2f} seconds.")