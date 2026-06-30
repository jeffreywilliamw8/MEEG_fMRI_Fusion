import numpy as np
import os
import random
import argparse
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
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
parser.add_argument('--fmri_split', type=int, default=1)
args = parser.parse_args()

# =============================================================================
# Loading the weights and fMRI data
# =============================================================================
# f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/jefe_phase_2/wb/vision_language_models/dnn_type-{args.dnn_type}/subject-{args.subject}/hemisphere-{args.hemisphere}'
vision_weights_file = os.path.join(f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/jefe_phase_2/wb/vision_language_models/dnn_type-vdnn/subject-{args.subject}/hemisphere-{args.hemisphere}', f'fmri_split-{args.fmri_split}_cv_split-odd.npy')
language_weights_file = os.path.join(f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/jefe_phase_2/wb/vision_language_models/dnn_type-llm/subject-{args.subject}/hemisphere-{args.hemisphere}', f'fmri_split-{args.fmri_split}_cv_split-odd.npy')
combined_weights_file = os.path.join(f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/jefe_phase_2/wb/vision_language_models/dnn_type-both/subject-{args.subject}/hemisphere-{args.hemisphere}', f'fmri_split-{args.fmri_split}_cv_split-odd.npy')


vision_weights = np.load(vision_weights_file, allow_pickle=True).item()
language_weights = np.load(language_weights_file, allow_pickle=True).item()
combined_weights = np.load(combined_weights_file, allow_pickle=True).item()

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

combined_features = np.concatenate((vision_features, language_features), axis=1)

print("Shape of the stimulus features (VDNN, LLM, Combined): {}, {}, {}".format(vision_features.shape, language_features.shape, combined_features.shape))

# =============================================================================
# Setup Loop Parameters
# =============================================================================
n_time = len(vision_weights['coef_'])
n_vertices = vision_weights['coef_'][0].shape[0]
n_samples = vision_features.shape[0]

print("Number of time points: ", n_time)
print("Number of vertices: ", n_vertices)
print("Number of samples: ", n_samples)

# Initialize storage arrays for the 3 target tracks
partial_correlations = {
    "total_correlation": np.zeros((n_time, n_vertices)),
    "vision_partial_correlation": np.zeros((n_time, n_vertices)),            # Vision controlling for Language
    "language_partial_correlation": np.zeros((n_time, n_vertices))           # Language controlling for Vision
}

print("Starting Partial Correlation Analysis Loop")

for t in tqdm(range(n_time)):
    # 1. Compute visual/language t-fMRI model predictions across all vertices at time t
    vision_t_fmri_model = LinearRegression()
    language_t_fmri_model = LinearRegression()
    combined_t_fmri_model = LinearRegression()

    vision_t_fmri_model.coef_ = vision_weights['coef_'][t]
    vision_t_fmri_model.intercept_ = vision_weights['intercept_'][t]

    language_t_fmri_model.coef_ = language_weights['coef_'][t]
    language_t_fmri_model.intercept_ = language_weights['intercept_'][t]

    combined_t_fmri_model.coef_ = combined_weights['coef_'][t]
    combined_t_fmri_model.intercept_ = combined_weights['intercept_'][t]

    # Predict profiles shape: (n_samples, n_vertices)
    vision_t_fmri = vision_t_fmri_model.predict(vision_features)
    language_t_fmri = language_t_fmri_model.predict(language_features)
    combined_t_fmri = combined_t_fmri_model.predict(combined_features)

    vision_t_fmri = np.expand_dims(vision_t_fmri, 2)
    language_t_fmri = np.expand_dims(language_t_fmri, 2)
    combined_t_fmri = np.expand_dims(combined_t_fmri, 2)

    for vertex in range(n_vertices):

        x_vis = vision_t_fmri[:, vertex]
        x_lang = language_t_fmri[:, vertex]
        x_comb = combined_t_fmri[:, vertex]

        y = fmri_test[:, vertex]
        # 1. Vision partial correlation, controlling for language
        residual_1 = y - LinearRegression().fit(x_lang, y).predict(x_lang) # variance in fMRI unexplained by language
        residual_2 = x_vis - LinearRegression().fit(x_lang, x_vis).predict(x_lang) # Vision stripped of linear relation with language
        vision_partial_correlation = np.corrcoef(residual_1.flatten(), residual_2.flatten())[1,0]

        # 2. Language Partial Correlation, controlling for vision
        residual_1 = y - LinearRegression().fit(x_vis, y).predict(x_vis) # variance in fMRI unexplained by vision
        residual_2 = x_lang - LinearRegression().fit(x_vis, x_lang).predict(x_vis) # Language stripped of linear relation with vision
        language_partial_correlation = np.corrcoef(residual_1.flatten(), residual_2.flatten())[1,0]

        # 3. Total variance explained by vision and language DNNs combined
        total_correlation = np.corrcoef(y.flatten(), x_comb.flatten())[1,0]

        # 4. Store results
        partial_correlations["vision_partial_correlation"][t, vertex] = vision_partial_correlation
        partial_correlations["language_partial_correlation"][t, vertex] = language_partial_correlation
        partial_correlations["total_correlation"][t, vertex] = total_correlation
        

print("Partial correlation calculation complete!")

# =============================================================================
# Saving the metrics
# =============================================================================   
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/partial_correlation/wb/subject-{args.subject}/hemisphere-{args.hemisphere}'
os.makedirs(save_dir, exist_ok=True)

file_name = f'fmri_split-{args.fmri_split}.npy'
np.save(os.path.join(save_dir, file_name), partial_correlations)

# End time
end_time = time.time()
print(f"Execution complete! Total Time: {end_time - start_time:.2f} seconds.")