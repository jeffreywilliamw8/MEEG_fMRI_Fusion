import numpy as np
import os
import argparse
import time
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from utils import load_fmri_roi_data2

# Start time
start_time = time.time()

seed = 8
np.random.seed(seed)

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--roi', type=str, default='V1')
args = parser.parse_args()

print('>>> RSA Variance Partitioning: Vision vs. Language <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)  # k=1 excludes diagonal

# =============================================================================
# 1. Load fMRI Target Data and Compute Target RDM
# =============================================================================
_, fmri_test = load_fmri_roi_data2(args.subject, roi=args.roi, nc_threshold=0.2)

print('Shape of the fMRI data (test):', fmri_test.shape)
fmri_rdm = flatten_rdm(pairwise_distances(fmri_test, metric='correlation'))
print("Shape of the fMRI RDM: ", fmri_rdm.shape)
del fmri_test

# =============================================================================
# 2. Load EEG Predictor Data and Compute Time-Resolved RDMs
# =============================================================================
data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
eeg_test = np.load(os.path.join(data_path, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True).item()['eeg_test'] # Shape: (515, 30, 160, 359)
eeg_test = np.mean(eeg_test, axis=1) # Averaging across repeats -> (515, 160, 359)
print('Shape of the EEG data (test):', eeg_test.shape)

eeg_rdms = []
for t in tqdm(range(eeg_test.shape[2]), desc='Computing EEG RDMs'):
    rdm = pairwise_distances(eeg_test[:, :, t], metric='correlation')
    eeg_rdms.append(flatten_rdm(rdm))
eeg_rdms = np.array(eeg_rdms, dtype=np.float32) # Shape: (n_time_points, n_pairs)
print("Shape of the EEG RDMs: ", eeg_rdms.shape)
del eeg_test

# =============================================================================
# 3. Load Static Model Predictor Data (Vision DNN and LLM) and Compute RDMs
# =============================================================================
# Vision DNN
vision_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/vit_b_32'
vision_test = np.load(os.path.join(vision_dir, f'fmri_sub-{args.subject:02d}_fmaps.npy'), allow_pickle=True).item()['fmaps_test']
vision_rdm = flatten_rdm(pairwise_distances(vision_test, metric='cosine'))
print("Shape of the Vision DNN RDM: ", vision_rdm.shape)
del vision_test

# LLM Language Model
lang_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/llms'
lang_test = np.load(os.path.join(lang_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_test']
lang_rdm = flatten_rdm(pairwise_distances(lang_test, metric='cosine'))
print("Shape of the LLM RDM: ", lang_rdm.shape)
del lang_test

# =============================================================================
# 4. Variance Partitioning Analysis Loop
# =============================================================================
n_time_points = eeg_rdms.shape[0]

# Initialize storage dictionaries for your 4 requested variances
results = {
    'unique_vision': np.zeros(n_time_points, dtype=np.float32),
    'unique_language': np.zeros(n_time_points, dtype=np.float32),
    'shared_vision_language': np.zeros(n_time_points, dtype=np.float32)
}

print("Variance partitioning loop has started...")

# Static Baseline Step: Features only model -> R2(Y | Vis, Lang)
combined_features_rdm = np.vstack((vision_rdm, lang_rdm)).T

# Analogy to double-encoding: fit feature RDM to EEG RDM, and then predicted EEG RDM to fMRI RDM1
for t in tqdm(range(n_time_points), desc="Processing Timepoints"):
    eeg_rdm = eeg_rdms[t]


    # -------------------------------------------------------------------------
    # STAGE 1: Controlling for vision and language DNN features
    # -------------------------------------------------------------------------

    # 1. Controlling for language features
    eeg_minus_lang_rdm = eeg_rdm - LinearRegression().fit(lang_rdm.reshape(-1, 1), eeg_rdm).predict(lang_rdm.reshape(-1, 1)) # EEG RDM stripped of linear relationship with language features
    vision_minus_lang_rdm = vision_rdm - LinearRegression().fit(lang_rdm.reshape(-1, 1), vision_rdm).predict(lang_rdm.reshape(-1, 1)) # Vision RDM stripped of linear relationship with language features
    fmri_minus_lang_rdm = fmri_rdm - LinearRegression().fit(lang_rdm.reshape(-1, 1), fmri_rdm).predict(lang_rdm.reshape(-1, 1)) # fMRI RDM stripped of linear relationship with language features

    # 2. Controlling for vision features
    eeg_minus_vis_rdm = eeg_rdm - LinearRegression().fit(vision_rdm.reshape(-1, 1), eeg_rdm).predict(vision_rdm.reshape(-1, 1)) # EEG RDM stripped of linear relationship with language features
    lang_minus_vis_rdm = lang_rdm - LinearRegression().fit(vision_rdm.reshape(-1, 1), lang_rdm).predict(vision_rdm.reshape(-1, 1)) # Language RDM stripped of linear relationship with language features
    fmri_minus_vis_rdm = fmri_rdm - LinearRegression().fit(vision_rdm.reshape(-1, 1), fmri_rdm).predict(vision_rdm.reshape(-1, 1)) # fMRI RDM stripped of linear relationship with language features

    # -----------------------------------------------------------------------------
    # STAGE 2: Computing uniquely shared variances
    # i.e variance shared only by vision features, EEG and fMRI on the one hand,
    # and variance shared only by language features, EEG and fMRI on the one hand
    # ----------------------------------------------------------------------------

    # 3. Variance uniquely shared with vision features
    vision_model = LinearRegression().fit(vision_minus_lang_rdm.reshape(-1, 1), fmri_minus_lang_rdm)
    vision_r2 = vision_model.score(vision_minus_lang_rdm.reshape(-1, 1), fmri_minus_lang_rdm)
    
    eeg_model = LinearRegression().fit(eeg_minus_lang_rdm.reshape(-1, 1), fmri_minus_lang_rdm)
    eeg_r2 = eeg_model.score(eeg_minus_lang_rdm.reshape(-1, 1), fmri_minus_lang_rdm)
    
    vision_eeg_model = LinearRegression().fit(np.vstack((vision_minus_lang_rdm, eeg_minus_lang_rdm)).T, fmri_minus_lang_rdm)
    vision_eeg_r2 = vision_eeg_model.score(np.vstack((vision_minus_lang_rdm, eeg_minus_lang_rdm)).T, fmri_minus_lang_rdm)

    results['unique_vision'][t] = vision_r2 + eeg_r2 - vision_eeg_r2

    # 4. Variance uniquely shared with language features

    language_model = LinearRegression().fit(lang_minus_vis_rdm.reshape(-1, 1), fmri_minus_vis_rdm)
    language_r2 = language_model.score(lang_minus_vis_rdm.reshape(-1, 1), fmri_minus_vis_rdm)

    eeg_model = LinearRegression().fit(eeg_minus_vis_rdm.reshape(-1, 1), fmri_minus_vis_rdm)
    eeg_r2 = eeg_model.score(eeg_minus_vis_rdm.reshape(-1, 1), fmri_minus_vis_rdm)

    language_eeg_model = LinearRegression().fit(np.vstack((lang_minus_vis_rdm, eeg_minus_vis_rdm)).T, fmri_minus_vis_rdm)
    language_eeg_r2 = language_eeg_model.score(np.vstack((lang_minus_vis_rdm, eeg_minus_vis_rdm)).T, fmri_minus_vis_rdm)

    results['unique_language'][t] = language_r2 + eeg_r2 - language_eeg_r2

    # -----------------------------------------------------------------------------
    # STAGE 3: Computing jointly shared variances
    # i.e variance explained by vision and language features taken together
    # ----------------------------------------------------------------------------

    features_model = LinearRegression().fit(combined_features_rdm, fmri_rdm)
    features_r2 = features_model.score(combined_features_rdm, fmri_rdm)

    eeg_model = LinearRegression().fit(eeg_rdm.reshape(-1, 1), fmri_rdm)
    eeg_r2 = eeg_model.score(eeg_rdm.reshape(-1, 1), fmri_rdm)

    features_eeg_model = LinearRegression().fit(np.vstack((vision_rdm, lang_rdm, eeg_rdm)).T, fmri_rdm)
    features_eeg_r2 = features_eeg_model.score(np.vstack((vision_rdm, lang_rdm, eeg_rdm)).T, fmri_rdm)

    results['shared_vision_language'][t] = features_r2 + eeg_r2 - features_eeg_r2

print("Variance partitioning analysis complete!")

# =============================================================================
# 5. Saving Results
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/variance_partitioning/roi/subject-{args.subject}'
os.makedirs(save_dir, exist_ok=True)

file_name = f'{args.roi}.npy'
np.save(os.path.join(save_dir, file_name), results)
print(f"Results successfully saved to: {os.path.join(save_dir, file_name)}")

# End time
execution_time = time.time() - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")