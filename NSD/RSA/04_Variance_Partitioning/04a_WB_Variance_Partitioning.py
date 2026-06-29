import numpy as np
import os
import argparse
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import h5py
from berg import BERG
from sklearn.linear_model import LinearRegression
import time

# Start time
start_time = time.time()

seed = 8
np.random.seed(seed)

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh')
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--n_neighbours', type=int, default=100) 
args = parser.parse_args()

print('>>> RSA Variance Partitioning: Vision vs. Language <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)  # k=1 excludes diagonal


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

combined_features_rdm = np.vstack((vision_rdm, lang_rdm)).T

# ==================================
# fMRI Noise ceilings
# ==================================
berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=args.subject)
wb_noise_ceilings = metadata['fmri'][f'{args.hemisphere}_ncsnr']
wb_noise_ceilings = wb_noise_ceilings[7802*(args.fmri_split - 1):7802*args.fmri_split]

# =============================================================================
# 4. Variance Partitioning Analysis Loop
# =============================================================================
n_time_points = eeg_rdms.shape[0]



print("Variance partitioning loop has started...")

# =============================================================================
# Loading the Precomputed fMRI RDMs
# =============================================================================
fmri_h5_file = os.path.join(
    f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/fmri_searchlight_rdms/n_neighbours-{args.n_neighbours}',
    f'fmri_sub-{args.subject}_hemi-{args.hemisphere}_rdms.h5'
)
# Open the file in read mode
with h5py.File(fmri_h5_file, 'r') as f:
    # Point to the dataset (this doesn't load it into memory yet)
    dset = f['rdms']
    fmri_rdms = dset[7802*(args.fmri_split - 1):7802*args.fmri_split, :]
    n_vertices = fmri_rdms.shape[0]
    # Initialize storage dictionaries for your 4 requested variances
    results = {
        'unique_vision': np.zeros((n_time_points, n_vertices), dtype=np.float32),
        'unique_language': np.zeros((n_time_points, n_vertices), dtype=np.float32),
        'shared_vision_language': np.zeros((n_time_points, n_vertices), dtype=np.float32)
    }



    # =================================================================================================
    # Commonality Analysis (variance partitioning) for each vertex (neighbourhood): compute shared
    # variance explained by the EEG and features RDMs with respect to the fMRI RDM at that vertex
    # ==================================================================================================
    shared_variances = np.zeros((eeg_rdms.shape[0], n_vertices), dtype=np.float32)
    print("Commonality analysis loop has started...")

    for vertex in tqdm(range(n_vertices)):
        if wb_noise_ceilings[vertex]<0.2: # Skip the computations if the vertex doesn't pass the noise ceiling criterion
            continue
        else:
            fmri_rdm = fmri_rdms[vertex, :]
            
            for t in [51, 77, 102, 128, 153, 179, 205, 230, 256, 281, 307, 333, 358]:
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

                # 3. Variance uniquely shared with vision features
                vision_model = LinearRegression().fit(vision_minus_lang_rdm.reshape(-1, 1), fmri_minus_lang_rdm)
                vision_r2 = vision_model.score(vision_minus_lang_rdm.reshape(-1, 1), fmri_minus_lang_rdm)
                
                eeg_model = LinearRegression().fit(eeg_minus_lang_rdm.reshape(-1, 1), fmri_minus_lang_rdm)
                eeg_r2 = eeg_model.score(eeg_minus_lang_rdm.reshape(-1, 1), fmri_minus_lang_rdm)
                
                vision_eeg_model = LinearRegression().fit(np.vstack((vision_minus_lang_rdm, eeg_minus_lang_rdm)).T, fmri_minus_lang_rdm)
                vision_eeg_r2 = vision_eeg_model.score(np.vstack((vision_minus_lang_rdm, eeg_minus_lang_rdm)).T, fmri_minus_lang_rdm)

                results['unique_vision'][t, vertex] = vision_r2 + eeg_r2 - vision_eeg_r2

                # 4. Variance uniquely shared with language features

                language_model = LinearRegression().fit(lang_minus_vis_rdm.reshape(-1, 1), fmri_minus_vis_rdm)
                language_r2 = language_model.score(lang_minus_vis_rdm.reshape(-1, 1), fmri_minus_vis_rdm)

                eeg_model = LinearRegression().fit(eeg_minus_vis_rdm.reshape(-1, 1), fmri_minus_vis_rdm)
                eeg_r2 = eeg_model.score(eeg_minus_vis_rdm.reshape(-1, 1), fmri_minus_vis_rdm)

                language_eeg_model = LinearRegression().fit(np.vstack((lang_minus_vis_rdm, eeg_minus_vis_rdm)).T, fmri_minus_vis_rdm)
                language_eeg_r2 = language_eeg_model.score(np.vstack((lang_minus_vis_rdm, eeg_minus_vis_rdm)).T, fmri_minus_vis_rdm)

                results['unique_language'][t, vertex] = language_r2 + eeg_r2 - language_eeg_r2

                # -----------------------------------------------------------------------------
                # 3. Computing jointly shared variances i.e variance explained by vision and language features taken together
                # ----------------------------------------------------------------------------

                features_model = LinearRegression().fit(combined_features_rdm, fmri_rdm)
                features_r2 = features_model.score(combined_features_rdm, fmri_rdm)

                eeg_model = LinearRegression().fit(eeg_rdm.reshape(-1, 1), fmri_rdm)
                eeg_r2 = eeg_model.score(eeg_rdm.reshape(-1, 1), fmri_rdm)

                features_eeg_model = LinearRegression().fit(np.vstack((vision_rdm, lang_rdm, eeg_rdm)).T, fmri_rdm)
                features_eeg_r2 = features_eeg_model.score(np.vstack((vision_rdm, lang_rdm, eeg_rdm)).T, fmri_rdm)

                results['shared_vision_language'][t, vertex] = features_r2 + eeg_r2 - features_eeg_r2



print("Variance partitioning analysis complete!")

# =============================================================================
# 5. Saving Results
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/variance_partitioning/wb/subject-{args.subject}/hemisphere-{args.hemisphere}'
os.makedirs(save_dir, exist_ok=True)

file_name = f'fmri_split-{args.fmri_split}.npy'
np.save(os.path.join(save_dir, file_name), results)
print(f"Results successfully saved to: {os.path.join(save_dir, file_name)}")

# End time
execution_time = time.time() - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")