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
parser.add_argument('--radius', type=float, default=10.0)
alexnet_layers = [
    'features.2',    # Conv1 + Pool
    'features.5',    # Conv2 + Pool
    'features.7',    # Conv3
    'features.9',    # Conv4
    'features.12',   # Conv5 + Pool
    'classifier.2',  # FC6
    'classifier.5',  # FC7
    'classifier.6'   # FC8 (Output)
]
parser.add_argument('--layer', type=str, default='features.2', choices=alexnet_layers,
                    help='Layer of the Alexnet model from which the features are extracted for the commonality analysis.')
args = parser.parse_args()

print('>>> AlexNet Layer-wise Commonality Analysis <<<')
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

#====================================================================
# 3. Loading the layer features data and computing the features RDM
#====================================================================
features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/alexnet'
features_test = np.load(os.path.join(features_dir, f"sub-{args.subject:02d}_layerwise_fmaps.npy"), allow_pickle=True).item()[args.layer]['test']
print("Shape of the features data (test):", features_test.shape)
features_rdm = flatten_rdm(pairwise_distances(features_test, metric='cosine'))
print("Shape of the features RDM: ", features_rdm.shape)


# ==================================
# fMRI Noise ceilings
# ==================================
berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=args.subject)
wb_noise_ceilings = metadata['fmri'][f'{args.hemisphere}_ncsnr']
wb_noise_ceilings = wb_noise_ceilings[7802*(args.fmri_split - 1):7802*args.fmri_split]
n_time_points = eeg_rdms.shape[0]


# =============================================================================
# Loading the Precomputed fMRI RDMs
# =============================================================================
fmri_h5_file = os.path.join(
    f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/fmri_searchlight_rdms/radius-{args.radius}',
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
    r2_scores = np.zeros((n_time_points, n_vertices), dtype=np.float32)



    # =================================================================================================
    # Commonality Analysis (variance partitioning) for each vertex (neighbourhood): compute shared
    # variance explained by the EEG and features RDMs with respect to the fMRI RDM at that vertex
    # ==================================================================================================
    print("Commonality analysis loop has started...")

    for vertex in tqdm(range(n_vertices)):
        if wb_noise_ceilings[vertex]<0.2: # Skip the computations if the vertex doesn't pass the noise ceiling criterion
            continue
        else:
            fmri_rdm = fmri_rdms[vertex, :]
            features_model = LinearRegression().fit(features_rdm, fmri_rdm)
            features_r2 = features_model.score(features_rdm, fmri_rdm)
            
            for t in [51, 77, 102, 128, 153, 179, 205, 230, 256, 281, 307, 333, 358]:
                eeg_rdm = eeg_rdms[t]

                eeg_model = LinearRegression().fit(eeg_rdm.reshape(-1, 1), fmri_rdm)
                eeg_r2 = eeg_model.score(eeg_rdm.reshape(-1, 1), fmri_rdm)

                features_eeg_model = LinearRegression().fit(np.vstack((features_rdm, eeg_rdm)).T, fmri_rdm)
                features_eeg_r2 = features_eeg_model.score(np.vstack((features_rdm, eeg_rdm)).T, fmri_rdm)

                r2_scores[t, vertex] = features_r2 + eeg_r2 - features_eeg_r2

print("Commonality analysis complete!")

# =============================================================================
# 5. Saving Results
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/commonality_analysis/layerwise_alexnet/wb/subject-{args.subject}/layer-{args.layer}/hemisphere-{args.hemisphere}'
os.makedirs(save_dir, exist_ok=True)

file_name = f'fmri_split-{args.fmri_split}.npy'
np.save(os.path.join(save_dir, file_name), results)
print(f"Results successfully saved to: {os.path.join(save_dir, file_name)}")

# End time
execution_time = time.time() - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")