import numpy as np
import os
import argparse
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import h5py
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
parser.add_argument('--dnn_type', type=str, default='vdnn', choices=['vdnn', 'llm'],
                    help='Type of DNN features to use for the joint encoding fusion: "vdnn" for vision DNN features, "llm" for language model features.')
args = parser.parse_args()


print('>>> Searchlight Whole-Brain Commonality Analysis <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)  # k=1 excludes diagonal

# =============================================================================================================
# Loading the EEG responses (test set) and computing the EEG RDMs
# =============================================================================================================
data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
eeg_test = np.load(os.path.join(data_path, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True).item()['eeg_test'] # Shape: (515, 30, 160, 359)
eeg_test = np.mean(eeg_test, axis=1) # Averaging across repeats to get shape (515, 160, 359)
print('Shape of the EEG data (test):', eeg_test.shape)
eeg_rdms = []
for t in tqdm(range(eeg_test.shape[2]), desc='Computing EEG RDMs'):
    rdm = pairwise_distances(eeg_test[:,:,t], metric='correlation')
    eeg_rdms.append(flatten_rdm(rdm))
eeg_rdms = np.array(eeg_rdms, dtype=np.float32) # Shape: (n_time_points, n_pairs)
print("Shape of the EEG RDMs: ", eeg_rdms.shape)
del eeg_test

#=======================================================================
# Loading the features data (test set) and computing the features RDMs
#=======================================================================
if args.dnn_type == 'vdnn':
    features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/vit_b_32'
    features_test = np.load(os.path.join(features_dir, f'fmri_sub-{args.subject:02d}_fmaps.npy'), allow_pickle=True).item()['fmaps_test']
elif args.dnn_type == 'llm':
    features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/llms'
    features_test = np.load(os.path.join(features_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_test']

print("Shape of the features data (test):", features_test.shape)
features_rdm = flatten_rdm(pairwise_distances(features_test, metric='cosine'))

print("Shape of the features RDM: ", features_rdm.shape)

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



    # =================================================================================================
    # Commonality Analysis (variance partitioning) for each vertex (neighbourhood): compute shared
    # variance explained by the EEG and features RDMs with respect to the fMRI RDM at that vertex
    # ==================================================================================================
    shared_variances = np.zeros((eeg_rdms.shape[0], n_vertices), dtype=np.float32)
    print("Commonality analysis loop has started...")

    for vertex in range(n_vertices):
        # Feature model: fMRI RDM ~ Features RDM
        feature_model = LinearRegression()
        feature_model.fit(features_rdm.reshape(-1, 1), fmri_rdms[vertex, :])
        r2_features = feature_model.score(features_rdm.reshape(-1, 1), fmri_rdms[vertex])
        #for t in range(eeg_rdms.shape[0]):
        for t in [51, 77, 102, 128, 153, 179, 205, 230, 256, 281, 307, 333, 358]:

            # EEG model: fMRI RDM ~ EEG RDM
            eeg_model = LinearRegression()
            eeg_model.fit(eeg_rdms[t].reshape(-1, 1), fmri_rdms[vertex])
            r2_eeg = eeg_model.score(eeg_rdms[t].reshape(-1, 1), fmri_rdms[vertex])

            # Combined model: fMRI RDM ~ EEG RDM + Features RDM
            combined_rdm = np.vstack((eeg_rdms[t], features_rdm)).T
            combined_model = LinearRegression()
            combined_model.fit(combined_rdm, fmri_rdms[vertex])
            r2_combined = combined_model.score(combined_rdm, fmri_rdms[vertex])

            # Shared variance between fMRI and Features in explaining EEG
            shared_var = r2_eeg + r2_features - r2_combined
            shared_variances[t,vertex] = shared_var

    print("Commonality analysis complete!")


# =============================================================================
# 5. Saving Results
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/commonality_analysis/wb/dnn_type-{args.dnn_type}/subject-{args.subject}/hemisphere-{args.hemisphere}'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = f'fmri_split-{args.fmri_split}.npy'
np.save(os.path.join(save_dir, file_name), shared_variances)

# End time
execution_time = time.time() - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")