import numpy as np
import os
import argparse
import time
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import h5py
from sklearn.linear_model import LinearRegression

# Start time
start_time = time.time()

seed = 8
np.random.seed(seed)

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=str, default='01')
parser.add_argument('--hemisphere', type=str, default='left')
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--dnn_type', type=str, default='vdnn', choices=['vdnn', 'llm'],
                    help='Type of DNN features to use for the joint encoding fusion: "vdnn" for vision DNN features, "llm" for language model features.')
parser.add_argument('--eeg_frequency', type=int, default=100)
parser.add_argument('--eeg_rdm_policy', type=str, default='appended_channels',
                    choices=['appended_channels', 'subject_averaged'],
                    help="Policy for computing EEG RDMs: 'appended_channels' concatenates all subjects along channels, 'subject_averaged' averages RDMs across subjects")
args = parser.parse_args()


print('>>> Searchlight Whole-Brain Commonality Analysis <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)  # k=1 excludes diagonal

# =============================================================================================================
# 1. Loading the EEG data and Computing the RDM for the Current Time Point
# =============================================================================================================
eeg_data_path = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/prepared_eeg/{args.eeg_frequency}_Hz'
if args.eeg_rdm_policy == 'appended_channels':
    eeg_data = np.load(os.path.join(eeg_data_path, 'concat_eeg_test_762.npy')) # Shape: (102, 762, n_time_points)
    eeg_rdms = np.array([flatten_rdm(pairwise_distances(eeg_data[:,:,t], metric='correlation')) for t in range(eeg_data.shape[2])], dtype=np.float32) # Shape: (n_time_points, 5151)

elif args.eeg_rdm_policy == 'subject_averaged':
    eeg_rdm_list = []
    for sub in ['01', '02', '03', '04', '05', '06']:
        sub_data = np.load(os.path.join(eeg_data_path, f'sub-{sub}_test_z.npy')) # Shape: (102, 127, n_time_points)
        eeg_rdm_list.append(flatten_rdm(pairwise_distances(sub_data, metric='correlation')))
    eeg_rdms = np.mean(eeg_rdm_list, axis=0, dtype=np.float32) # Shape: (n_time_points, 5151)

#=======================================================================
# 2. Loading the features data
#=======================================================================
if args.dnn_type == 'vdnn':
    features_dir = '/scratch/jeffreykatab/Code/Encoding_Models/stimulus_features/pca_model_features/modality-visual'
    features = np.load(os.path.join(features_dir, 'pca_stimulus_features_test_model-s3d.npy'), allow_pickle=True)
elif args.dnn_type == 'llm':
    features_dir = '/scratch/jeffreykatabo/Code/Encoding_Models/semantic_models/embeddings/gte-Qwen1.5-7B-instruct'
    features = np.load(os.path.join(features_dir, 'gte-Qwen1.5-7B-instruct_1000PCs_test.npy'), allow_pickle=True)

print("Shape of the features data (test):", features.shape)
features_rdm = flatten_rdm(pairwise_distances(features, metric='cosine'))
print("Shape of the features RDM: ", features_rdm.shape)

# =============================================================================
# 3. Loading the Precomputed fMRI RDMs
# =============================================================================
fmri_rdms_path = os.path.join(
    '/scratch/jeffreykatab/Code/RSA', 'fmri_searchlight_rdms',
    f'n_neighbours-{args.n_neighbours}',
    f'fmri_sub-{args.fmri_subject}_hemi-{args.hemisphere}_rdms.h5'
)

print(f"Opening fMRI HDF5 RDMs: {fmri_rdms_path}")

# Using a context manager to handle the H5 file
with h5py.File(fmri_rdms_path, 'r') as f:
    # Access the dataset pointer
    # We only load the portion of the dataset corresponding to the current split for computational efficiency
    fmri_rdms = f['rdms'][7802*(args.fmri_split - 1):7802*args.fmri_split, :] # Shape: (7802, 5151)
    n_vertices = fmri_rdms.shape[0]


    # =================================================================================================
    # 4. Commonality Analysis (variance partitioning) for each vertex (neighbourhood): compute shared
    # variance explained by the EEG and features RDMs with respect to the fMRI RDM at that vertex
    # ==================================================================================================
    shared_variances = np.zeros((eeg_rdms.shape[0], n_vertices), dtype=np.float32)
    print("Commonality analysis loop has started...")

    for vertex in tqdm(range(n_vertices)):
        # Feature model: fMRI RDM ~ Features RDM
        feature_model = LinearRegression()
        feature_model.fit(features_rdm.reshape(-1, 1), fmri_rdms[vertex])
        r2_features = feature_model.score(features_rdm.reshape(-1, 1), fmri_rdms[vertex])
        for t in range(eeg_rdms.shape[0]):

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
save_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/results/variances/commonality_analysis_wb/{args.eeg_frequency}_Hz/fmri_sub-{args.fmri_subject}/{args.hemisphere}_hemisphere'
os.makedirs(save_dir, exist_ok=True)

file_name = f'fmri_split-{args.fmri_split}.npy'
np.save(os.path.join(save_dir, file_name), shared_variances)

# End time
execution_time = time.time() - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")