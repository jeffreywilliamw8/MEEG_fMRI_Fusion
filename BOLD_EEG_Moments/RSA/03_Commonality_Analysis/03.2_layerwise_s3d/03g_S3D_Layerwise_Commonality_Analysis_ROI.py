import numpy as np
import os
import argparse
import time
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import h5py
from sklearn.linear_model import LinearRegression
from utils import load_fmri_data

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
parser.add_argument('--roi', type=str, default='V1v')
parser.add_argument('--layer', type=str, default='layer2', choices=['layer2', 'layer5', 'layer7', 'layer9', 'layer11', 'layer13'],
                    help='Layer of the S3D model from which the features are extracted for the commonality analysis.')
parser.add_argument('--eeg_frequency', type=int, default=100)
parser.add_argument('--eeg_rdm_policy', type=str, default='appended_channels',
                    choices=['appended_channels', 'subject_averaged'],
                    help="Policy for computing EEG RDMs: 'appended_channels' concatenates all subjects along channels, 'subject_averaged' averages RDMs across subjects")
args = parser.parse_args()


print('>>> ROI Commonality Analysis <<<')
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

# =============================================================================
# 2. Loading the fMRI data
# =============================================================================
_, fmri_data = load_fmri_data(args.fmri_subject, args.hemisphere, roi=args.roi, threshold=20.0)
fmri_rdm = flatten_rdm(pairwise_distances(fmri_data, metric='correlation')) # Shape: (5151,)
del fmri_data
print("Shape of the fMRI RDM: ", fmri_rdm.shape)


#====================================================================
# 3. Loading the layer features data
#====================================================================
features_dir = '/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/features/layerwise_s3d_features'
features = np.load(os.path.join(features_dir, f'{args.layer}_test_pca1000.npy'), allow_pickle=True)
features_rdm = flatten_rdm(pairwise_distances(features, metric='cosine'))
print("Shape of the features data (test):", features.shape)
print("Shape of the features RDM: ", features_rdm.shape)
del features


# =================================================================================================
# 4. Commonality Analysis (variance partitioning) for each vertex (neighbourhood): compute shared
# variance explained by the EEG and features RDMs with respect to the fMRI RDM at that vertex
# ==================================================================================================
shared_variances = [] # Shape: (n_time_points,)
print("Commonality analysis loop has started...")


# Feature model: fMRI RDM ~ Features RDM
feature_model = LinearRegression()
feature_model.fit(features_rdm.reshape(-1, 1), fmri_rdm)
r2_features = feature_model.score(features_rdm.reshape(-1, 1), fmri_rdm)
for t in range(eeg_rdms.shape[0]):

    # EEG model: fMRI RDM ~ EEG RDM
    eeg_model = LinearRegression()
    eeg_model.fit(eeg_rdms[t].reshape(-1, 1), fmri_rdm)
    r2_eeg = eeg_model.score(eeg_rdms[t].reshape(-1, 1), fmri_rdm)

    # Combined model: fMRI RDM ~ EEG RDM + Features RDM
    combined_rdm = np.vstack((eeg_rdms[t], features_rdm)).T
    combined_model = LinearRegression()
    combined_model.fit(combined_rdm, fmri_rdm)
    r2_combined = combined_model.score(combined_rdm, fmri_rdm)

    # Shared variance between fMRI and Features in explaining EEG
    shared_var = r2_eeg + r2_features - r2_combined
    shared_variances.append(shared_var)

print("Commonality analysis complete!")
shared_variances = np.array(shared_variances, dtype=np.float32) # Shape: (n_time_points,)

# =============================================================================
# 5. Saving Results
# =============================================================================
save_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/results/variances/commonality_analysis_roi/layerwise_s3d/{args.eeg_frequency}_Hz/fmri_sub-{args.fmri_subject}'
os.makedirs(save_dir, exist_ok=True)

file_name = f'{args.roi}_{args.hemisphere}.npy'
np.save(os.path.join(save_dir, file_name), shared_variances)

# End time
execution_time = time.time() - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")