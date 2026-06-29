import numpy as np
import os
import argparse
import time
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import h5py
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


print('>>> ROI Commonality Analysis <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)  # k=1 excludes diagonal

# =============================================================================
# 1. Loading the fMRI data
# =============================================================================
_, fmri_data = load_fmri_roi_data2(args.subject, roi=args.roi, nc_threshold=0.2)
print('Shape of the fMRI data (test):', fmri_data.shape)

fmri_rdm = flatten_rdm(pairwise_distances(fmri_data, metric='correlation'))
del fmri_data
print("Shape of the fMRI RDM: ", fmri_rdm.shape)


# =============================================================================================================
# 2. Loading the EEG responses (test set) and computing the EEG RDMs
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



#====================================================================
# 3. Loading the layer features data and computing the features RDM
#====================================================================
features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/alexnet'
features_test = np.load(os.path.join(features_dir, f"sub-{args.subject:02d}_layerwise_fmaps.npy"), allow_pickle=True).item()[args.layer]['test']
print("Shape of the features data (test):", features_test.shape)
features_rdm = flatten_rdm(pairwise_distances(features_test, metric='cosine'))
print("Shape of the features RDM: ", features_rdm.shape)


#=======================================================================
# Commonality analysis: Decomposing the variance in the fMRI RDM
# explained by the EEG RDM and the features RDM
#=======================================================================
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
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/commonality_analysis/layerwise_alexnet/layer-{args.layer}/subject-{args.subject}'
os.makedirs(save_dir, exist_ok=True)

file_name = f'{args.roi}.npy'
np.save(os.path.join(save_dir, file_name), shared_variances)



# End time
execution_time = time.time() - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")