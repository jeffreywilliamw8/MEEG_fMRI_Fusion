import numpy as np
import os
import argparse
import time
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import h5py
from sklearn.linear_model import LinearRegression
from utils import load_fmri_roi_data

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
parser.add_argument('--roi', type=str, default='V1v')
parser.add_argument('--dnn_type', type=str, default='vdnn', choices=['vdnn', 'llm'], help='Type of DNN features to use for the joint encoding fusion: "vdnn" for vision DNN features, "llm" for language model features.')
args = parser.parse_args()

print('>>> ROI Commonality Analysis <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

def flatten_rdm(rdm):
        return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)  # k=1 excludes diagonal
# =============================================================================
# Load the fMRI responses (test set) and computing the fMRI RDMs
# =============================================================================
_, fmri_test = load_fmri_roi_data(args.subject, args.hemisphere, roi=args.roi, nc_threshold=0.2)
if fmri_test.shape[1]>0:

    print('Shape of the fMRI data (test):', fmri_test.shape)
    fmri_rdm = flatten_rdm(pairwise_distances(fmri_test, metric='correlation'))
    print("Shape of the fMRI RDM: ", fmri_rdm.shape)
    del fmri_test

    

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
        features_dir = '/scratch/giffordale95/projects/brain-encoding-response-generator/eeg_fmri_fusion/invivo_nsd_eeg_fmri_control/dnn_llm_modeling/stimulus_features'
        features_test = np.load(os.path.join(features_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_test']

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
    save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/commonality_analysis/vision_language_models/roi/dnn_type-{args.dnn_type}/subject-{args.subject}'
    os.makedirs(save_dir, exist_ok=True)

    file_name = f'{args.roi}_{args.hemisphere}.npy'
    np.save(os.path.join(save_dir, file_name), shared_variances)

else:
     print("No vertices above noise ceiling threshold found in this ROI. Terminating...")

# End time
execution_time = time.time() - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")