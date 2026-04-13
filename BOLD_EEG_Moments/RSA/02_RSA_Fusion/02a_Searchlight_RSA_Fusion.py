import numpy as np
import os
import argparse
import time
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import h5py

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
parser.add_argument('--time_point', type=int, default=0)
parser.add_argument('--eeg_frequency', type=int, default=100)
parser.add_argument('--n_neighbours', type=int, default=100) 
parser.add_argument('--eeg_rdm_policy', type=str, default='appended_channels',
                    choices=['appended_channels', 'subject_averaged'],
                    help="Policy for computing EEG RDMs: 'appended_channels' concatenates all subjects along channels, 'subject_averaged' averages RDMs across subjects")
args = parser.parse_args()


print('>>> Searchlight RSA Fusion <<<')
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
    eeg_data = np.load(os.path.join(eeg_data_path, 'concat_eeg_test_762.npy'))[:,:,args.time_point] # Shape: (102, 762)
    eeg_rdm = flatten_rdm(pairwise_distances(eeg_data, metric='correlation')) # Shape: (5151,)

elif args.eeg_rdm_policy == 'subject_averaged':
    eeg_rdm_list = []
    for sub in ['01', '02', '03', '04', '05', '06']:
        sub_data = np.load(os.path.join(eeg_data_path, f'sub-{sub}_test_z.npy'))[:,:,args.time_point] # Shape: (102, 127)
        eeg_rdm_list.append(flatten_rdm(pairwise_distances(sub_data, metric='correlation'))) # Shape: (5151,)
    eeg_rdm = np.mean(eeg_rdm_list, axis=0, dtype=np.float32) # Shape: (5151,)

# =============================================================================
# 2. Loading the Precomputed fMRI RDMs
# =============================================================================
fmri_rdms_path = os.path.join(
    '/scratch', 'jeffreykatab', 'Code', 'RSA', 'fmri_searchlight_rdms',
    f'n_neighbours-{args.n_neighbours}',
    f'fmri_sub-{args.fmri_subject}_hemi-{args.hemisphere}_rdms.h5'
)

print(f"Opening fMRI HDF5 RDMs: {fmri_rdms_path}")

# Using a context manager to handle the H5 file
with h5py.File(fmri_rdms_path, 'r') as f:
    # Access the dataset pointer
    fmri_dset = f['rdms']
    n_vertices = fmri_dset.shape
    
    # Pre-allocate results array
    searchlight_corrs = np.zeros(n_vertices, dtype=np.float32)

    # =============================================================================
    # 3. Searchlight Correlation (Spearman)
    # =============================================================================
    print(f"Computing Spearman correlations across {n_vertices} vertices...")
    for v in tqdm(range(n_vertices)):
        # Slicing directly from the HDF5 dataset for vertex v
        # This keeps RAM usage low while iterating
        v_fmri_rdm = fmri_dset[v, :]
        
        # Correlate the 5151-length vector of the fMRI vertex with the 5151-length EEG vector
        searchlight_corrs[v] = spearmanr(v_fmri_rdm, eeg_rdm).correlation

# =============================================================================
# 4. Saving Results
# =============================================================================
save_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/results/correlations/searchlight_fusion/n_neighbours-{args.n_neighbours}/{args.eeg_frequency}_Hz/fmri_sub-{args.fmri_subject}/{args.hemisphere}_hemisphere'
os.makedirs(save_dir, exist_ok=True)

file_name = f'time_point_{args.time_point:04d}.npy'
np.save(os.path.join(save_dir, file_name), searchlight_corrs)

# End time
execution_time = time.time() - start_time
print(f"Searchlight complete for time point {args.time_point}")
print(f"Results saved to: {os.path.join(save_dir, file_name)}")
print(f"Total Execution time: {execution_time:.2f} seconds.")