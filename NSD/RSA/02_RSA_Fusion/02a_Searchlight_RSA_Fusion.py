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
parser.add_argument('--subject', type=int, default=8)
parser.add_argument('--hemisphere', type=str, default='lh')
parser.add_argument('--time_point', type=int, default=0)
parser.add_argument('--n_neighbours', type=int, default=100) 
parser.add_argument('--distance_metric', type=str, default='correlation', choices=['correlation', 'decoding_accuracy'])

args = parser.parse_args()

print('>>> Searchlight RSA Fusion <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)

# =============================================================================================================
# 1. Loading the EEG data and Computing the RDM
# =============================================================================================================
if args.distance_metric == 'correlation':
    data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
    eeg_dict = np.load(os.path.join(data_dir, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True).item()
    eeg_data = eeg_dict['eeg_test'][:,:,:, args.time_point] 
    eeg_data = np.mean(eeg_data, axis=1) 
    eeg_rdm = flatten_rdm(pairwise_distances(eeg_data, metric='correlation'))
elif args.distance_metric == 'decoding_accuracy':
     data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/decoding_rdms'
     eeg_rdm = np.load(os.path.join(data_dir, f"decoding_rdm_eeg_sub-{args.subject}.npy"))[args.time_point]

# =============================================================================
# 2. Loading the Precomputed fMRI RDMs from HDF5
# =============================================================================
fmri_h5_file = os.path.join(
    f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/fmri_searchlight_rdms/n_neighbours-{args.n_neighbours}',
    f'fmri_sub-{args.subject}_hemi-{args.hemisphere}_rdms.h5'
)
# Pre-allocate results array in RAM
searchlight_corrs = np.zeros(163842, dtype=np.float32)
# Open the file in read mode
with h5py.File(fmri_h5_file, 'r') as f:
    # Point to the dataset (this doesn't load it into memory yet)
    dset = f['rdms']
    n_vertices, n_pairs = dset.shape
    fmri_rdms = dset[:]
    
    

    # =============================================================================
    # 3. Searchlight Correlation (Spearman)
    # =============================================================================
    print(f"Computing Spearman correlations for {n_vertices} vertices...")
    for v in tqdm(range(n_vertices)):
        
        
        # Spearman correlation (note: spearmanr handles rank-ordering internally)
        # We take .correlation to ignore the p-value
        searchlight_corrs[v] = spearmanr(fmri_rdms[v,:], eeg_rdm).correlation

# =============================================================================
# 4. Saving Results
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/searchlight_fusion/n_neighbours-{args.n_neighbours}/metric_{args.distance_metric}/subject-{args.subject}/{args.hemisphere}_hemisphere'
os.makedirs(save_dir, exist_ok=True)
     
file_name = f'time_point_{args.time_point:04d}.npy'
np.save(os.path.join(save_dir, file_name), searchlight_corrs)

# End time
execution_time = time.time() - start_time
print(f"Searchlight complete for time point {args.time_point}")
print(f"Results saved to: {os.path.join(save_dir, file_name)}")
print(f"Total Execution time: {execution_time:.2f} seconds.")