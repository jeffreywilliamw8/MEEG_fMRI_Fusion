import numpy as np
import os
import argparse
import time
import h5py

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
parser.add_argument('--n_neighbours', type=int, default=100) 

args = parser.parse_args()


print('>>> Extracting RDMs from .h5 files and saving them to .npy files <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)  # k=1 excludes diagonal

# =============================================================================
# Loading the Precomputed fMRI RDMs
# =============================================================================
save_path = os.path.join(f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/fmri_searchlight_rdms/n_neighbours-{args.n_neighbours}')
if os.path.isdir(save_path) == False:
	os.makedirs(save_path)
fmri_rdms_file = os.path.join(
    f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/fmri_searchlight_rdms/n_neighbours-{args.n_neighbours}',
    f'fmri_sub-{args.subject}_hemi-{args.hemisphere}_rdms.h5'
)

print(f"Opening fMRI HDF5 RDMs: {fmri_rdms_file}")

# Using a context manager to handle the H5 file
with h5py.File(fmri_rdms_file, 'r') as f:
    # Access the dataset pointer
    fmri_rdms = f['rdms']

        
    file_name = f'fmri_sub-{args.subject}_hemi-{args.hemisphere}_rdms.npy'
    np.save(os.path.join(save_path, file_name), fmri_rdms)

# End time
execution_time = time.time() - start_time
print(f"Total Execution time: {execution_time:.2f} seconds.")