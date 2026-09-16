"""
Computes a searchlight RDM (1 - Pearson r) per voxel from the THINGS test-set
fMRI responses, using the precomputed searchlight neighbor lookup tables. Saved
as a single .npy file, shape (n_voxels, 4950).
"""

import argparse
import os
import numpy as np
from berg import BERG
from tqdm import tqdm
import random
from utils import load_fmri_wb_data
import time
from sklearn.metrics import pairwise_distances



parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--radius', default=10.0, type=float)
parser.add_argument('--berg_dir', default='/scratch/jeffreykatab/Code/Encoding_Models/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 8
random.seed(seed)
np.random.seed(seed)

# Start time
start_time = time.time()


ROOT_DIR = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS'

# =============================================================================
# Load the in vivo THINGS fMRI1 train responses (We use only the test set)
# =============================================================================
# Load the fMRI responses
_, fmri_responses = load_fmri_wb_data(args.subject) # loading test set only

# --- Paths ---
# f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/searchlight_look_ups/sub-{args.subject}'
LUT_PATH = f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/searchlight_look_ups/sub-{args.subject:02d}/searchlight_lut_r-{args.radius}mm.npy'
SAVE_DIR = f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/searchlight_rdms/sub-{args.subject:02d}'
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, f'searchlight_rdms_r-{args.radius}.npy')


#==================================================================
# 1. Load Data & Setup
#==================================================================
lut = np.load(LUT_PATH, allow_pickle=True).item()
# Assuming fmri_responses is loaded: (n_stimuli, n_voxels)
n_stimuli, n_voxels = fmri_responses.shape

# Calculate size of the upper triangle (excluding diagonal)
n_triu = n_stimuli * (n_stimuli - 1) // 2
triu_indices = np.triu_indices(n_stimuli, k=1)

print(f' Subject {args.subject:02d} | Voxels: {n_voxels} | Stimuli: {n_stimuli}')

#==================================================================
# 2. Preallocate the output array (replaces the HDF5 dataset)
#==================================================================
# Shape (n_voxels, n_triu), filled in per-voxel by the loop below, then
# written to disk once at the end via np.save.
rdms = np.zeros((n_voxels, n_triu), dtype=np.float32)

def flatten_rdm(rdm):
    return np.float32(rdm[np.triu_indices_from(rdm, k=1)])  # k=1 excludes diagonal
#============================================================================================
# 3. RDM compution loop, optimized via vectorized correlations and matrix multimplications
#============================================================================================
def get_rdm_triu(data_matrix):
    """Computes RDM and returns only the upper triangle flattened."""
    # Center and Normalize
    centered = data_matrix - np.mean(data_matrix, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = centered / norms

    # Pearson Correlation Matrix
    corr_matrix = np.matmul(normalized, normalized.T)
    rdm = 1 - corr_matrix

    # Extract upper triangle to save space
    return rdm[triu_indices]

print(f' Processing voxels...')
for v_id, neighbors in tqdm(lut.items()):

    # Extract data, compute RDM, and write into the in-memory array
    searchlight_data = fmri_responses[:, neighbors]
    rdm_vector = flatten_rdm(pairwise_distances(searchlight_data, metric='correlation'))

    rdms[v_id, :] = rdm_vector

# Write the full array to disk once
np.save(SAVE_PATH, rdms)

execution_time = time.time() - start_time
print(f"\n Success! .npy file saved at: {SAVE_PATH}")
print(f" -> Time: {execution_time:.2f} seconds.")