"""
This script computes vertexwise searchlight RDMs for fMRI data using geodesic neighborhoods.
It processes the fMRI data in parallel, computing RDMs for each vertex based on its geodesic neighborhood, and saves the results in an HDF5 file.
Parameters:
- subject: The subject number (integer) for which to compute the fMRI RDMs.
- hemisphere: The hemisphere of the brain to analyze ('lh' for left hemisphere, 'rh' for right hemisphere).
- n_neighbours: The number of nearest neighbors to include in the searchlight for each vertex.
- chunk_size: The number of vertices to process in each parallel job.

"""

import numpy as np
from tqdm import tqdm
import argparse
import random
import os
import time 
import h5py

# --- Core Environment Safeguards against worker over-subscription ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from joblib import Parallel, delayed
from utils import load_fmri_hemi_data

# Start time
start_time = time.time()

seed = 8
np.random.seed(seed)
random.seed(seed)


#=============================================================================
# Input arguments
#=============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh')
parser.add_argument('--n_neighbours', type=int, default=100)
parser.add_argument('--chunk_size', type=int, default=1000, help='Number of vertices per joblib task')
args = parser.parse_args()

print('>>> Parallel fMRI Vertexwise Searchlight RDMs <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

#=============================================================================
# Loading the fMRI data
#=============================================================================
_, fmri_test = load_fmri_hemi_data(args.subject, args.hemisphere)
print("Shape of the fMRI data: ", fmri_test.shape)

# =============================================================================
# Helper Math Functions
# =============================================================================
def corr_matrix(X, z_score=True):
    if z_score:
        Xc = X - X.mean(axis=0)
        Xc /= np.sqrt((Xc**2).sum(axis=0))
        return (Xc.T @ Xc).astype(np.float32)
    else:
        return (X.T @ X).astype(np.float32)

def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)

#=============================================================================
# Parallel Processing Chunk Worker
#=============================================================================
def process_vertex_chunk(start_v, end_v, fmri_test, geo_dist_path, n_neighbours, n_pairs):
    """
    Worker function that handles a continuous block of vertices. 
    Opens a read-only handle to the geodesic HDF5 matrix within its local process.
    """
    chunk_len = end_v - start_v
    chunk_results = np.zeros((chunk_len, n_pairs), dtype=np.float32)
    
    # Open local file pointer inside worker for thread safety
    with h5py.File(geo_dist_path, 'r') as f_geo:
        geo_dataset = f_geo['geodesic_distances']
        
        for idx, v in enumerate(range(start_v, end_v)):
            # Extract geodesic distance profile for vertex v
            neighborhood = np.argsort(geo_dataset[v])[:n_neighbours]

            # Compute custom RDM profile
            current_rdm = 1 - corr_matrix(fmri_test[:, neighborhood].T)
            chunk_results[idx, :] = flatten_rdm(current_rdm)
            
    return start_v, end_v, chunk_results

#=============================================================================
# Main Pipeline Setup
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/fmri_searchlight_rdms/n_neighbours-{args.n_neighbours}'
os.makedirs(save_dir, exist_ok=True)

geo_dist_path = os.path.join(
    '/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/geodesic_vertex_distances', 
    f'geodesic_vertex_distances_{args.hemisphere}.h5'
)

n_stimuli = fmri_test.shape[0] 
n_pairs = 132355 
n_vertices = fmri_test.shape[1] 

# Generate slice indices for distributing blocks across the CPU cores
chunks = []
for start_v in range(0, n_vertices, args.chunk_size):
    end_v = min(start_v + args.chunk_size, n_vertices)
    chunks.append((start_v, end_v))

h5_save_file = os.path.join(save_dir, f'fmri_sub-{args.subject}_hemi-{args.hemisphere}_rdms.h5')

# Execute computation pool and write outputs sequentially to disk
with h5py.File(h5_save_file, 'w') as f_out:
    dset = f_out.create_dataset(
        'rdms', 
        shape=(n_vertices, n_pairs), 
        dtype='float32',
        chunks=(100, n_pairs),
        compression="gzip",
        compression_opts=4
    )

    print(f"\n>>> Dispatching {len(chunks)} vertex chunks to Joblib pool <<<")
    
    # Use backend="loky" for robust multi-process parallel execution
    results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(process_vertex_chunk)(start, end, fmri_test, geo_dist_path, args.n_neighbours, n_pairs)
        for start, end in chunks
    )
    
    print("\n>>> Collecting worker outputs and committing to HDF5 file <<<")
    for start_v, end_v, chunk_data in tqdm(results, desc="Writing to file"):
        dset[start_v:end_v, :] = chunk_data
        
    f_out.flush()

print(f"File saved to {h5_save_file}")

# End time
end_time = time.time()
print(f"Execution complete! Total time: {end_time - start_time:.2f} seconds.")