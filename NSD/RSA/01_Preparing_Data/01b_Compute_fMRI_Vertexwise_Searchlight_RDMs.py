"""
To compute the fMRI searchlight RDMs, we use pre-computed geodesic distances
which were obtained using the code at:
https://github.com/gifale95/BERG/blob/neural_signature_validation/paper_analyses/geodesic_vertex_distances/01_compute_geodesic_vertex_distances.py
"""


import numpy as np
from tqdm import tqdm
import argparse
import random
import numpy as np
import os
import random
import time 
import h5py
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
parser.add_argument('--subject', type=str, default='01')
parser.add_argument('--hemisphere', type=str, default='lh')
parser.add_argument('--radius', type=float, default=10.0)
args = parser.parse_args()

print('>>> Computing fMRI Vertexwise Searchlight RDMs <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#=============================================================================
  # Loading the fMRI data
#=============================================================================
_, fmri_test = load_fmri_hemi_data(args.subject, args.hemisphere)
print("Shape of the fMRI data: ", fmri_test.shape)

# =============================================================================
# Defining the vectorized correlation function
# =============================================================================
def corr_matrix(X, z_score=True):
    """
    Computes the correlation matrix of the input data.
    Parameters
    ----------
    X : (N, M) float array
        Input data matrix with N features and M samples.

    Returns
    -------
    corr : (M, M) float array
        Correlation matrix of the input data.
    """
    if z_score:
        Xc = X - X.mean(axis=0)
        Xc /= np.sqrt((Xc**2).sum(axis=0))
        return (Xc.T @ Xc).astype(np.float32)
    else:
        return (X.T @ X).astype(np.float32)


#=========================================================================================================
# Finding the neighborhood for each vertex (using pre-computed geodesic distances) and computing the RDMs
#=========================================================================================================
def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)  # k=1 excludes diagonal

save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/fmri_searchlight_rdms/radius-{args.radius}'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

# Access the precomputed geodesic distances # /scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/geodesic_vertex_distances
data_dir = os.path.join('/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/geodesic_vertex_distances', 'geodesic_vertex_distances_'+args.hemisphere+'.h5')
geodesic_distances = h5py.File(data_dir, 'r')['geodesic_distances']

n_stimuli = fmri_test.shape[0] # Number of stimuli (515)
n_pairs = 132355 # Number of unique pairs in the upper triangle of the RDM (with 515 stimuli, this is 132355 pairs)
n_vertices = fmri_test.shape[1] # Number of vertices

# Setup the HDF5 file
h5_save_file = os.path.join(save_dir, f'fmri_sub-{args.subject}_hemi-{args.hemisphere}_rdms.h5')

# We use 'w' to overwrite/create.
with h5py.File(h5_save_file, 'w') as f:
    # Create a chunked dataset for better performance
    # Chunking by vertex allows us to read/write specific vertices efficiently
    dset = f.create_dataset(
        'rdms', 
        shape=(n_vertices, n_pairs), 
        dtype='float32',
        chunks=(100, n_pairs),
        compression="gzip",
        compression_opts=4
    )

    print(f"Starting RDM computation for {n_vertices} vertices...")
    for v in tqdm(range(n_vertices)):
        # Finding neighbors
        #neighborhood = np.argsort(geodesic_distances[v])[:args.radius]
        # Select all vertices whose distance is within the radius
        mask = geodesic_distances[v] <= args.radius
        neighborhood = np.where(mask)[0]

        # Creating and flattening fMRI RDM
        current_rdm = 1 - corr_matrix(fmri_test[:, neighborhood].T)
        flat_rdm = flatten_rdm(current_rdm)
        
        # 3. Writing directly into the HDF5 dataset at the vertex index
        dset[v, :] = flat_rdm
    #np.save(os.path.join(save_dir, f'fmri_sub-{args.subject}_hemi-{args.hemisphere}_rdms.npy'), dset[:]) # Also save as numpy array for easy loading later

print(f" RDM computation complete! All RDMs saved to {h5_save_file}")


# End time
end_time = time.time()
execution_time = end_time - start_time

print(f"Execution complete! Time: {execution_time:.2f} seconds.")

