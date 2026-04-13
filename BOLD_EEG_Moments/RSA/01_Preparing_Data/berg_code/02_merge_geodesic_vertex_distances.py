"""Merge the geodesic vertex distances across vertex splits.

Parameters
----------
hemisphere : str
    String containing the hemisphere used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
total_vertex_splits : int
    Total number of splits to divide the vertices into smaller chunks for
    parallel processing.
berg_dir : str
    Directory of the BERG.

"""

import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
import nibabel as nib
from nilearn import datasets
import pygeodesic.geodesic as geodesic

parser = argparse.ArgumentParser()
parser.add_argument('--hemisphere', default='lh', type=str)
parser.add_argument('--total_vertex_splits', default=81, type=int)
args, unknown = parser.parse_known_args()

print('>>> Merge ertex geodesic distance <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load and merge the geodesic distances
# =============================================================================
data_dir = '/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/geodesic_vertex_distances'

for split in tqdm(range(args.total_vertex_splits)):
    file_name = 'geodesic_vertex_distances_' + args.hemisphere + '_split-' + \
        format(split, '03') + '.h5'
    if split == 0:
        geodesic_distances = h5py.File(os.path.join(data_dir, file_name),
            'r')['geodesic_distances'][:]
    else:
        geodesic_distances = np.append(geodesic_distances, h5py.File(
            os.path.join(data_dir, file_name), 'r')['geodesic_distances'][:],
            0)


# =============================================================================
# Save the merged geodesic distances
# =============================================================================
file_name = 'geodesic_vertex_distances_' + args.hemisphere + '.h5'

with h5py.File(os.path.join(data_dir, file_name), 'w') as f:
    f.create_dataset('geodesic_distances', data=geodesic_distances,
        dtype=np.float32)