"""
This scripts creates look-up tables (LUTs) for indicating, for each voxel,
what are the indices of the neighbouring voxels within a 10 mm radius.
The closest neighbours are determined using the voxels (x,y,z) 3D coordinates
and the Euclidean distance
These LUTs will be used in another script to compute fMRI RDMs for each
voxel based on these neighbourhoods

-----------------
Parameters:

--subject: fMRI participant ID
--radius: searchight radius in mm

"""

import numpy as np
import pandas as pd
import os
import argparse
import time
from scipy.spatial import cKDTree
from tqdm import tqdm

# Start time
start_time = time.time()

#==================================================================
# Input arguments
#==================================================================
parser = argparse.ArgumentParser(description="Generate Voxel Searchlight Look-Up Table")
parser.add_argument('--subject', type=str, default='01', choices=['01', '02', '03'], help="fMRI subject ID")
parser.add_argument('--radius', type=float, default=10.0, help="Searchlight radius in mm")
args = parser.parse_args()

print('Input arguments:')
print(f' -> Subject: {args.subject}')
print(f' -> Radius:  {args.radius} mm')

#==================================================================
# Load Voxel Metadata
#==================================================================
# Voxel metadata provided in the official THINGS data release
metadata_file = f'/scratch/jeffreykatab/Code/Encoding_Models/THINGS/fMRI/sub-{args.subject}_VoxelMetadata.csv'

if not os.path.exists(metadata_file):
    raise FileNotFoundError(f"Could not find metadata for subject {args.subject}. Check your path!")

df = pd.read_csv(metadata_file)

# Extract coordinates (x, y, z in mm)
# Shape: (n_voxels, 3)
coords = df[['voxel_x', 'voxel_y', 'voxel_z']].values

voxel_ids = df['voxel_id'].values # Keep track of voxel IDs

print(f' Loaded {len(coords)} voxels.')

#==================================================================
# Building KDTree and Querying Neighbors
#==================================================================
tree = cKDTree(coords)

print(f'Calculating neighborhoods within {args.radius} mm...')

# query_ball_point returns a list of indices for each point in coords
# that are within 'r' distance.
# We process the whole array at once for maximum speed.
neighbours_list = tree.query_ball_point(coords, r=args.radius)

# Converting to a dictionary for the Look-Up Table (LUT)
# Format: {voxel_id: [neighbor_voxel_indices]}
searchlight_lut = {
    int(voxel_ids[i]): np.array(neighbours_list[i], dtype=np.uint32) 
    for i in range(len(voxel_ids))
}

#==================================================================
# Saving Results
#==================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/searchlight_look_ups/sub-{args.subject}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

file_name = f'searchlight_lut_r-{args.radius}mm.npy'
save_path = os.path.join(save_dir, file_name)

# Saving as a dictionary allows for easy access later by voxel_id
np.save(save_path, searchlight_lut)

# End time
execution_time = time.time() - start_time
print(f" -> LUT saved to: {save_path}")
print(f" -> Execution Time: {execution_time:.2f} seconds.")