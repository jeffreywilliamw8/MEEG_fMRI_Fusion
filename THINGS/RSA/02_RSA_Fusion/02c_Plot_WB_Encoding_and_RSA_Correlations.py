"""
This script plots the Encoding and RSA fusion vertex-wise correlation time courses
on brain surfaces for a given fmRI subject, using pycortex. At each time point, the Encoding and RSA surface plots
are vertically stacked and an image file is saved. The set of all image files are then combined using
an external tool to create whole-brain movies

"""

import argparse
import os
import numpy as np
import cortex
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import time

# Start time
start_time = time.time()

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=str, default='01')
parser.add_argument('--pycortex_filestore', type=str, default='/scratch/jeffreykatab/Code/Encoding_Models/THINGS/fMRI/pycortex_filestore')
parser.add_argument('--transform', type=str, default='align_auto')
args = parser.parse_args()

# Setup Pycortex
cortex.database.default_filestore = args.pycortex_filestore
subject_map = {'01': 'S1', '02': 'S2', '03': 'S3'}
pycortex_subject = subject_map[args.fmri_subject]

PLOTS_DIR = f'/home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/plots/meg2fmri_fusion/sub-{args.fmri_subject}'
os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================================================================
# Load Metadata & Data
# =============================================================================
# 1. Load Voxel Mapping
fmri_metadata_dir = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/fMRI/prepared'
fmri_metadata = np.load(os.path.join(fmri_metadata_dir, f'fmri_{args.fmri_subject}_metadata.npy'), allow_pickle=True).item()
coords = fmri_metadata['fmri']['voxel_coords']
vol_shape = (coords[:, 2].max() + 1, coords[:, 1].max() + 1, coords[:, 0].max() + 1)

# 2. Load Time info
meg_metadata_dir = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/MEG/prepared'
meg_metadata = np.load(os.path.join(meg_metadata_dir, 'meg_P1_metadata.npy'), allow_pickle=True).item()
times = 1000 * meg_metadata['meg']['times'][:141]

# 3. Load Both Correlation Datasets
# /scratch/jeffreykatab/Code/Encoding_Models/THINGS/correlations/meg2fmri_fusion/sub-01
em_corrs_path = '/home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/correlations/meg2fmri_fusion'
rsa_corrs_path = '/home/jeffreykatab/Projects/fusion/THINGS/RSA/correlations/meg2fmri_fusion/searchlight'
em_ctc = np.load(os.path.join(em_corrs_path, f'sub-{args.fmri_subject}_correlation_time_courses.npy')) # Encoding models correlation time course
rsa_ctc = np.load(os.path.join(rsa_corrs_path, f'sub-{args.fmri_subject}_correlation_time_courses.npy')) # RSA correlation time course

print("Shape of the correlation time courses (Encoding Models):", em_ctc.shape)
print("Shape of the correlation time courses (RSA):", rsa_ctc.shape)
#em_ctc = np.mean(em_ctc, axis=2) # Average across sessions


#noise_ceilings = fmri_metadata['encoding_model']['noise_ceiling_testset']
#mask = noise_ceilings < 20.0


# 2. Apply the mask across the voxel dimension (columns)
#em_ctc[:, mask] = np.nan
#rsa_ctc[:, mask] = np.nan

M1 = np.max(em_ctc)
M2 = np.max(rsa_ctc)

# =============================================================================
# Helper: Volume Reconstruction
# =============================================================================
def get_volume(correlations, shape, voxel_coords):
    vol = np.zeros(shape)
    vol[voxel_coords[:, 2], voxel_coords[:, 1], voxel_coords[:, 0]] = correlations
    return vol

# =============================================================================
# Main Plotting Loop
# =============================================================================
print(f"Generating stacked surfaces for {len(times)} timepoints...")

for t in tqdm(range(len(times))):
    # Create 3D Volumes for both datasets
    vol1_3d = get_volume(em_ctc[t, :], vol_shape, coords)
    vol2_3d = get_volume(rsa_ctc[t, :], vol_shape, coords)

    # Create Pycortex Volume objects
    volume1 = cortex.Volume(vol1_3d, pycortex_subject, args.transform, vmin=0.0, vmax=M1, cmap='viridis')
    volume2 = cortex.Volume(vol2_3d, pycortex_subject, args.transform, vmin=0.0, vmax=M2, cmap='viridis')

    # Temporary filenames for the parts
    tmp_png1 = f"tmp1_sub{args.fmri_subject}.png"
    tmp_png2 = f"tmp2_sub{args.fmri_subject}.png"

    # Generate individual flatmaps
    cortex.quickflat.make_png(tmp_png1, volume1, with_curvature=True, with_rois=True, with_labels=True)
    cortex.quickflat.make_png(tmp_png2, volume2, with_curvature=True, with_rois=True, with_labels=True)

    # Load images with PIL for stacking
    img1 = Image.open(tmp_png1)
    img2 = Image.open(tmp_png2)

    # Combine with Matplotlib
    fig, axes = plt.subplots(2, 1, figsize=(16, 18), facecolor='white')
    
    axes[0].imshow(img1)
    axes[0].axis("off")
    axes[0].set_title(f"Encoding Fusion (Pearson\'s r) | Time: {times[t]:.2f} ms", fontsize=24, fontweight='bold', pad=20)

    axes[1].imshow(img2)
    axes[1].axis("off")
    axes[1].set_title(f"RSA Fusion (Spearman\'s R) | Time: {times[t]:.2f} ms", fontsize=24, fontweight='bold', pad=20)

    # Add a global footer or timestamp if desired
    fig.text(0.5, 0.02, f"Subject {args.fmri_subject} | MEG-fMRI Fusion ", ha='center', fontsize=24, fontweight='bold', color='gray')

    plt.tight_layout()
    
    # Save combined figure
    save_path = os.path.join(PLOTS_DIR, f'time_point_{t}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Cleanup temp files
    os.remove(tmp_png1)
    os.remove(tmp_png2)

print(f"\n Done! Plots saved in: {PLOTS_DIR}")
print(f"Total time: {time.time() - start_time:.2f} seconds.")