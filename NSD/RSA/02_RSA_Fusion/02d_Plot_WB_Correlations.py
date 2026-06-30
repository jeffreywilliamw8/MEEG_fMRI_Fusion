"""
This script plots the fusion results (correlation time courses)
for encoding and RSA on brain flatmaps, at each EEG time point.
The results are averaged across subjects
"""


import os
import numpy as np
import cortex
from PIL import Image
import matplotlib
from tqdm import tqdm
from utils import get_eeg_times
import matplotlib.pyplot as plt
import time


# Start time
start_time = time.time()



# =============================================================================
# Load the encoding models and RSA correlation results
# =============================================================================
# Encoding model results
em_corrs_left = np.zeros((359, 163842)) # Initialize an empty array for left hemisphere results
em_corrs_right = np.zeros((359, 163842)) # Initialize an empty array for right hemisphere results
subject_list = [1,4,5,6,7,8]   # List of subjects to process
# Loop through each subject and load the corresponding results
print("Loading Encoding model results...")
for subject in subject_list:
    corrs_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion/whole_brain/subject-{subject}'
    data_dir_l = os.path.join(corrs_dir, 'correlations_left.npy')
    data_dir_r = os.path.join(corrs_dir, 'correlations_right.npy')
    em_corrs_left += np.load(data_dir_l)
    em_corrs_right += np.load(data_dir_r)
em_corrs_left /= len(subject_list)  # Average the results across subjects
em_corrs_right /= len(subject_list)  # Average the results across subjects
print("Shape of the Encoding correlation time courses (left, right): ", em_corrs_left.shape, em_corrs_right.shape)
M1 = max(np.max(em_corrs_left), np.max(em_corrs_right))  #

# RSA results
rsa_corrs_left = np.zeros((359, 163842)).astype(np.float32) # Initialize an empty array for left hemisphere results
rsa_corrs_right = np.zeros((359, 163842)).astype(np.float32)  # Initialize an empty array for right hemisphere results
# Loop through each subject and load the corresponding results
print("Loading RSA results...")
for subject in subject_list:
    corrs_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/searchlight_fusion/n_neighbours-100/metric_correlation/aggregated_results/subject-{subject}'
    data_dir_l = os.path.join(corrs_dir, f'subject-{subject}_lh_hemisphere_timecourse.npy')
    data_dir_r = os.path.join(corrs_dir, f'subject-{subject}_rh_hemisphere_timecourse.npy')
    rsa_corrs_left += np.load(data_dir_l)
    rsa_corrs_right += np.load(data_dir_r)

rsa_corrs_left /= len(subject_list)  # Average the results across subjects
rsa_corrs_right /= len(subject_list)  # Average the results across subjects

print("Shape of the RSA correlation time courses (left, right): ", rsa_corrs_left.shape, rsa_corrs_right.shape)
M2 = max(np.max(rsa_corrs_left), np.max(rsa_corrs_right))  #

# =============================================================================
# Plot parameters for colorbar
# =============================================================================
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage_nsd_sub-01'

# =============================================================================
# Plotting the correlations
# =============================================================================
plots_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots/fusion_correlations/whole_brain/em_rsa_subject_averaged'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)  # Create the directory if it does not exist
times = get_eeg_times()
n_time_points = em_corrs_left.shape[0]    # number of time points


print("Plotting the correlations on brain flatmaps...") 
for t in tqdm(range(n_time_points)):
    
    # Combine hemispheres for both datasets
    em_corrs = np.append(em_corrs_left[t, :], em_corrs_right[t, :])
    rsa_corrs = np.append(rsa_corrs_left[t, :], rsa_corrs_right[t, :])

    # Create pycortex vertex data for both maps
    vertex_data_1 = cortex.Vertex(em_corrs, subject, cmap='viridis', vmin=0, vmax=M1)
    vertex_data_2 = cortex.Vertex(rsa_corrs, subject, cmap='viridis', vmin=0, vmax=M2)

    # --------------------
    # 1. Save each flatmap to disk using pycortex quickflat (PNG)
    # --------------------
    png1 = os.path.join(plots_dir, f"surface1_t{t}.png")
    png2 = os.path.join(plots_dir, f"surface2_t{t}.png")

    # Generate the two cortical flatmaps
    cortex.quickflat.make_png(
    png1,
    vertex_data_1,
    with_curvature=True,
    curvature_brightness=0.5,
    with_rois=True,
    roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
    with_labels=True)
    cortex.quickflat.make_png(
        png2,
        vertex_data_2,
        with_curvature=True,
        curvature_brightness=0.5,
        with_rois=True,
        roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
        with_labels=True)

    # --------------------
    # 2. Load PNGs with PIL
    # --------------------
    img1 = Image.open(png1)
    img2 = Image.open(png2)

    # --------------------
    # 3. Combine with Matplotlib and add titles
    # --------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    axes[0].imshow(img1)
    axes[0].axis("off")
    axes[0].set_title(
        f"Encoding (Pearson's r) | Time: {times[t]:.2f} ms",fontsize=26)

    axes[1].imshow(img2)
    axes[1].axis("off")
    axes[1].set_title(f"RSA (Spearman's R) | Time: {times[t]:.2f} ms",fontsize=26)

    plt.tight_layout()

    out_file = os.path.join(plots_dir, f"time_point_{t}.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight", transparent=False)
    plt.close()
    # Cleanup temp files
    os.remove(png1)
    os.remove(png2)

    print("Saved combined image to:", out_file)

print("Plotting complete!")

# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")
