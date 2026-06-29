import os
import numpy as np
from copy import copy
import cortex
from PIL import Image
import cortex.polyutils
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from berg import BERG
from utils import get_eeg_times
import time


# Start time
start_time = time.time()


# =============================================================================
# Loading the correlations
# =============================================================================
times = get_eeg_times()
# Correlations
left_corrs = np.zeros((359, 163842)) # Initialize an empty array for left hemisphere results
right_corrs = np.zeros((359, 163842)) # Initialize an empty array for right hemisphere results
subject_list = [1,4,5,6,7,8]   # List of subjects to process
# Loop through each subject and load the corresponding results
for subject in subject_list:
    corrs_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion/whole_brain/subject-{subject}'
    data_dir_l = os.path.join(corrs_dir, 'correlations_left.npy')
    data_dir_r = os.path.join(corrs_dir, 'correlations_right.npy')
    left_corrs += np.load(data_dir_l)
    right_corrs += np.load(data_dir_r)
left_corrs /= len(subject_list)  # Average the results across subjects
right_corrs /= len(subject_list)  # Average the results across subjects

# Noise ceilings
ncsnr_left = np.zeros(163842).astype(np.float32) # Initialize an empty array for left hemisphere results
ncsnr_right = np.zeros(163842).astype(np.float32)  # Initialize an empty array for right hemisphere results
# Loop through each subject and load the corresponding results
for subject in subject_list:
    berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
    metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)
    ncsnr_left += metadata['fmri']['lh_ncsnr']
    ncsnr_right += metadata['fmri']['rh_ncsnr']
ncsnr_left /= len(subject_list)  # Average the results across subjects
ncsnr_right /= len(subject_list)  # Average the results across subjects

# Selecting only vertices above noise ceiling threshold
valid_idx_left = np.where(ncsnr_left >= 0.2)[0]
valid_idx_right = np.where(ncsnr_right >= 0.2)[0]



# Peak Latencies
#peak_latencies_left = np.zeros(163842, dtype=float)
#peak_latencies_right = np.zeros(163842, dtype=float)
peak_latencies_left = np.full(163842, np.nan, dtype=float)
peak_latencies_right = np.full(163842, np.nan, dtype=float)

print("Processing left hemisphere...")
#for idx in tqdm(valid_idx_left):
for idx in tqdm(valid_idx_left):
    peak_latencies_left[idx] = times[np.argmax(left_corrs[:, idx])]

print("Processing right hemisphere...")
for idx in tqdm(valid_idx_right):
    peak_latencies_right[idx] = times[np.argmax(right_corrs[:, idx])]

# Average correlations
avg_corrs_left = np.mean(left_corrs, axis=0)
avg_corrs_right = np.mean(right_corrs, axis=0)



# =============================================================================
# Plot parameters for colorbar
# =============================================================================
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
#subject = 'fsaverage'
subject = 'fsaverage_nsd_sub-01'


plots_dir = '/scratch/jeffreykatab/Code/Encoding_Models/NSD/plots/encoding_fusion'
os.makedirs(plots_dir, exist_ok=True)
file_name = 'sa_peak_latencies_time_avg_corrs_nsd.png'

# Combine hemispheres for both datasets
peak_latencies = np.append(peak_latencies_left, peak_latencies_right)
avg_corrs = np.append(avg_corrs_left, avg_corrs_right)


# Create pycortex vertex data for both maps
vertex_data_1 = cortex.Vertex(peak_latencies, subject, cmap='viridis', vmin=50, vmax=250)
vertex_data_2 = cortex.Vertex(avg_corrs, subject, cmap='viridis', vmin=0, vmax=max(avg_corrs))

# --------------------
# 1. Save each flatmap to disk using pycortex quickflat (PNG)
# --------------------
png1 = os.path.join(plots_dir, "surface1.png")
png2 = os.path.join(plots_dir, "surface2.png")

print("Plotting...")

# Generate the two cortical flatmaps
cortex.quickflat.make_png(
    png1,
    vertex_data_1,
    with_curvature=True,
    curvature_brightness=0.5,
    with_rois=True,
    roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
    with_labels=True
)
cortex.quickflat.make_png(
    png2,
    vertex_data_2,
    with_curvature=True,
    curvature_brightness=0.5,
    with_rois=True,
    roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
    with_labels=True
)

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
axes[0].set_title("Peak latencies (ms)",fontsize=22)

axes[1].imshow(img2)
axes[1].axis("off")
axes[1].set_title("Time-averaged correlations (Pearson's r)",fontsize=22)

plt.tight_layout()

fig.savefig(os.path.join(plots_dir, file_name), dpi=300, bbox_inches='tight', transparent=False, format='png')
plt.savefig(file_name, dpi=300, bbox_inches="tight", transparent=False)
plt.close()
# Cleanup temp files
os.remove(png1)
os.remove(png2)

print("✅ Saved combined image to:", os.path.join(plots_dir, file_name))


print(f"Execution complete! Total Time: {time.time() - start_time:.2f} seconds.")