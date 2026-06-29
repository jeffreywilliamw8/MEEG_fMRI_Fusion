import os
import numpy as np
import cortex
from PIL import Image
import matplotlib
from tqdm import tqdm
from utils import get_significance_mask, get_eeg_times
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from berg import BERG
import time

# Start time
start_time = time.time()

# =============================================================================
# Load the encoding models and RSA correlation results
# =============================================================================
eeg_times = get_eeg_times()
post_stimulus_times = np.where(eeg_times >= 0)[0] # We will compute similarity/dissimilarity of time courses only for post-stimulus time points (0ms to 600ms)
n_timepoints = 359
n_vertices = 163842

em_corrs_left = np.zeros((n_timepoints, n_vertices))
em_corrs_right = np.zeros((n_timepoints, n_vertices))
subject_list = [1]

print("Loading Encoding results...")
for subject in subject_list:
    corrs_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion/whole_brain/subject-{subject}'
    data_dir_l = os.path.join(corrs_dir, 'correlations_left.npy')
    data_dir_r = os.path.join(corrs_dir, 'correlations_right.npy')
    em_corrs_left += np.load(data_dir_l)
    em_corrs_right += np.load(data_dir_r)
em_corrs_left /= len(subject_list)
em_corrs_right /= len(subject_list)

# Combine hemispheres into full-brain matrices: shape (n_timepoints, 2 * n_vertices)
em_full_brain = np.concatenate([em_corrs_left, em_corrs_right], axis=1)
print("Combined Encoding full-brain shape:", em_full_brain.shape)

rsa_corrs_left = np.zeros((n_timepoints, n_vertices)).astype(np.float32)
rsa_corrs_right = np.zeros((n_timepoints, n_vertices)).astype(np.float32)

print("Loading RSA results...")
for subject in subject_list:
    corrs_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/searchlight_fusion/n_neighbours-100/metric_correlation/aggregated_results/subject-{subject}'
    data_dir_l = os.path.join(corrs_dir, f'subject-{subject}_lh_hemisphere_timecourse.npy')
    data_dir_r = os.path.join(corrs_dir, f'subject-{subject}_rh_hemisphere_timecourse.npy')
    rsa_corrs_left += np.load(data_dir_l)
    rsa_corrs_right += np.load(data_dir_r)
rsa_corrs_left /= len(subject_list)
rsa_corrs_right /= len(subject_list)

# Combine hemispheres into full-brain matrices: shape (n_timepoints, 2 * n_vertices)
rsa_full_brain = np.concatenate([rsa_corrs_left, rsa_corrs_right], axis=1)
print("Combined RSA full-brain shape:", rsa_full_brain.shape)

# =============================================================================
# Vertex-wise Standard Spearman Correlation Loop
# =============================================================================
left_corrs = np.zeros(n_vertices, dtype=np.float32)
right_corrs = np.zeros(n_vertices, dtype=np.float32)
for v in tqdm(range(n_vertices)):
    left_corrs[v] = spearmanr(em_corrs_left[:, v], rsa_corrs_left[:, v]).correlation
    right_corrs[v] = spearmanr(em_corrs_right[:, v], rsa_corrs_right[:, v]).correlation
print("Shape of the vertex-wise correlations (left, right): ({}, {})".format(left_corrs.shape, right_corrs.shape))
wb_corrs = np.append(left_corrs, right_corrs)
print("Mean similarity across vertices: ", np.nanmean(wb_corrs))



# Clean up any residual edge-case NaNs if they slipped through
#wb_corrs = np.nan_to_num(wb_corrs, nan=0.0)

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
# Render and Save Single Flatmap Result
# =============================================================================
plots_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots'
os.makedirs(plots_dir, exist_ok=True)

print("Plotting dis/similarity metrics on brain flatmap...")

vertex_data = cortex.Vertex(wb_corrs, subject, cmap='viridis', vmin=0, vmax=1.0, with_colorbar=True)

# ----------------------------
# Show brain
# ----------------------------
fig = cortex.quickshow(
    vertex_data,
    with_curvature=True,
    curvature_brightness=0.5,
    with_rois=True,
    roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
    with_labels=False,
    linewidth=5,
    linecolor=(1, 1, 1),
    with_colorbar=True
)
plt.title('Correlations of Encoding and RSA Time Courses' , fontdict={'fontsize': 30})
plot_dir = '/home/jeffreykatab/Projects/fusion/NSD/RSA/plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
# Save flatmap
fig.savefig(os.path.join(plot_dir, 'wb_em_rsa_ctc_spearman.svg'), dpi=300, bbox_inches='tight', transparent=False, format='svg')
plt.close()

# End time
end_time = time.time()
print(f"\nExecution Complete! Total Time: {end_time - start_time:.2f} seconds.")

# /home/jeffreykatab/Projects/fusion/NSD/RSA/plots/wb_em_rsa_ctc_spearman.svg