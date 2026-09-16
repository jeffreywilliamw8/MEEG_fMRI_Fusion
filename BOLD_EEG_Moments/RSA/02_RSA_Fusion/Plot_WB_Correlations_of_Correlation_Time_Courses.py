import numpy as np
import random
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cortex
from scipy.stats import spearmanr



# Start time
start_time = time.time()


# Random seed for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

N_VERTICES= 163842  # Number of vertices in fsaverage
N_TIME_POINTS = 1800  # Number of time points in the EEG data
EEG_TEMPORAL_RESOLUTION = 2  # Temporal resolution of EEG data in ms (2 ms = 500 Hz)


subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # List of subjects to process
#=========================================================
# Loading the Encoding Models correlatation time courses
#=========================================================
em_left_corrs = np.zeros((N_TIME_POINTS, N_VERTICES), dtype=np.float32)
em_right_corrs = np.zeros((N_TIME_POINTS, N_VERTICES), dtype=np.float32)
for subject in subject_list:
    corrs_path = f'/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/encoding_fusion/whole_brain/eeg_temporal_resolution-{EEG_TEMPORAL_RESOLUTION}ms/subject-{subject}'
    em_left_corrs += np.load(os.path.join(corrs_path, 'correlations_left.npy'))
    em_right_corrs += np.load(os.path.join(corrs_path, 'correlations_right.npy'))
em_left_corrs /= len(subject_list)
em_right_corrs /= len(subject_list)
print("Shape of the EM correlation time courses (left, right): ({}, {})".format(em_left_corrs.shape, em_right_corrs.shape))

#=========================================================
# Loading the RSA correlatation time courses
#=========================================================
rsa_left_corrs = np.zeros((N_TIME_POINTS, N_VERTICES), dtype=np.float32)
rsa_right_corrs = np.zeros((N_TIME_POINTS, N_VERTICES), dtype=np.float32)
for fmri_subject in subject_list:
    corrs_path = f'/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/results/correlations/searchlight_fusion/eeg_rdm_metric-pearsonr/n_neighbours-100/aggregated_results/subject-{fmri_subject}'
    rsa_left_corrs += np.load(os.path.join(corrs_path, f'subject-{fmri_subject}_left_hemisphere_timecourse.npy'))
    rsa_right_corrs += np.load(os.path.join(corrs_path, f'subject-{fmri_subject}_right_hemisphere_timecourse.npy'))
rsa_left_corrs /= len(subject_list)
rsa_right_corrs /= len(subject_list)
print("Shape of the RSA correlation time courses (left, right): ({}, {})".format(rsa_left_corrs.shape, rsa_right_corrs.shape))

#==============================================================================================
# Computing the vertex-wise Spearman correlations between the time courses for each method
#==============================================================================================
left_corrs = np.zeros((N_VERTICES,), dtype=np.float32)
right_corrs = np.zeros((N_VERTICES,), dtype=np.float32)
for i in tqdm(range(N_VERTICES)):
    left_corrs[i] = spearmanr(em_left_corrs[:, i], rsa_left_corrs[:, i]).correlation
    right_corrs[i] = spearmanr(em_right_corrs[:, i], rsa_right_corrs[:, i]).correlation
print("Shape of the vertex-wise correlations (left, right): ({}, {})".format(left_corrs.shape, right_corrs.shape))

#=========================================================
# Plotting Vertex-wise Correlations on the brain surface
#=========================================================
subject = 'fsaverage'
corrs = np.append(left_corrs, right_corrs)
vertex_data = cortex.Vertex(corrs, subject, cmap='viridis', vmin=0, vmax=max(corrs), with_colorbar=True)

# =========================================================
# Show brain plot
# =========================================================
fig = cortex.quickshow(
    vertex_data,
    with_curvature=True,
    curvature_brightness=0.5,
    with_rois=True,
    with_labels=True,
    linewidth=5,
    linecolor=(1, 1, 1),
    with_colorbar=True,
    roi_list=['V1', 'V4', 'PPA', 'FFA', 'OFA'],
    labelsize='40pt'
)
plt.title('Correlation of Encoding and RSA Time Courses' , fontdict={'fontsize': 42})
plot_dir = '/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
# Save flatmap
fig.savefig(os.path.join(plot_dir, 'vertex_wise_correlations_of_time_courses.svg'), dpi=300, bbox_inches='tight', transparent=False, format='svg')
plt.close()

# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print("Plot saved to: {}".format(os.path.join(plot_dir, 'vertex_wise_correlations_of_time_courses.svg')))
print(f"Execution time: {execution_time:.2f} seconds")