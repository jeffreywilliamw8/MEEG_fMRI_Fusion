import numpy as np
from nilearn import plotting, datasets
from nilearn.surface import load_surf_mesh
import matplotlib.pyplot as plt
import os
import argparse
import time
from tqdm import tqdm
import pickle

# Start time
start_time = time.time()
#========================
# Input arguments
#========================

parser = argparse.ArgumentParser()
parser.add_argument('--eeg_frequency', type=int, default=100)
parser.add_argument('--eeg_channel_policy', type=str, default='average', 
                    help='Policy for handling EEG channels: "append" to use all channels, "average" to average across subjects',
                    choices=['append', 'average'])
args = parser.parse_args()

# Load fsaverage surface mesh (this fetches the file paths)
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')

# Load the surface mesh data from the file paths
left_mesh = load_surf_mesh(fsaverage.infl_left)
right_mesh = load_surf_mesh(fsaverage.infl_right)

# Get the number of vertices from the loaded mesh data
n_vertices_left_hemi = left_mesh[0].shape[0]
n_vertices_right_hemi = right_mesh[0].shape[0]


# =============================================================================
# Load the encoding accuracy results
# =============================================================================
n_time_points = int(3.7*args.eeg_frequency) # 185 or 370 time points if frequency is 50 Hz or 100Hz respectively
results_left = np.zeros((n_time_points, n_vertices_left_hemi), dtype=np.float32) # Initialize an empty array for left hemisphere results
results_right = np.zeros((n_time_points, n_vertices_right_hemi), dtype=np.float32) # Initialize an empty array for right hemisphere results
subject_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']   # List of subjects to process
# Loop through each subject and load the corresponding results
for subject in subject_list:
    correlations_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/eeg2fmri_wb/{args.eeg_frequency}_Hz/eeg_channel_policy_{args.eeg_channel_policy}/fmri_sub-{subject}'
    data_dir_l = os.path.join(correlations_dir, 'correlations_left.npy')
    data_dir_r = os.path.join(correlations_dir, 'correlations_right.npy')
    results_left += np.load(data_dir_l)
    results_right += np.load(data_dir_r)
results_left /= len(subject_list)  # Average the results across subjects
results_right /= len(subject_list)  # Average the results across subjects
min_value = 0.  # Minimum correlation value across both hemispheres
Max_Value = max(np.max(results_left), np.max(results_right)) 
print("Shape of corr_lh: {}, corr_rh: {}".format(results_left.shape, results_right.shape))  # Displaying the shape of the correlation arrays
noise_ceiling_file = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_noiseceiling_space-fsaverage_task-test_hemi-{}_n-10.pkl'
noise_ceiling = pickle.load(open(noise_ceiling_file.format("01", "01", 'left'), 'rb'))[1]
noisy_voxels = noise_ceiling < 20
results_left[:, noisy_voxels] = np.nan

plots_dir = '/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/plots/encoding_fusion_correlations/3D_plots/subject_averaged'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)  # Create the directory if it does not exist

eeg_path = '/scratch/giffordale95/projects/eeg_moments/dataset/preprocessed_data/eeg/sub-01/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-'+format(args.eeg_frequency, '04')+'/preprocessed_data.npy'
times = np.load(eeg_path, allow_pickle=True).item()['times']  # Loading the time vector from the EEG data
n_time_points = len(times)   # number of time points

# Creating a dictionary that maps index to timestamp
time_map = {i: times[i] for i in range(185)}

"""
for t in tqdm(len(times)):
    # Create a single figure with 4 subplots (2 rows, 2 columns)
    # Adjust figsize as needed to make space for 4 plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), subplot_kw={'projection': '3d'})

    # Plot Lateral View (First Row)
    lateral_row_idx = 0
    view = 'lateral'

    # Left Hemisphere, Lateral View (Top Left)
    plotting.plot_surf_stat_map(
        surf_mesh=fsaverage.infl_left,
        stat_map=results_left[t,:],
        hemi='left',
        title=f'Left Hemisphere ({view} view)',
        colorbar=True, # No colorbar on the left plot
        bg_map=fsaverage.sulc_left,
        view=view,
        vmin=min_value,
        vmax=Max_Value,
        axes=axes[lateral_row_idx, 0] # First row, first column
    )

    # Right Hemisphere, Lateral View (Top Right)
    plotting.plot_surf_stat_map(
        surf_mesh=fsaverage.infl_right,
        stat_map=results_right[t,:],
        hemi='right',
        title=f'Right Hemisphere ({view} view)',
        colorbar=True, # Colorbar on the right plot
        bg_map=fsaverage.sulc_right,
        view=view,
        vmin=min_value,
        vmax=Max_Value,
        axes=axes[lateral_row_idx, 1] # First row, second column
    )

    # Plot Posterior View (Second Row)
    medial_row_idx = 1
    view = 'posterior'
    # Left Hemisphere, Medial View (Bottom Left)
    plotting.plot_surf_stat_map(
        surf_mesh=fsaverage.infl_left,
        stat_map=results_left[t,:],
        hemi='left',
        title=f'Left Hemisphere ({view} view)',
        colorbar=True, # No colorbar on the left plot
        bg_map=fsaverage.sulc_left,
        view=view,
        vmin=min_value,
        vmax=Max_Value,
        axes=axes[medial_row_idx, 0] # Second row, first column
    )

    # Right Hemisphere, Medial View (Bottom Right)
    plotting.plot_surf_stat_map(
        surf_mesh=fsaverage.infl_right,
        stat_map=results_right[t,:],
        hemi='right',
        title=f'Right Hemisphere ({view} view)',
        colorbar=True, # Colorbar on the right plot
        bg_map=fsaverage.sulc_right,
        view=view,
        vmin=min_value,
        vmax=Max_Value,
        axes=axes[medial_row_idx, 1] # Second row, second column
    )


    file_name = f'time_point_{t}.png'
    fig.suptitle(f'Encoding Correlations (Pearson\'s r) | Time: {time_map[t]*1000:.2f} ms', fontsize=22) 
    fig.savefig(os.path.join(plots_dir, file_name), dpi=300, bbox_inches='tight', transparent=False, format='svg')
    plt.close()
"""

for t in tqdm([0,10,20,30,40,50,60,100]):
    # 1. Just call the plot. Nilearn creates the figure/axis internally.
    display = plotting.plot_surf_stat_map(
        surf_mesh=fsaverage.infl_left,
        stat_map=results_left[t, :],
        hemi='left',
        colorbar=False,
        bg_map=fsaverage.sulc_left,
        view='lateral',
        vmin=0,
        vmax=Max_Value,
        cmap='hot'
    )

    # 2. Grab the figure and axis Nilearn just made to clean them up
    fig = plt.gcf()
    ax = plt.gca()
    
    ax.set_axis_off()      # Kill the axes
    fig.patch.set_alpha(0) # Make background transparent for Inkscape

    # 3. Save it
    file_name = f'time_point_{t}.png'
    plt.savefig(os.path.join(plots_dir, file_name), 
                bbox_inches='tight', 
                pad_inches=0, 
                transparent=True, 
                format='png')
    
    plt.close() # Crucial to prevent memory leaks in a loop
    # End time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds.")