import numpy as np
from nilearn import plotting, datasets
from nilearn.surface import load_surf_mesh
import matplotlib.pyplot as plt
import os
from berg import BERG
from tqdm import tqdm
import time


# Start time
start_time = time.time()
#========================
# Input arguments
#========================

# Load fsaverage surface mesh (this fetches the file paths)
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')

# Load the surface mesh data from the file paths
left_mesh = load_surf_mesh(fsaverage.infl_left)

# Get the number of vertices from the loaded mesh data
n_vertices_left_hemi = left_mesh[0].shape[0]


# =============================================================================
# Load the encoding accuracy results
# =============================================================================
# Encoding model results
em_corrs_left = np.zeros((359, 163842)) # Initialize an empty array for left hemisphere results
subject_list = [1]   # List of subjects to process
# Loop through each subject and load the corresponding results
print("Loading Encoding model results...")
for subject in subject_list:
    corrs_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion/whole_brain/subject-{subject}'
    data_dir_l = os.path.join(corrs_dir, 'correlations_left.npy')
    em_corrs_left = np.load(data_dir_l)
print("Shape of the Encoding correlation time courses (left hemisphere): ", em_corrs_left.shape)
Max_Value = np.max(em_corrs_left) #
print("Global max value: ", Max_Value)


# Noise ceilings
ncsnr_left = np.zeros(163842).astype(np.float32) # Initialize an empty array for left hemisphere results
# Loop through each subject and load the corresponding results
for subject in subject_list:
    berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
    metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)
    ncsnr_left = metadata['fmri']['lh_ncsnr']

noisy_voxels = ncsnr_left < 0.2
em_corrs_left[:, noisy_voxels] = np.nan

plots_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/plots/encoding_fusion/3D_brain_plots/subject_averaged'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)  # Create the directory if it does not exist

cmaps = ['viridis', 'hot'] # Color maps to be used for plotting
views = ['lateral', 'ventral', 'posterior']

time_map = { # dictionary mapping time indices to milliseconds w.r.t stimulus onset
    '51': '0_ms',
    '77': '50_ms',
    '102': '100_ms',
    '128': '150_ms',
    '153': '200_ms',
    '179': '250_ms',
    '205': '300_ms',
    '230': '350_ms',
    '256': '400_ms',
    '281': '450_ms',
    '307': '500_ms',
    '333': '550_ms',
    '358': '600_ms'
}

for t in tqdm([51, 77, 102, 128, 153, 179, 205, 230, 256, 281, 307, 333, 358]):
    for cmap in cmaps:
        for view in views:
            if t==51 and view=='lateral':
                plotting.plot_surf_stat_map(
                surf_mesh=fsaverage.infl_left,
                stat_map=em_corrs_left[t, :],
                hemi='left',
                colorbar=True, # Plotting the color bar only once, for the first time point
                cbar_tick_format="%.2f",
                bg_map=fsaverage.sulc_left,
                view=view,
                vmin=0,
                vmax=Max_Value,
                cmap=cmap)
            

                fig = plt.gcf()
                ax = plt.gca()
                
                #ax.set_axis_off()      # remove axes
                fig.patch.set_alpha(0) # Make background transparent
                    
                file_name = f'{time_map[str(t)]}_{view}_view_cmap-{cmap}_wcb.png'
                plt.savefig(os.path.join(plots_dir, file_name), 
                            bbox_inches='tight', 
                            pad_inches=0, 
                            transparent=True, 
                            format='png')
                
                plt.close()

                # Making the same plot but without color bar
                plotting.plot_surf_stat_map(
                surf_mesh=fsaverage.infl_left,
                stat_map=em_corrs_left[t, :],
                hemi='left',
                colorbar=False,
                cbar_tick_format="%.2f",
                bg_map=fsaverage.sulc_left,
                view=view,
                vmin=0,
                vmax=Max_Value,
                cmap=cmap)
            

                fig = plt.gcf()
                ax = plt.gca()
                
                #ax.set_axis_off()      # remove axes
                fig.patch.set_alpha(0) # Make background transparent
                    
                file_name = f'{time_map[str(t)]}_{view}_view_cmap-{cmap}.png'
                plt.savefig(os.path.join(plots_dir, file_name), 
                            bbox_inches='tight', 
                            pad_inches=0, 
                            transparent=True, 
                            format='png')
                
                plt.close()

            plotting.plot_surf_stat_map(
                surf_mesh=fsaverage.infl_left,
                stat_map=em_corrs_left[t, :],
                hemi='left',
                colorbar=False,
                cbar_tick_format="%.2f",
                bg_map=fsaverage.sulc_left,
                view=view,
                vmin=0,
                vmax=Max_Value,
                cmap=cmap
            )
            

            fig = plt.gcf()
            ax = plt.gca()
            
            #ax.set_axis_off()      # remove axes
            fig.patch.set_alpha(0) # Make background transparent
                
            file_name = f'{time_map[str(t)]}_{view}_view_cmap-{cmap}.png'
            plt.savefig(os.path.join(plots_dir, file_name), 
                        bbox_inches='tight', 
                        pad_inches=0, 
                        transparent=True, 
                        format='png')
            
            plt.close()

    # End time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds.")