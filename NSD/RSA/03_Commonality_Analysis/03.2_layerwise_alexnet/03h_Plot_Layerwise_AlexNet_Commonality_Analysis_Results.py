import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.stats import sem
from utils import sign_permutation_cluster_test, get_eeg_times
import time

# Start time
start_time = time.time()

# --- Configuration ---
subject_list = [1, 4, 5, 6, 7, 8]
n_bootstraps = 10000


rois = ['V1', 'V2', 'V3', 'hV4', 'ventral', 'ventral']

# Defining individual labels for AlexNet layers
layer_keys = [
    'features.2',    # Conv1 + Pool
    'features.5',    # Conv2 + Pool
    'features.7',    # Conv3
    'features.9',    # Conv4
    'features.12',   # Conv5 + Pool
    'classifier.2',  # FC6
    'classifier.5',  # FC7
    'classifier.6'   # FC8 (Output)
]
layer_labels = {f'layer-{i}': f'AlexNet Layer {i}' for i in range(1, 9)}


cmap = plt.get_cmap('coolwarm')


layer_colors = {layer_keys[i]: cmap(v) for i,v in enumerate(np.linspace(0, 1, len(layer_keys)))}


# Base results root directory for AlexNet hierarchy
base_results_root = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/commonality_analysis/layerwise_alexnet'

PLOTS_DIR = '/home/jeffreykatab/Projects/fusion/NSD/plots/commonality_analysis/roi/layerwise_alexnet'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()
n_timepoints = len(times)

# --- Data Aggregation ---
# Dictionary structure: processed_data[layer_key][area_name] -> numpy array of shape (n_subjects, n_time)
processed_data = {layer: {} for layer in layer_keys}

print(">>> Aggregating ROI Data across AlexNet Layers <<<")
for layer in layer_keys:
    # Build path targeting the specific layer subfolder
    base_dir = os.path.join(base_results_root, 'layer-'+layer)
    
    for roi in tqdm(rois):
        subject_roi_corrs = []
        
        for subject in subject_list:
            roi_corrs = []
            
            try:
                path = os.path.join(base_dir, f'subject-{subject}', f'{roi}.npy')

                data = np.load(path) # Shape: (n_time, n_vertices)
                print(f"Subject {subject}, ROI {roi}: Shape = {data.shape}")
                
                roi_corrs.append(data)
                
            except FileNotFoundError:
                print(f" Subject {subject}, ROI {roi}: data not found")
                continue

            subject_roi_corrs.append(np.mean(roi_corrs, axis=0))
                                
        processed_data[layer][roi] = np.array(subject_roi_corrs)

# maxes = [0.006, 0.006, 0.006, 0.006, 0.018, 0.018] # Use this config to see all singificance bars

maxes = [0.005, 0.005, 0.005, 0.005, 0.0175, 0.0175]

# --- Plotting Subplots (2x3 Grid) ---
print("\n>>> Plotting 2x3 Layer Matrix Figure <<<")
fig, axes = plt.subplots(2, 3, figsize=(30, 18), sharex=False)

for idx, area in enumerate(rois):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # 1. Determine dynamic Y-limits for this specific ROI panel to optimize visibility
    local_max_y = 0.005
    for layer in layer_keys:
        data = processed_data[layer][area]
        if len(data) > 0:
            m = np.mean(data, axis=0)
            s = sem(data, axis=0)
            local_max_y = max(local_max_y, np.max(m + s))
            
    # 2. Iterate through layers and plot curves inside current axis
    for l_idx, layer in enumerate(layer_keys):
        section_data = processed_data[layer][area]
        if len(section_data) == 0:
            continue
            
        n_subs = len(section_data)
        m_group = np.mean(section_data, axis=0)
        s_err = sem(section_data, axis=0)
        color = layer_colors[layer]
        
        # Cluster Permutation Test
        res = sign_permutation_cluster_test(section_data, n_permutations=10000)
        sig_mask = np.zeros(n_timepoints, dtype=bool)
        for c_idx, _, _ in res['significant_clusters']:
            sig_mask[c_idx] = True

        # Significance Markers (Staggered layout at upper boundary)
        dot_y = local_max_y * (1.1 + (l_idx * 0.05)) 
        if np.any(sig_mask):
            ax.scatter(times[sig_mask], [dot_y] * np.sum(sig_mask), 
                       color=color, s=8, marker='s', alpha=0.8, edgecolors='none')
            
        #onsets = times[sig_mask]
        #print(f"ROI:{rois[idx]}; Layer:{layer}; Onset: {onsets[0]} ms; Mean score: {1000*np.mean(m_group[51:])}")

        # Bootstrap Peak Latency
        #boot_peaks = []
        #for _ in range(n_bootstraps):
        #    res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
        #    boot_peaks.append(times[np.argmax(np.mean(section_data[res_idx], axis=0))])
            
        #low, high = np.percentile(boot_peaks, [2.5, 97.5])
        #obs_peak = times[np.argmax(m_group)]
        
        # Cleaned legend display to prevent layout crowding
        legend_text = f"{layer}"
        
        # Plot timecourse and SEM variance bands
        ax.plot(times, m_group, color=color, lw=3.5, label=legend_text, zorder=3)
        ax.fill_between(times, m_group - s_err, m_group + s_err, color=color, alpha=0.06, zorder=2)
        
        # Mark absolute observed group peak coordinate
        peak_val = np.max(m_group)
        #ax.scatter(obs_peak, peak_val, color=color, s=35, edgecolors='white', zorder=5)
        


    # Subplot Styling Details
    ax.set_title(f"{area}", fontweight='bold', fontsize=30, pad=10)
    
    # Sync labels with the 2x3 geography
    #if col == 0:
    #    ax.set_ylabel("Explained Variance ($R^2$)", fontsize=24)
    #if row == 1:
    #    ax.set_xlabel('Time (ms)', fontsize=28)
        
    ax.axvline(0, color='black', linestyle='--', alpha=0.4)
    ax.axhline(0, color='black', lw=2, alpha=0.2)
    xticks = [-100, 0, 200, 400, 600]
    xlabels = [-100, 0, 200, 400, 600]
    ax.set_xticks(ticks=xticks)
    ax.set_xticklabels(labels=xlabels)
    ax.set_xlim(-100, 600)
    ax.set_ylim(bottom=-0.0005, top=maxes[idx])

    # Place localized legends in the upper right quadrant
    #ax.legend(loc='upper right', frameon=False, fontsize=26, ncol=2, handlelength=1.2)
    
    
    # Modern clean layout adjustments
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=26)

# Global Figure settings
plt.suptitle('', fontweight='bold', fontsize=22, y=1.0)

plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, 'layerwise_alexnet_ca.svg')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to: {save_path}")
print(f"Total Execution time: {time.time() - start_time:.2f} seconds.")