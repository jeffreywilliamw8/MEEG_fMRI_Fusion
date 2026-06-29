import numpy as np
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

# Define the grouping hierarchy
roi_groups = {
    'V1': ['V1v', 'V1d'],
    'V2': ['V2v', 'V2d'],
    'V3': ['V3v', 'V3d'],
    'hV4': ['hV4'],
    'ventral': ['ventral']
}

rois = ['V1', 'V2', 'V3', 'hV4', 'ventral']
area_colors = [
    "#480758",  # V1 (Deep Purple)
    "#5b4fab",  # V2 (Dark Blue)
    "#63a1cc",  # V3 (Steel Blue)
    "#8fd744",  # V4 (Light Green)
    "#fd9f25"   # ventral (Orange)
]


n_bootstraps = 10000

# Mapping keys to plot vertical positions (pos_idx)
partitions = {
    'shared_vision_language': {'title': 'Total Vision DNN and LLM Variance', 'pos_idx': 0},
    'unique_vision': {'title': 'Unique Vision DNN Variance', 'pos_idx': 1},
    'unique_language': {'title': 'Unique LLM Variance', 'pos_idx': 2}
    
}
maxes = [0.017, 0.0035, 0.005]
# Source data tracking directory 
base_results_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/variance_partitioning/roi'
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots/variance_partitioning'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()
n_timepoints = len(times)

# --- Data Aggregation ---
# Structural layout: aggregated_data[partition_key][roi_idx] -> shape: (n_subjects, n_time)
aggregated_data = {part: [[] for _ in rois] for part in partitions.keys()}

print(">>> Aggregating Variance Partitioning ROI Data <<<")

for roi_idx, roi in enumerate(rois):
    
    for subject in subject_list:
        # Temporary dictionary to collect subject timecourses across sub_rois and hemispheres
        subject_pool = {part: [] for part in partitions.keys()}
        
        file_path = os.path.join(base_results_dir, f'subject-{subject}', f'{roi}.npy')

        # Load saved result dictionary
        data = np.load(file_path, allow_pickle=True).item()
        #data = np.nan_to_num(data, nan=0.0)
        
        for part in partitions.keys():
            subject_pool[part].append(data[part])
                        
        # Append to global area index
        for part in partitions.keys():
            aggregated_data[part][roi_idx].append(subject_pool[part])

# Cast lists into clean arrays for calculations
for part in partitions.keys():
    for roi_idx in range(len(rois)):
        data = np.array(aggregated_data[part][roi_idx])
        print("Shape of data:", data.shape)
        aggregated_data[part][roi_idx] = np.mean(data, axis=1)

# --- Plotting 3x1 Vertically Stacked Subplots ---
print("Plotting 3x1 Vertical Stack...")
fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=False)

for part_key, config in partitions.items():
    idx = config['pos_idx']
    ax = axes[idx]
    data_list = aggregated_data[part_key]
    
    # Calculate unique local y-limits to maximize tracking resolution inside this subplot
    all_means = [np.mean(d, axis=0) for d in data_list if len(d) > 0]
    all_sems = [sem(d, axis=0) for d in data_list if len(d) > 0]
    local_max_y = max([np.max(m + s) for m, s in zip(all_means, all_sems)]) if all_means else 0.05
    
    print(f"Plotting Panel: {config['title']}")
    
    for i, area_data in enumerate(data_list):
        if len(area_data) == 0:
            continue
            
        n_subs = len(area_data)
        m_group = np.mean(area_data, axis=0)
        s_err = sem(area_data, axis=0)
        color = area_colors[i]
        
        # 1. Cluster Permutation Test
        cluster_results = sign_permutation_cluster_test(area_data, n_permutations=10000)
        sig_mask = np.zeros(n_timepoints, dtype=bool)
        for cluster_idx, _, _ in cluster_results['significant_clusters']:
            sig_mask[cluster_idx] = True

        # 2. Bootstrap Peak Latency
        boot_peaks = []
        for _ in range(n_bootstraps):
            res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
            boot_peaks.append(times[np.argmax(np.mean(area_data[res_idx], axis=0))])
        
        low, high = np.percentile(boot_peaks, [2.5, 97.5])
        obs_peak = times[np.argmax(m_group)]
        
        # 3. Curve and Variance Ribbon Plotting
        leg_text = f"{rois[i]}"
        ax.plot(times, m_group, color=color, lw=5, label=leg_text, zorder=3)
        ax.fill_between(times, m_group - s_err, m_group + s_err, color=color, alpha=0.1, zorder=2)
        
        # Peak Markers with Errorbars
        peak_val = np.max(m_group)
        #ax.scatter(obs_peak, peak_val, color=color, s=60, edgecolors='white', zorder=5)
        #ax.errorbar(obs_peak, peak_val, xerr=[[obs_peak-low], [high-obs_peak]], 
        #            fmt='none', ecolor='k', elinewidth=1, capsize=3, zorder=4)
        
        # Significance Markers (Staggered layout at upper boundary)
        dot_y = local_max_y * (1.1 + (i * 0.05)) 
        if np.any(sig_mask):
            ax.scatter(times[sig_mask], [dot_y] * np.sum(sig_mask), 
                       color=color, s=8, marker='s', alpha=0.8, edgecolors='none')

    # Subplot Aesthetic Configuration
    ax.set_title(config['title'], fontweight='bold', fontsize=26, pad=15)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=2, alpha=0.2)
    ax.set_xlim(-100, 600)
    #ax.set_ylim(bottom=-0.0001, top=maxes[idx])
    ax.set_ylim(bottom=-0.0001, top=0.006)
    
    # Consistent vertical axis labels across all rows
    #ax.set_ylabel("($R^2$ Score)", fontsize=26)
    
    # Only assign the x-axis label to the final bottom subplot
    #if idx == 2:
    #    ax.set_xlabel("Time (ms)", fontsize=26)
        
    #ax.legend(loc='upper right', frameon=False, fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=22)

# Global Figure Layout settings
plt.suptitle('', fontweight='bold', fontsize=18, y=1.01)

plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, "roi_rsa_vl_variance_partitioning_2.svg")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nExecution Complete! Stacked matrix figure saved to: {save_path}")
print(f"Total Execution Time: {time.time() - start_time:.2f} seconds.")