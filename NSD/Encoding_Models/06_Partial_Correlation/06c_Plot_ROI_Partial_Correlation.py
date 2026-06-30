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
hemis = ['lh', 'rh']

roi_groups = {
    'V1': ['V1v', 'V1d'],
    'V2': ['V2v', 'V2d'],
    'V3': ['V3v', 'V3d'],
    'hV4': ['hV4'],
    'ventral': ['ventral']
}

area_labels = ['V1', 'V2', 'V3', 'hV4', 'ventral']
area_colors = [
    "#480758",  # V1 (Deep Purple)
    "#5b4fab",  # V2 (Dark Blue)
    "#63a1cc",  # V3 (Steel Blue)
    "#8fd744",  # V4 (Light Green)
    "#fd9f25"   # ventral (Orange)
]

n_bootstraps = 10000

partitions = {
    'total_correlation': {'title': 'Total Vision DNN and LLM Variance', 'pos_idx': 0},
    'vision_partial_correlation': {'title': 'Unique Vision DNN Variance', 'pos_idx': 1},
    'language_partial_correlation': {'title': 'Unique LLM Variance', 'pos_idx': 2}
}

# Pathing updated for Partial Correlation outputs
# f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/partial_correlation/subject-{args.subject}'
base_results_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/partial_correlation'
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/plots/partial_correlation'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()
n_timepoints = len(times)

# --- Data Aggregation ---
aggregated_data = {part: [[] for _ in area_labels] for part in partitions}

print(">>> Aggregating Encoding Partial Correlation ROI Data <<<")

for a_idx, area in enumerate(area_labels):
    sub_rois = roi_groups[area]
    
    for subject in subject_list:
        # Dictionary to hold the raw (n_time, n_vertices) arrays across sub-ROIs and hemispheres
        pooled_vertices = {part: [] for part in partitions}
        
        for sub_roi in sub_rois:
            for hemi in hemis:
                file_path = os.path.join(base_results_dir, f'subject-{subject}', f'{sub_roi}_{hemi}.npy')
                
                if os.path.exists(file_path):
                    # Load the unified partial correlation dictionary file
                    results_dict = np.load(file_path, allow_pickle=True).item()
                    
                    for part in partitions:
                        # shape: (n_time, n_vertices)
                        data = results_dict[part]
                        
                        pooled_vertices[part].append(data)
                        
        # For each partition type, concatenate across vertices and take the spatial mean
        for part in partitions:
            if len(pooled_vertices[part]) > 0:
                # Concatenate all hemispheres and sub-ROIs along the vertex dimension (axis=1)
                all_area_vertices = np.concatenate(pooled_vertices[part], axis=1)
                # Average across the combined vertex pool to get a 1D timecourse
                subject_timecourse = np.mean(all_area_vertices, axis=1)
                aggregated_data[part][a_idx].append(subject_timecourse)

# Cast lists into clean numpy arrays
for part in partitions:
    for a_idx in range(len(area_labels)):
        aggregated_data[part][a_idx] = np.array(aggregated_data[part][a_idx])

# --- Plotting 3x1 Vertical Stack ---
print("\n>>> Plotting 3x1 Vertical Stack <<<")
fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=False)
maxes = [0.4, 0.34, 0.20]
for part_key, config in partitions.items():
    idx = config['pos_idx']
    ax = axes[idx]
    data_list = aggregated_data[part_key]
    
    # Calculate unique local y-limits to maximize tracking resolution inside this subplot
    all_means = [np.mean(d, axis=0) for d in data_list if len(d) > 0]
    all_sems = [sem(d, axis=0) for d in data_list if len(d) > 0]
    
    # Handle local limits dynamically considering partial correlation bounds
    local_max_y = max([np.max(m + s) for m, s in zip(all_means, all_sems)]) if all_means else 0.1
    local_min_y = min([np.min(m - s) for m, s in zip(all_means, all_sems)]) if all_means else -0.02
    
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
        leg_text = f"{area_labels[i]}: {obs_peak:.0f}ms [{low:.0f}-{high:.0f}ms]"
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
    
    # Adjust y-limit constraints smoothly to support standard correlation ranges
    #ax.set_ylim(bottom=-0.01, top=maxes[idx])
    ax.set_ylim(bottom=-0.01, top=0.32)
    
    # Update label assignments for partial correlation dimensions
    #ax.set_ylabel("$R^2$ Score", fontsize=26)
    
    #if idx == 2:
    #    ax.set_xlabel("Time (ms)", fontsize=26)
        
    #ax.legend(loc='upper right', frameon=False, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=26)
    #ax.legend(loc='upper right', frameon=False, fontsize=26)

# Global Figure Layout settings
plt.suptitle('', fontweight='bold', fontsize=26, y=1.01)

plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, "roi_vl_partial_correlation.svg")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSuccess! Stacked partial correlation plot saved to: {save_path}")
print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")