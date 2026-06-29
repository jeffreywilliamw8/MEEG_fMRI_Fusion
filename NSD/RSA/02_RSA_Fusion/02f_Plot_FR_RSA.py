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

rois = ['V1', 'V2', 'V3', 'hV4', 'ventral']

# Fixed Approach-specific Color Profile
approach_colors = {
    'classical': '#1f77b4',  # Crisp Blue
    'fr_rsa': '#2ca02c'     # Emerald Green
}
approach_labels = {
    'classical': 'Classical RSA',
    'fr_rsa': 'Feature-Reweighted RSA'
}

n_bootstraps = 10000

# --- Directory Paths ---
# Classical ('Vanilla') RSA Data Directory
base_classical_dir = os.path.join(
    '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/',
    'correlations',
    'roi',
    f'eeg_rdm_metric-correlation',
    f'fmri_rdm_metric-correlation',
)

# Feature-Reweighted RSA Data Directory
# f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/feature_reweighted_rsa/target-fmri/target_rdm_metric-correlation
base_fr_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/feature_reweighted_rsa/target-fmri/target_rdm_metric-correlation'

PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots/roi_rsa_control'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()
n_timepoints = len(times)

# --- Data Aggregation ---
# Structure: aggregated_data[approach][area_name] -> shape (n_subjects, n_time)
aggregated_data = {'classical': {}, 'fr_rsa': {}}

print(">>> Aggregating Classical vs Feature-Reweighted ROI Data (with NaN scrubbing) <<<")

for approach, base_dir in [('classical', base_classical_dir), ('fr_rsa', base_fr_dir)]:

    for roi in rois:
        subject_roi_corrs = []

        for subject in subject_list:
            try:
                path = os.path.join(base_dir, f'subject-{subject}', f'{roi}.npy')
                if os.path.exists(path):
                    data = np.load(path)
                    
                    if approach == 'fr_rsa':
                        data = np.nan_to_num(data, nan=0.0)
                        
            except FileNotFoundError:
                continue

            subject_roi_corrs.append(data)
                
        aggregated_data[approach][roi] = np.array(subject_roi_corrs)

# --- Plotting 3x2 ---
print("\n>>> Building 3x2 Comparative Grid Figure <<<")
fig, axes = plt.subplots(3, 2, figsize=(16, 15), sharex=True)

for idx, area in enumerate(rois):
    # Calculate grid position coordinates
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # 1. Determine dynamic local Y-limits to optimize scale limits for this specific ROI panel
    local_max_y = 0.05
    for approach in ['classical', 'fr_rsa']:
        data = aggregated_data[approach][area]
        if len(data) > 0:
            m = np.mean(data, axis=0)
            s = sem(data, axis=0)
            local_max_y = max(local_max_y, np.max(m + s))
            
    # 2. Iterate and plot both approaches inside this subplot frame
    for app_idx, approach in enumerate(['classical', 'fr_rsa']):
        area_data = aggregated_data[approach][area]
        if len(area_data) == 0:
            continue
            
        n_subs = len(area_data)
        m_group = np.mean(area_data, axis=0)
        s_err = sem(area_data, axis=0)
        color = approach_colors[approach]
        
        # Cluster Permutation Test
        cluster_results = sign_permutation_cluster_test(area_data, n_permutations=10000)
        sig_mask = np.zeros(n_timepoints, dtype=bool)
        for c_idx, _, _ in cluster_results['significant_clusters']:
            sig_mask[c_idx] = True

        # Bootstrap Peak Latency
        boot_peaks = []
        for _ in range(n_bootstraps):
            res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
            boot_peaks.append(times[np.argmax(np.mean(area_data[res_idx], axis=0))])
        
        low, high = np.percentile(boot_peaks, [2.5, 97.5])
        obs_peak = times[np.argmax(m_group)]
        
        # Curve + Ribbon Plotting
        leg_text = f"{approach_labels[approach]}\nPeak: {obs_peak:.0f}ms [{low:.0f}-{high:.0f}ms]"
        ax.plot(times, m_group, color=color, lw=2.5, label=leg_text, zorder=3)
        ax.fill_between(times, m_group - s_err, m_group + s_err, color=color, alpha=0.1, zorder=2)
        
        # Peak Markers with Errorbars
        peak_val = np.max(m_group)
        ax.scatter(obs_peak, peak_val, color=color, s=60, edgecolors='white', zorder=5)
        ax.errorbar(obs_peak, peak_val, xerr=[[obs_peak-low], [high-obs_peak]], 
                    fmt='none', ecolor='k', elinewidth=1, capsize=3, zorder=4)
        
        # Significance Markers (Staggered vertically at the top of the axis)
        dot_y = local_max_y * (1.15 + (app_idx * 0.07)) 
        if np.any(sig_mask):
            ax.scatter(times[sig_mask], [dot_y] * np.sum(sig_mask), 
                       color=color, s=8, marker='s', alpha=0.8, edgecolors='none')

    # Subplot Clean Aesthetics
    ax.set_title(f'ROI: {area}', fontweight='bold', fontsize=14, pad=10)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=1, alpha=0.2)
    ax.set_xlim(-100, 600)
    ax.set_ylim(bottom=-0.01, top=local_max_y * 1.4)
    
    # Label Assignments based on Grid location to keep it uncluttered
    if col == 0:
        ax.set_ylabel("Spearman's R", fontsize=12)
    if row == 2:
        ax.set_xlabel('Time (ms)', fontsize=12)
        
    ax.legend(loc='upper right', frameon=False, fontsize=9.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Global Figure Layout configuration
plt.suptitle('Classical vs. Feature-Reweighted RSA', 
             fontweight='bold', fontsize=18, y=0.99)

plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, "classical_vs_reweighted_rsa.svg")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSuccess! Comparative 3x2 grid figure saved to: {save_path}")
print(f"Total Execution Time: {time.time() - start_time:.2f} seconds.")