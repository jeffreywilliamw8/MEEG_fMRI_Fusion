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

# Ordered labels and explicit high-contrast colors for each integrated group
rois = ['V1', 'V2', 'V3', 'hV4', 'ventral']
area_colors = [
    "#480758",  # V1 (Deep Purple)
    "#5b4fab",  # V2 (Dark Blue)
    "#63a1cc",  # V3 (Steel Blue)
    "#8fd744",  # V4 (Light Green)
    "#fd9f25"   # ventral (Orange)
]

n_bootstraps = 10000

# Pathing
base_results_dir = os.path.join(
    '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/',
    'correlations',
    'roi',
    f'eeg_rdm_metric-correlation',
    f'fmri_rdm_metric-correlation',
)
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots/roi_rsa'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()


# --- Data Aggregation ---
roi_data_list = [] # Will store [Area][Subject, Time]

print(">>> Aggregating ROI Data (Averaging sub-ROIs) <<<")

for roi in rois:
    subject_roi_corrs = []
    
    for subject in subject_list:
        try:
            # Direct load of pre-averaged npy files
            path = os.path.join(base_results_dir, f'subject-{subject}', f'{roi}.npy')

            data = np.load(path)

            print(f"Loaded correlations for Sub {subject}, ROI {roi}: shape = {data.shape}")
            
            subject_roi_corrs.append(data)

        except FileNotFoundError:
                continue
            
    roi_data_list.append(np.array(subject_roi_corrs))

# --- Plotting & Stats ---
def plot_roi_results(data_list, title, filename):
    n_timepoints = len(times)
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    # Calculate y-limit based on data
    all_means = [np.mean(d, axis=0) for d in data_list]
    all_sems = [sem(d, axis=0) for d in data_list]
    global_max_y = max([np.max(m + s) for m, s in zip(all_means, all_sems)])

    for i, area_data in enumerate(data_list):
        n_subs = len(subject_list)
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
        
        # 3. Plotting
        leg_text = f"{rois[i]}: {obs_peak:.0f}ms [{low:.0f}-{high:.0f}ms]"
        
        ax.plot(times, m_group, color=color, lw=5, label=leg_text, zorder=3)
        ax.fill_between(times, m_group - s_err, m_group + s_err, color=color, alpha=0.1, zorder=2)
        
        # Peak Marker
        peak_val = np.max(m_group)
        ax.scatter(obs_peak, peak_val, color=color, s=60, edgecolors='white', zorder=5)
        ax.errorbar(obs_peak, peak_val, xerr=[[obs_peak-low], [high-obs_peak]], 
                    fmt='none', ecolor='k', elinewidth=1, capsize=3, zorder=4)
        
        # Significance Dots (Staggered at top)
        dot_y = global_max_y * (1.1 + (i * 0.05)) 
        if np.any(sig_mask):
            ax.scatter(times[sig_mask], [dot_y] * np.sum(sig_mask), 
                       color=color, s=8, marker='s', alpha=0.8, edgecolors='none')
        onsets = times[sig_mask]
        print(f"ROI:{rois[i]}; Onset: {onsets[0]} ms")

    # Styling
    #ax.set_title('', fontweight='bold', fontsize=18, pad=40)
    #ax.set_xlabel('Time (ms)', fontsize=26)
    #ax.set_ylabel("Spearman's R", fontsize=26)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=2, alpha=0.2)
    ax.set_xlim(-100, 600)
    ax.set_ylim(bottom=-0.01, top=0.175)
    
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=26)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=26)
    
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

# Run the plot
plot_roi_results(roi_data_list, "RSA Correlations", "roi_rsa_fusion.svg")

print(f"Execution complete! Total Time: {time.time() - start_time:.2f} seconds.")