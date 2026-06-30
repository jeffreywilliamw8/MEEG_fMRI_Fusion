import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.stats import sem
from berg import BERG
from utils import sign_permutation_cluster_test, get_eeg_times
import time

# Start time
start_time = time.time()

# --- Configuration ---
subject_list = [1,4,5,6,7,8]

# Ordered labels and explicit high-contrast colors for each integrated group
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal']
stream_labels = ['Early', 'Midventral', 'Midlateral', 'Midparietal', 'Ventral', 'Lateral', 'Parietal']

stream_colors = [
    "#480758",  # early (Deep Purple)
    "#534998",  # midventral (Dark Blue)
    "#4380ac",  # midlateral (Steel Blue)
    "#398f66",  # midparietal (Dark Green)
    "#9ddb5a",  # ventral (Light Green)
    "#e9d730",  # lateral (Yellow)
    "#fd9f25"   # parietal (Orange)
]


n_bootstraps = 10000

# Path
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()


# --- Data Aggregation ---
stream_data_list = [] # Will store [Area][Subject, Time]

print(">>> Aggregating stream Data <<<")

for stream in streams:
    subject_stream_corrs = []
    
    for subject in subject_list:
        corrs_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion/whole_brain/subject-{subject}'
        data_dir_l = os.path.join(corrs_dir, 'correlations_left.npy')
        data_dir_r = os.path.join(corrs_dir, 'correlations_right.npy')
        data_lh = np.load(data_dir_l)
        data_rh = np.load(data_dir_r)

        berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')

        metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)

        stream_idx_lh = metadata['fmri']['lh_fsaverage_rois'][stream]
        stream_idx_rh = metadata['fmri']['rh_fsaverage_rois'][stream]


        stream_mask_lh = np.zeros(163842, dtype=bool)
        stream_mask_rh = np.zeros(163842, dtype=bool)

        stream_mask_lh[stream_idx_lh] = True
        stream_mask_rh[stream_idx_rh] = True


        wb_noise_ceilings_lh = metadata['fmri']['lh_ncsnr']
        wb_noise_ceilings_rh = metadata['fmri']['rh_ncsnr']

        nc_mask_lh = np.zeros(163842, dtype=bool)
        nc_mask_rh = np.zeros(163842, dtype=bool)


        nc_idx_lh = np.where(wb_noise_ceilings_lh >= 0.2)[0] # Selecting vertices above noise ceiling threshold of 20%
        nc_mask_lh[nc_idx_lh] = True
        valid_vertices_lh = np.where(stream_mask_lh & nc_mask_lh)[0] # Selecting vertices above noise ceiling threshold of 20%

        nc_idx_rh = np.where(wb_noise_ceilings_rh >= 0.2)[0] # Selecting vertices above noise ceiling threshold of 20%
        nc_mask_rh[nc_idx_rh] = True
        valid_vertices_rh = np.where(stream_mask_rh & nc_mask_rh)[0] # Selecting vertices above noise ceiling threshold of 20%
        
        data_lh = data_lh[:, valid_vertices_lh]
        data_rh = data_rh[:, valid_vertices_rh]

        print(f"Loaded correlations for Sub {subject}, Stream {stream}: LH shape = {data_lh.shape}, RH shape = {data_rh.shape}")
        stream_data = np.concatenate([data_lh, data_rh], axis=1)
        
        subject_stream_corrs.append(np.mean(stream_data, axis=1)) # averaging across vertices
        


    stream_data_list.append(np.array(subject_stream_corrs))

stream_curves = {
    'early': stream_data_list[0],
    'midventral': stream_data_list[1],
    'midlateral': stream_data_list[2],
    'midparietal': stream_data_list[3],
    'ventral': stream_data_list[4],
    'lateral': stream_data_list[5],
    'dorsal': stream_data_list[6]
}
np.save(os.path.join(PLOTS_DIR, 'em_stream_curves.npy'), stream_curves)

# --- Plotting & Stats ---
def plot_stream_results(data_list, title, filename):
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
        color = stream_colors[i]
        
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
        leg_text = f"{streams[i]}: {obs_peak:.0f}ms [{low:.0f}-{high:.0f}ms]"
        
        ax.plot(times, m_group, color=color, lw=2.5, label=leg_text, zorder=3)
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

    # Styling
    ax.set_title(f'{title}', fontweight='bold', fontsize=18, pad=40)
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel("Spearman's R", fontsize=14)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=1, alpha=0.2)
    ax.set_xlim(-100, 600)
    ax.set_ylim(bottom=-0.01, top=0.23)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=11)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

# Run the plot
plot_stream_results(stream_data_list, "Encoding Correlations", "stream_encoding_fusion.svg")

print(f"Execution complete! Total Time: {time.time() - start_time:.2f} seconds.")