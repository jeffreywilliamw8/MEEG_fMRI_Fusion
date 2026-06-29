import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.stats import sem
from berg import BERG
from utils import sign_permutation_cluster_test
import time

# Start time
start_time = time.time()

# --- Configuration ---
subject_list = [1, 4, 5, 6, 7, 8]
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal']
stream_labels = ['Early', 'Midventral', 'Midlateral', 'Midparietal', 'Ventral', 'Lateral', 'Parietal']
stream_colors = ["#480758", "#5572bb", "#38bdb6", "#2ec25a", "#cefd25", "#fd9f25", "#fd4d25"]
n_bootstraps = 10000

# Pathing (Adjust base_dir as needed)
base_results_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion/streamwise'
PLOTS_DIR = '/scratch/jeffreykatab/Code/Encoding_Models/NSD/plots/encoding_fusion/streamwise'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
n_times = 615
times_raw = np.round(np.linspace(-200, 1000, n_times)).astype(int)
times = times_raw - 50 # Account for 50ms shift

# Selection mask for -100ms to 600ms
mask = (times >= -100) & (times <= 600)
times = times[mask]
t_start_idx = np.where(mask)[0][0]
t_end_idx = np.where(mask)[0][-1]

# --- Data Aggregation ---
stream_data_list = [] # Store [Stream][Subject, Time]

# Load the fMRI metadata
berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')

print(">>> Aggregating EEG-fMRI Regression Data (Pooled Hemispheres) <<<")
for stream in tqdm(streams, desc="Streams"):
    subject_corrs = []
    for subject in subject_list:
        try:
            # Loading hemispheres
            path_lh = os.path.join(base_results_dir, f'subject-{subject}', f'{stream}_lh.npy')
            path_rh = os.path.join(base_results_dir, f'subject-{subject}', f'{stream}_rh.npy')

            corrs_lh = np.load(path_lh)
            corrs_rh = np.load(path_rh)

            # Load the fMRI metadata
            metadata_fmri = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)

            # Selecting indices and noise ceilings
            stream_idx_lh = metadata_fmri['fmri']['lh_fsaverage_rois'][stream]
            stream_idx_rh = metadata_fmri['fmri']['rh_fsaverage_rois'][stream]
            stream_mask_lh = np.zeros(163842, dtype=bool)
            stream_mask_lh[stream_idx_lh] = True
            stream_mask_rh = np.zeros(163842, dtype=bool)
            stream_mask_rh[stream_idx_rh] = True


            
            wb_nc_lh = metadata_fmri['fmri']['lh_ncsnr']
            wb_nc_rh = metadata_fmri['fmri']['rh_ncsnr']
            nc_mask_lh = np.zeros(163842, dtype=bool)
            #nc_idx_lh

            valid_v_lh = np.where(wb_nc_lh[stream_idx_lh] >= 0.2)[0]
            valid_v_rh = np.where(wb_nc_rh[stream_idx_rh] >= 0.2)[0]

            # 1. Extract valid vertices for each hemisphere
            # Resulting shapes: (Time, N_valid_vertices)
            valid_corrs_lh = corrs_lh[:, valid_v_lh]
            valid_corrs_rh = corrs_rh[:, valid_v_rh]
            print("Shape of valid vertices (LH, RH):", valid_corrs_lh.shape, valid_corrs_rh.shape)
            

            # 2. CONCATENATE across hemispheres (axis 1 is the vertex dimension)
            # Resulting shape: (Time, N_total_valid_vertices)
            combined_vertices = np.concatenate([valid_corrs_lh, valid_corrs_rh], axis=1)

            # 3. AVERAGE across the combined vertex pool
            # Resulting shape: (Time,)
            subject_stream_mean = np.mean(combined_vertices, axis=1)

            # Slice to match your time mask (-100 to 600ms)
            subject_corrs.append(subject_stream_mean)

        except FileNotFoundError:
            print(f"Warning: Data missing for Sub {subject}, Stream {stream}")
            continue
            
    stream_data_list.append(np.array(subject_corrs))

# --- Plotting & Stats ---
def plot_eeg_fmri_results(data_list, title, filename):
    n_timepoints = len(times)
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    # We'll calculate a local max for better dot placement
    all_means = [np.mean(d, axis=0) for d in data_list]
    all_sems = [sem(d, axis=0) for d in data_list]
    global_max_y = max([np.max(m + s) for m, s in zip(all_means, all_sems)])
    
    global_max_y = 0.25

    for i, stream_data in enumerate(data_list):
        n_subs = stream_data.shape[0]
        m_group = np.mean(stream_data, axis=0)
        s_err = sem(stream_data, axis=0)
        color = stream_colors[i]
        
        # 1. Cluster Permutation Test
        cluster_results = sign_permutation_cluster_test(stream_data, n_permutations=10000)
        sig_mask = np.zeros(n_timepoints, dtype=bool)
        for cluster_idx, _, _ in cluster_results['significant_clusters']:
            sig_mask[cluster_idx] = True

        # 2. Bootstrap Peak Latency
        boot_peaks = []
        for _ in range(n_bootstraps):
            res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
            boot_peaks.append(times[np.argmax(np.mean(stream_data[res_idx], axis=0))])
        
        low, high = np.percentile(boot_peaks, [2.5, 97.5])
        obs_peak = times[np.argmax(m_group)]
        
        # 3. Plotting
        leg_text = f"{stream_labels[i]}: {obs_peak:.0f}ms [{low:.0f}-{high:.0f}ms]"
        
        ax.plot(times, m_group, color=color, lw=2.5, label=leg_text, zorder=3)
        ax.fill_between(times, m_group - s_err, m_group + s_err, color=color, alpha=0.1, zorder=2)
        
        # Peak Marker
        peak_val = np.max(m_group)
        ax.scatter(obs_peak, peak_val, color=color, s=60, edgecolors='white', zorder=5)
        ax.errorbar(obs_peak, peak_val, xerr=[[obs_peak-low], [high-obs_peak]], 
                    fmt='none', ecolor='k', elinewidth=1, capsize=3, zorder=4)
        
        # Significance Dots (Staggered at top)
        dot_y = global_max_y * (1.1 + (i * 0.04)) 
        if np.any(sig_mask):
            ax.scatter(times[sig_mask], [dot_y] * np.sum(sig_mask), 
                       color=color, s=8, marker='s', alpha=0.8, edgecolors='none')

    # Final Polish
    ax.set_title(f'EEG-fMRI Fusion: {title}', fontweight='bold', fontsize=18, pad=40)
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel("Pearson's r", fontsize=14)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=1, alpha=0.2)
    ax.set_xlim(-100, 600)
    ax.set_ylim(bottom=-0.02, top=global_max_y * 1.5)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=11)
    
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

# Run the single plot
plot_eeg_fmri_results(stream_data_list, "Encoding Correlations", "enc_eeg2fmri_concat.png")

print(f"Execution complete! Time: {time.time() - start_time:.2f}s")