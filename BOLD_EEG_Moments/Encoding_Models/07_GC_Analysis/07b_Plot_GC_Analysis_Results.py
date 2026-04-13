import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import sem
import time
from utils import load_eeg_times, sign_permutation_cluster_test
from tqdm import tqdm

# Start time
start_time = time.time()

# --- 1. CONFIGURATION ---
subject_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
eeg_frequency = 100
roi_pairs = ["V1-V2", "V1-V3", "V1-V4", "V1-FFA", "V1-OFA", "V1-EBA", "V1-PPA", "V1-LOC", "V2-V3", 
             "V2-V4", "V2-FFA", "V2-OFA", "V2-EBA", "V2-PPA", "V2-LOC", "V3-V4", "V3-FFA", "V3-OFA", 
             "V3-EBA", "V3-PPA", "V3-LOC", "V4-FFA", "V4-OFA", "V4-EBA", "V4-PPA", "V4-LOC", "FFA-OFA", 
             "FFA-EBA", "FFA-PPA", "FFA-LOC", "OFA-EBA", "OFA-PPA", "OFA-LOC", "EBA-PPA", "EBA-LOC", "PPA-LOC"] # 9 ROIs => 36 unique pairs

PLOTS_DIR = '/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/plots/granger_causality/crop_500ms'
os.makedirs(PLOTS_DIR, exist_ok=True)

full_times = 1000 * load_eeg_times(eeg_frequency) 
t_plot = full_times[12:341]
n_times = len(t_plot)

colors = {'feedforward': "#3dad8a", 'feedback': "#ebcd08"}
n_permutations = 10000 

# =============================================================================
# 1. STATS
# =============================================================================
def run_stats(data_matrix, baseline_means, n_perms=10000):
    """
    Adapts the custom sign_permutation_cluster_test for the plotting logic.
    
    data_matrix: (n_subjects, n_timepoints)
    baseline_means: (n_subjects,) 
    """
    if data_matrix.shape <= 1:
        return np.zeros(data_matrix.shape, dtype=bool)
    
    # 1. Center the data so that '0' represents the baseline level
    # Your function tests for deviations from zero via sign-flipping.
    centered_data = data_matrix - baseline_means[:, np.newaxis]
    
    # 2. Call your custom function
    # Note: adjusting p_thresh (pointwise) and alpha (cluster-level) as needed
    stats_results = sign_permutation_cluster_test(
        centered_data, 
        n_permutations=n_perms, 
        p_thresh=0.01, 
        alpha=0.05
    )
    
    # 3. Reconstruct the boolean mask for the plotting logic
    # The plotting logic expects a boolean array of shape (n_times,)
    sig_mask = np.zeros(data_matrix.shape, dtype=bool)
    
    # Extract indices from significant clusters
    for cluster_info in stats_results['significant_clusters']:
        indices = cluster_info # The 'idx' array from your function
        sig_mask[indices] = True
        
    return sig_mask

# =============================================================================
# 2. PLOTTING
# =============================================================================
cv_ta_dict = {
    "cv": "True",
    "ncv": "False",
    "ta": "True",
    "nta": "False"
}
for roi_pair in tqdm(roi_pairs):
    for cross_validate in ['cv', 'ncv']:
        for time_averaged in ['ta', 'nta']:

            plt.figure(figsize=(12, 7))
            ax = plt.gca()
            area_a, area_b = roi_pair.split('-')
            
            data_fwd, data_bwd = [], []
            base_fwd_means, base_bwd_means = [], []
            
            for sub in subject_list:
                data_dir = f'/home/jeffreykatab/Projects/fusion/Bold_EEG_Moments/Encoding_Models/results/granger_causality_analysis/phase_1/{cross_validate}/fmri_sub-{sub}'
                res_path = os.path.join(data_dir, f'{roi_pair}_{time_averaged}.npy')
                
                if os.path.exists(res_path):
                    res = np.load(res_path, allow_pickle=True).item()
                    base_fwd_means.append(np.mean(res['baseline_ff']))
                    base_bwd_means.append(np.mean(res['baseline_fb']))
                    data_fwd.append(list(res['baseline_ff']) + list(res['gc_ff'])) 
                    data_bwd.append(list(res['baseline_fb']) + list(res['gc_fb']))

            if not data_fwd:
                continue

            fwd_arr, bwd_arr = np.array(data_fwd), np.array(data_bwd)
            fwd_baselines, bwd_baselines = np.array(base_fwd_means), np.array(base_bwd_means)
            
            combined_mean_max = max(np.max(np.mean(fwd_arr, axis=0)+sem(fwd_arr, axis=0)), np.max(np.mean(bwd_arr, axis=0)+sem(bwd_arr, axis=0)))
            dot_base_y = combined_mean_max * 1.15
            dot_spacing = combined_mean_max * 0.08
            
            lines = [] # To store line objects for the legend

            for d_idx, (label, d_data, d_baselines, d_color) in enumerate([
                (f'{area_a} → {area_b}', fwd_arr, fwd_baselines, colors['feedforward']),
                (f'{area_b} → {area_a}', bwd_arr, bwd_baselines, colors['feedback'])
            ]):
                m = np.mean(d_data, axis=0)
                s = sem(d_data, axis=0)
                
                # 1. Plot Curve and Shading
                ln, = ax.plot(t_plot, m, color=d_color, lw=2.5, zorder=3)
                lines.append(ln) 
                ax.fill_between(t_plot, m-s, m+s, color=d_color, alpha=0.2, zorder=2)
                
                # 2. Call Stats Function

                stats = sign_permutation_cluster_test(d_data - d_baselines[:, np.newaxis],n_permutations=n_permutations, p_thresh=0.01, alpha=0.05)

                # Build significance mask
                sig_mask = np.zeros(n_times, dtype=bool)
                for idx, _, _ in stats["significant_clusters"]:
                    sig_mask[idx] = True

                # 3. Plot Dots
                y_pos = dot_base_y + (d_idx * dot_spacing)
                ax.scatter(t_plot[sig_mask], np.full(np.sum(sig_mask), y_pos), color=d_color, s=15, zorder=4)
                #ax.scatter(t_plot[~sig_mask], np.full(np.sum(~sig_mask), y_pos), color='lightgrey', s=10, alpha=0.4, zorder=1)

            # Styling
            # Increase 'pad' to make room for the legend between title and plot
            ax.set_title(f'Granger Causality: {area_a} ↔ {area_b} | Cross-Validated: {cv_ta_dict[cross_validate]} | Time-Averaged RDMs: {cv_ta_dict[time_averaged]}', fontweight='bold', pad=50) 
            ax.set_ylabel('GC Influence', fontsize=12)
            ax.set_xlabel('Time (ms)')
            ax.set_xlim(-80, 500)
            
            ax.axhline(0, color='black', lw=1, ls='--', alpha=0.3)
            ax.axvline(0, color='black', lw=1.5, alpha=0.8)

            # Modified Legend Logic
            # bbox_to_anchor=(0.5, 1.1) puts it above the axis; ncol=2 makes it horizontal
            ax.legend(handles=lines, labels=[f'{area_a} → {area_b}', f'{area_b} → {area_a}'], 
                    loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.08), 
                    frameon=False, fontsize=11)
            
            
            plt.tight_layout()
            file_name = f'gc_1_{roi_pair}_{cross_validate}_{time_averaged}_z.png'
            if not os.path.exists(os.path.join(PLOTS_DIR, roi_pair)):
                os.makedirs(os.path.join(PLOTS_DIR, roi_pair))
            plt.savefig(os.path.join(PLOTS_DIR, roi_pair, file_name), dpi=300)
            plt.close()
            print(f"Created: {file_name}")

print(f"Plots saved in: {PLOTS_DIR}")
print(f"Execution complete! Time: {time.time() - start_time:.2f} seconds.")