"""
This script plots the searchlight RSA fusion results for specific ROIs (V1, V4, ventral) across time, averaged across subjects.
It also performs statistical analyses, including cluster permutation tests and bootstrap confidence intervals for peak latencies.
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.stats import sem, t as t_dist
from utils import sign_permutation_cluster_test, get_eeg_times
from berg import BERG
import time


# Start time
start_time = time.time()

# --- Configuration ---
subject_list = [1,4,5,6,7,8]

# Define the grouping: Area -> list of sub-ROI filenames
# Define the grouping hierarchy: Group Label -> List of sub-ROI file names
roi_groups = {
    'V1': ['V1v', 'V1d'],
    'V4': ['hV4'],
    'ventral': ['ventral']
}

# Ordered labels and explicit high-contrast colors for each integrated group
area_labels = ['V1', 'V4', 'ventral']
area_colors = [
    "#480758",  # V1 (Deep Purple)
    "#63a1cc",  # hV4 (Steel Blue)
    "#8fd744",  # ventral (Light Green)
]


n_bootstraps = 10000
N_NEIGHBOURS = 100

base_results_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/searchlight_fusion/eeg_rdm_metric-pearsonr/n_neighbours-{N_NEIGHBOURS}/aggregated_results'
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()

# Helper function to select vertices based on mask and noise ceiling threshold
def select_vertices(data, mask, noise_ceilings, threshold=0.2):
    # Selecting vertices from the whole-brain surface based on 2 conditions: belonging to the ROI and noise ceiling above the threshold
    # Condition 1: Non-zero values of the mask (selecting vertices belonging to the ROI)
    condition1 = mask != 0

    # Condition 2: Vertices with a noise ceiling greater than the threshold
    condition2 = noise_ceilings >= threshold

    # Combined conditions
    combined_condition = condition1 & condition2

    # Selecting the vertices that satisfy both conditions

    return data[:,combined_condition]

# --- Data Aggregation ---
area_data_list = [] # Will store [Area][Subject, Time]

print(">>> Aggregating ROI Data (Averaging sub-ROIs) <<<")

for area in area_labels:
    sub_rois = roi_groups[area]
    subject_area_corrs = []

    for subject in subject_list:
        sub_roi_corrs = []
        for sr, sub_roi in enumerate(sub_rois):
            try:
                # Direct load of pre-averaged npy files
                path_lh = os.path.join(base_results_dir, f'subject-{subject}', f'subject-{subject}_lh_hemisphere_timecourse.npy')
                path_rh = os.path.join(base_results_dir, f'subject-{subject}', f'subject-{subject}_rh_hemisphere_timecourse.npy')

                data_lh = np.load(path_lh)
                data_rh = np.load(path_rh)

                data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
                berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
                metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)
                # Available ROIS:
                # dict_keys(['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FBA-1', 'FBA-2', 'mTL-bodies', 'OFA', 'FFA-1',
                #  'FFA-2', 'mTL-faces', 'aTL-faces', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words', 'early',
                # 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal', 'nsdgeneral'])
                # Selecting the ROI indices
                roi_idx_lh = metadata['fmri']['lh_fsaverage_rois'][sub_roi]
                roi_mask_lh = np.zeros(163842, dtype=bool)
                roi_mask_lh[roi_idx_lh] = True

                roi_idx_rh = metadata['fmri']['rh_fsaverage_rois'][sub_roi]
                roi_mask_rh = np.zeros(163842, dtype=bool)
                roi_mask_rh[roi_idx_rh] = True

                wb_noise_ceilings_lh = metadata['fmri']['lh_ncsnr']
                wb_noise_ceilings_rh = metadata['fmri']['rh_ncsnr']

                roi_corrs_left = select_vertices(data_lh, roi_mask_lh, wb_noise_ceilings_lh)
                roi_corrs_right = select_vertices(data_rh, roi_mask_rh, wb_noise_ceilings_rh)

                data_concat = np.concatenate([roi_corrs_left, roi_corrs_right], axis=1)

                if sr == 0:
                   sub_roi_corrs = data_concat
                else:
                    sub_roi_corrs = np.concatenate([sub_roi_corrs, data_concat], axis=1)

            except FileNotFoundError:
                    continue


        subject_area_corrs.append(np.mean(sub_roi_corrs, axis=1)) # Averaging across vertices
        #subject_area_corrs.append(sub_roi_corrs[:, 8])  # Selecting vertex 8

    area_data_list.append(np.array(subject_area_corrs))


def ci95_across_subjects(area_data):
    """
    95% confidence interval of the across-subject mean at each timepoint, via the
    standard normal-theory formula (t-critical value * SEM). 

    """
    n_subs = area_data.shape[0]
    s_err = sem(area_data, axis=0)
    t_crit = t_dist.ppf(0.975, df=n_subs - 1)
    return s_err * t_crit


# --- Plotting & Stats ---
def plot_roi_results(data_list, title, filename):
    n_timepoints = len(times)
    plt.figure(figsize=(14, 8))
    ax = plt.gca()

    # Calculate y-limit based on data 
    all_means = [np.mean(d, axis=0) for d in data_list]
    all_cis = [ci95_across_subjects(d) for d in data_list]
    global_max_y = max([np.max(m + c) for m, c in zip(all_means, all_cis)])

    # Row spacing for the significance bars below y=0 (one lane per area)
    row_gap = global_max_y * 0.05

    print("\n>>> Peak latency (95% CI, bootstrap over subjects) per ROI <<<")

    for i, area_data in enumerate(data_list):
        n_subs = len(subject_list)
        m_group = np.mean(area_data, axis=0)
        ci_err = ci95_across_subjects(area_data)  #  CI: uncertainty in the correlation value itself
        color = area_colors[i]

        # 1. Cluster Permutation Test
        cluster_results = sign_permutation_cluster_test(area_data, n_permutations=10000)
        sig_mask = np.zeros(n_timepoints, dtype=bool)
        for cluster_idx, _, _ in cluster_results['significant_clusters']:
            sig_mask[cluster_idx] = True

        if np.any(sig_mask):
            sig_times = times[sig_mask]
            print(f"{area_labels[i]}: significant from {sig_times.min():.0f}ms to {sig_times.max():.0f}ms")
        else:
            print(f"{area_labels[i]}: no significant time points")

        # 2. Bootstrap Peak Latency CI
        boot_peaks = []
        for _ in range(n_bootstraps):
            res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
            boot_peaks.append(times[np.argmax(np.mean(area_data[res_idx], axis=0))])

        low, high = np.percentile(boot_peaks, [2.5, 97.5])
        obs_peak = times[np.argmax(m_group)]

        print(f"{area_labels[i]}: peak latency = {obs_peak:.0f}ms [95% CI: {low:.0f}-{high:.0f}ms]")

        # 3. Plotting
        leg_text = f"{area_labels[i]}: {obs_peak:.0f}ms [{low:.0f}-{high:.0f}ms]"

        ax.plot(times, m_group, color=color, lw=12.0, label=leg_text, zorder=3)
        ax.fill_between(times, m_group - ci_err, m_group + ci_err, color=color, alpha=0.20, zorder=2)

        # Peak Marker
        peak_val = np.max(m_group)
        ax.scatter(obs_peak, peak_val, color=color, s=600, edgecolors='white', zorder=5)
        ax.errorbar(obs_peak, peak_val, xerr=[[obs_peak-low], [high-obs_peak]], fmt='none', ecolor='k', elinewidth=1, capsize=3, zorder=4)

        # Significance Dots bars below the y=0 line (one bar per ROI)
        sig_y = -row_gap * (i + 1)
        if np.any(sig_mask):
            ax.scatter(times[sig_mask], [sig_y] * np.sum(sig_mask),
                       color=color, s=50, marker='s', alpha=0.8, edgecolors='none', zorder=3)

    # Styling
    #ax.set_title(f'{title}', fontweight='bold', fontsize=18, pad=40)
    #ax.set_xlabel('Time (ms)', fontsize=26)
    #ax.set_ylabel("Spearman's R", fontsize=26)
    ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=3, alpha=0.2)
    ax.set_xlim(-100, 600)
    bottom_limit = -row_gap * (len(area_labels) + 1.5)
    ax.set_ylim(bottom=bottom_limit, top=0.035)

    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.tick_params(axis='both', labelsize=26, width=3.0, length=14.0)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

# Run the plot
plot_roi_results(area_data_list, "RSA Correlations", "roi_rsa_fusion.svg")

print(f"Execution complete! Total Time: {time.time() - start_time:.2f}s")