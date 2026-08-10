"""
This script aggregates the ROI-wise EEG-fMRI encoding fusion results, computes statistics, and generates plots for each ROI.
The whole-brain plots are generated in a separate script, together with RSA whoole-brain results, to generate a single movie for both methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.stats import sem, t as t_dist
from utils import sign_permutation_cluster_test, get_eeg_times, get_roi_noise_ceiling_corr
from berg import BERG
import time

# Start time
start_time = time.time()

# --- Configuration ---
subject_list = [1, 4, 5, 6, 7, 8]

# Define the grouping: list of sub-ROIs for each integrated ROI group (e.g. V1 = V1v + V1d)
roi_groups = {
    'V1': ['V1v', 'V1d'],
    'V4': ['hV4'],
    'ventral': ['ventral']
}

# Ordered labels and explicit high-contrast colors for each integrated group
area_labels = ['V1', 'V4', 'ventral']

area_colors = [
    "#480758",  # V1 (Deep Purple)
    "#63a1cc",  # V4 (Steel Blue)
    "#8fd744",  # ventral (Light Green)
]

n_bootstraps = 10000

# Whole-brain noise-ceiling threshold -- same convention used by the RSA / other whole-brain
# scripts in this project (vertices below this ncsnr are excluded from the per-area average).
NCSNR_THRESHOLD = 0.2
N_VERT_PER_HEMI = 163842

# Pathing -- whole-brain per-vertex correlation time courses (same source the RSA whole-brain
# scripts read from), NOT the pre-averaged per-ROI files used previously.
base_results_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion/whole_brain'
PLOTS_DIR = '/scratch/jeffreykatab/Code/Encoding_Models/NSD/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- get EEG times, in ms, for plotting ---
times = get_eeg_times()

# --- Per-subject, per-sub-ROI vertex masks (ROI + noise-ceiling) ---
print(f">>> Building per-subject, per-sub-ROI masks (ROI + noise-ceiling >= {NCSNR_THRESHOLD}) <<<")
berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')

all_sub_rois = sorted(set(sub_roi for sub_rois in roi_groups.values() for sub_roi in sub_rois))

subject_masks = {}  # subject -> sub_roi -> boolean mask over concatenated [lh, rh] vertices
for subject in subject_list:
    metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)
    ncsnr_lh = metadata['fmri']['lh_ncsnr']
    ncsnr_rh = metadata['fmri']['rh_ncsnr']

    subject_masks[subject] = {}
    for sub_roi in all_sub_rois:
        roi_idx_lh = metadata['fmri']['lh_fsaverage_rois'][sub_roi]
        roi_mask_lh = np.zeros(N_VERT_PER_HEMI, dtype=bool)
        roi_mask_lh[roi_idx_lh] = True

        roi_idx_rh = metadata['fmri']['rh_fsaverage_rois'][sub_roi]
        roi_mask_rh = np.zeros(N_VERT_PER_HEMI, dtype=bool)
        roi_mask_rh[roi_idx_rh] = True

        combined_lh = roi_mask_lh & (ncsnr_lh >= NCSNR_THRESHOLD)
        combined_rh = roi_mask_rh & (ncsnr_rh >= NCSNR_THRESHOLD)
        subject_masks[subject][sub_roi] = np.concatenate([combined_lh, combined_rh])


# --- Data Aggregation ---
area_data_list = []  # Will store [Area][Subject, Time]

print(">>> Aggregating Whole-Brain Data (masking vertices per area, averaging across them) <<<")

for area in area_labels:
    sub_rois = roi_groups[area]
    subject_area_corrs = []

    for subject in subject_list:
        path_lh = os.path.join(base_results_dir, f'subject-{subject}', 'correlations_left.npy')
        path_rh = os.path.join(base_results_dir, f'subject-{subject}', 'correlations_right.npy')

        data_lh = np.load(path_lh)
        data_rh = np.load(path_rh)
        data_concat = np.concatenate([data_lh, data_rh], axis=1)  # (n_timepoints, n_vertices)

        # Union of this area's sub-ROI masks (e.g. V1v + V1d for 'V1'), each already
        # noise-ceiling-filtered.
        combined_mask = np.zeros(data_concat.shape[1], dtype=bool)
        for sub_roi in sub_rois:
            combined_mask |= subject_masks[subject][sub_roi]

        print(f"Loaded whole-brain correlations for Sub {subject}, Area {area} data shape = {data_concat.shape}")

        subject_area_corrs.append(np.mean(data_concat[:, combined_mask], axis=1))  # Averaging across vertices

    area_data_list.append(np.array(subject_area_corrs))


def ci95_across_subjects(area_data):
    """
    95% confidence interval of the across-subject mean at each timepoint, via the
    standard normal-theory formula (t-critical value * SEM). 

    This is a distinct quantity from the bootstrap CI computed further below for peak
    LATENCY (the timepoint at which the curve peaks): that one is a CI over a discrete
    time index, obtained by resampling subjects and re-finding the argmax each time.
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

    # Calculate y-limit based on data (using the 95% CI half-width, not SEM)
    all_means = [np.mean(d, axis=0) for d in data_list]
    all_cis = [ci95_across_subjects(d) for d in data_list]
    global_max_y = max([np.max(m + c) for m, c in zip(all_means, all_cis)])

    # Row spacing for the staggered significance lanes below y=0 (one lane per area)
    row_gap = global_max_y * 0.05

    print("\n>>> Peak latency (95% CI, bootstrap over subjects) per ROI <<<")

    for i, area_data in enumerate(data_list):
        n_subs = len(subject_list)
        m_group = np.mean(area_data, axis=0)
        ci_err = ci95_across_subjects(area_data)
        color = area_colors[i]

        # 1. Cluster Permutation Test
        cluster_results = sign_permutation_cluster_test(area_data, n_permutations=10000)
        sig_mask = np.zeros(n_timepoints, dtype=bool)
        for cluster_idx, _, _ in cluster_results['significant_clusters']:
            sig_mask[cluster_idx] = True

        # Significant time window: first and last significant timepoint, pooled across all
        if np.any(sig_mask):
            sig_times = times[sig_mask]
            print(f"{area_labels[i]}: significant from {sig_times.min():.0f} ms to {sig_times.max():.0f} ms")
        else:
            print(f"{area_labels[i]}: no significant time points")

        # 2. Bootstrap Peak Latency CI:
        boot_peaks = []
        for _ in range(n_bootstraps):
            res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
            boot_peaks.append(times[np.argmax(np.mean(area_data[res_idx], axis=0))])

        low, high = np.percentile(boot_peaks, [2.5, 97.5])
        obs_peak = times[np.argmax(m_group)]

        print(f"{area_labels[i]}: peak latency = {obs_peak:.0f} ms [95% CI: {low:.0f}-{high:.0f} ms]")

        # 3. Plotting
        leg_text = f"{area_labels[i]}: {obs_peak:.0f} ms [{low:.0f}-{high:.0f} ms]"

        ax.plot(times, m_group, color=color, lw=12.0, label=leg_text, zorder=3)
        ax.fill_between(times, m_group - ci_err, m_group + ci_err, color=color, alpha=0.20, zorder=2)

        # Peak Marker
        peak_val = np.max(m_group)
        ax.scatter(obs_peak, peak_val, color=color, s=600, edgecolors='white', zorder=5)
        ax.errorbar(obs_peak, peak_val, xerr=[[obs_peak-low], [high-obs_peak]],
                    fmt='none', ecolor='k', elinewidth=1, capsize=3, zorder=4)

        # Significance bars -- bars below the y=0 line (one bar per ROI)
        sig_y = -row_gap * (i + 1)
        if np.any(sig_mask):
            ax.scatter(times[sig_mask], [sig_y] * np.sum(sig_mask),
                       color=color, s=50, marker='s', alpha=0.8, edgecolors='none', zorder=3)


    # Styling
    #ax.set_title(f'{title}', fontweight='bold', fontsize=26, pad=40)
    #ax.set_xlabel('Time (ms)', fontsize=28)
    #ax.set_ylabel("Pearson's r", fontsize=28)
    ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=3, alpha=0.2)
    ax.set_xlim(-100, 600)
    bottom_limit = -row_gap * (len(area_labels) + 1.5)
    ax.set_ylim(bottom=bottom_limit, top=0.3)

    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=18)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.tick_params(axis='both', labelsize=26, width=3.0, length=12.0)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

# Run the plot
plot_roi_results(area_data_list, "ROI-wise Encoding Correlations", "roi_enc_eeg2fmri.svg")

print(f"Execution complete! Total Time: {time.time() - start_time:.2f} seconds.")