"""
This script plots group-average RSA fusion time courses (Pearson-r EEG RDM), one subplot per searchlight
neighbourhood size k (2x3 grid), each subplot showing all three ROIs (V1, V4, ventral), 
For six values of k (1, 10, 50, 100, 200, 775).
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from berg import BERG

from utils import get_eeg_times, sign_permutation_cluster_test

start_time = time.time()

# --- Configuration ---
subject_list = [1, 4, 5, 6, 7, 8]
eeg_rdm_metric = 'pearsonr'
k_values = [1, 50, 50, 100, 200, 775]  # fills the 2x3 grid exactly, one panel per k

roi_groups = {
    'V1': ['V1v', 'V1d'],
    'V4': ['hV4'],
    'ventral': ['ventral'],
}
area_labels = ['V1', 'V4', 'ventral']
area_colors = ["#480758", "#468fc3", "#ea8e16"]  # V1, V4, ventral

n_bootstraps = 10000

PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

times = get_eeg_times()
n_timepoints = len(times)

berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')


def select_vertices(data, mask, noise_ceilings, threshold=0.2):
    """Select vertices belonging to the ROI (mask) AND passing the noise-ceiling threshold."""
    condition1 = mask != 0
    condition2 = noise_ceilings >= threshold
    combined_condition = condition1 & condition2
    return data[:, combined_condition]


def get_subject_dir(k, subject):
    """
    Resolves the per-subject results directory for a given k. k=1 (univariate) lives under a
    different results tree than the true searchlights, and its directory name is literally
    'eeg_rdm_metric-{metric}subject-{subject}' (no separator between metric and 'subject-') --
    reproduced here exactly as written by the script that saves it.
    """
    if k == 1:
        return (f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/'
                f'univariate_rsa/eeg_rdm_metric-{eeg_rdm_metric}/subject-{subject}')
    else:
        return os.path.join(
            f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/'
            f'searchlight_fusion/eeg_rdm_metric-{eeg_rdm_metric}/n_neighbours-{k}/aggregated_results',
            f'subject-{subject}'
        )


def ci95_across_subjects(area_data, n_bootstraps=n_bootstraps):
    """
    95% confidence interval of the across-subject mean at each timepoint, via percentile
    bootstrap: resample subjects with replacement n_bootstraps times, recompute the mean across
    the resampled subjects at each timepoint, then take the 2.5th/97.5th percentiles of that
    bootstrap distribution.
    """
    n_subs = area_data.shape[0]
    boot_means = np.zeros((n_bootstraps, area_data.shape[1]))
    for i in range(n_bootstraps):
        res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
        boot_means[i] = np.mean(area_data[res_idx], axis=0)
    low, high = np.percentile(boot_means, [2.5, 97.5], axis=0)
    return low, high


# =============================================================================
# Data aggregation: aggregated[k][area] -> (n_subjects, n_time)
# =============================================================================
print(">>> Aggregating ROI-averaged RSA fusion time courses across k values <<<")

aggregated = {k: {area: [] for area in area_labels} for k in k_values}

for k in k_values:
    for subject in tqdm(subject_list, desc=f'k={k}'):
        subject_dir = get_subject_dir(k, subject)
        try:
            data_lh = np.load(os.path.join(
                subject_dir, f'subject-{subject}_lh_hemisphere_timecourse.npy'))
            data_rh = np.load(os.path.join(
                subject_dir, f'subject-{subject}_rh_hemisphere_timecourse.npy'))
        except FileNotFoundError:
            print(f"  Missing data for k={k}, subject {subject}")
            continue

        metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)
        wb_noise_ceilings_lh = metadata['fmri']['lh_ncsnr']
        wb_noise_ceilings_rh = metadata['fmri']['rh_ncsnr']

        for area in area_labels:
            sub_rois = roi_groups[area]
            sub_roi_corrs = None

            for sr, sub_roi in enumerate(sub_rois):
                roi_idx_lh = metadata['fmri']['lh_fsaverage_rois'][sub_roi]
                roi_mask_lh = np.zeros(163842, dtype=bool)
                roi_mask_lh[roi_idx_lh] = True

                roi_idx_rh = metadata['fmri']['rh_fsaverage_rois'][sub_roi]
                roi_mask_rh = np.zeros(163842, dtype=bool)
                roi_mask_rh[roi_idx_rh] = True

                roi_corrs_left = select_vertices(data_lh, roi_mask_lh, wb_noise_ceilings_lh)
                roi_corrs_right = select_vertices(data_rh, roi_mask_rh, wb_noise_ceilings_rh)
                data_concat = np.concatenate([roi_corrs_left, roi_corrs_right], axis=1)

                sub_roi_corrs = data_concat if sr == 0 else np.concatenate(
                    [sub_roi_corrs, data_concat], axis=1)

            if sub_roi_corrs is not None:
                aggregated[k][area].append(np.mean(sub_roi_corrs, axis=1))  # avg across vertices

for k in k_values:
    for area in area_labels:
        aggregated[k][area] = np.array(aggregated[k][area])  # (n_subjects, n_time)

# =============================================================================
# Plot: 2x3 grid, one subplot per k, 3 ROI curves per subplot (group-average + 95% CI ribbon +
# peak marker + peak-latency-with-CI legend). All six k values fill the grid exactly.
# =============================================================================
print("\nPlotting group-average RSA fusion time courses, one subplot per k...")

fig, axes = plt.subplots(2, 3, figsize=(21, 12), sharex=True, sharey=False)
axes = axes.reshape(-1)

for panel_idx, k in enumerate(k_values):
    ax = axes[panel_idx]

    print(f"\n>>> k={k} <<<")

    # 1. Compute all per-ROI stats once (mean, CI ribbon, cluster significance, peak + CI),
    # before touching the axes -- this panel's own row_gap/y-scaling (below) depends on them.
    area_stats = {}
    for a_idx, area in enumerate(area_labels):
        area_data = aggregated[k][area]
        if len(area_data) == 0:
            continue

        n_subs = area_data.shape[0]
        m_group = np.mean(area_data, axis=0)
        ci_low, ci_high = ci95_across_subjects(area_data)

        # Cluster-based sign-permutation test: timepoints where the group-average correlation
        # is significantly different from zero, cluster-corrected across time.
        cluster_results = sign_permutation_cluster_test(area_data, n_permutations=10000)
        sig_mask = np.zeros(n_timepoints, dtype=bool)
        for cluster_idx, _, _ in cluster_results['significant_clusters']:
            sig_mask[cluster_idx] = True

        # Bootstrap peak latency CI: resample subjects with replacement, re-find the argmax of
        # the resampled group mean each time -- uncertainty in the peak's TIME INDEX, distinct
        # from the ribbon CI above (which is about the correlation VALUE at each timepoint).
        boot_peaks = []
        for _ in range(n_bootstraps):
            res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
            boot_peaks.append(times[np.argmax(np.mean(area_data[res_idx], axis=0))])
        low, high = np.percentile(boot_peaks, [2.5, 97.5])
        obs_peak = times[np.argmax(m_group)]
        peak_val = np.max(m_group)

        print(f"  {area}: peak latency = {obs_peak:.0f}ms [95% CI: {low:.0f}-{high:.0f}ms]")
        if np.any(sig_mask):
            sig_times = times[sig_mask]
            print(f"  {area}: significant from {sig_times.min():.0f}ms to {sig_times.max():.0f}ms")
        else:
            print(f"  {area}: no significant time points")

        area_stats[area] = {
            'm_group': m_group, 'ci_low': ci_low, 'ci_high': ci_high, 'sig_mask': sig_mask,
            'obs_peak': obs_peak, 'peak_val': peak_val, 'low': low, 'high': high,
            'color': area_colors[a_idx], 'a_idx': a_idx,
        }

    if not area_stats:
        ax.axis('off')
        continue

    # 2. This panel's own dynamic y-scaling and significance-lane spacing -- since sharey=False,
    # every k gets a row_gap sized to its OWN data, not a figure-wide constant.
    local_max_y = max(np.max(s['ci_high']) for s in area_stats.values())
    row_gap = local_max_y * 0.05

    # 3. Plot each ROI's curve, ribbon, peak marker, and staggered significance lane.
    for area, s in area_stats.items():
        leg_text = f"{area}: {s['obs_peak']:.0f}ms [{s['low']:.0f}-{s['high']:.0f}ms]"
        ax.plot(times, s['m_group'], color=s['color'], lw=4.0, label=leg_text, zorder=3)
        ax.fill_between(times, s['ci_low'], s['ci_high'], color=s['color'], alpha=0.20, zorder=2)
        ax.scatter(s['obs_peak'], s['peak_val'], color=s['color'], s=160, edgecolors='white',
                   linewidth=1.2, zorder=5)

        # Significance marker -- staggered lane below y=0, one per ROI.
        sig_y = -row_gap * (s['a_idx'] + 1)
        if np.any(s['sig_mask']):
            ax.scatter(times[s['sig_mask']], [sig_y] * np.sum(s['sig_mask']),
                       color=s['color'], s=30, marker='s', alpha=0.8, edgecolors='none', zorder=3)

    title = 'k=1 (univariate)' if k == 1 else f'k={k}'
    ax.set_title(title, fontweight='bold', fontsize=18, pad=12)
    ax.axvline(0, color='black', lw=2.5, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=2.5, alpha=0.2)
    ax.set_xlim(-100, 600)
    ax.set_xticks([0, 200, 400, 600])
    # Only the bottom is pinned (to fit the significance lanes) -- top stays auto/dynamic.
    bottom_limit = -row_gap * (len(area_labels) + 1.5)
    ax.set_ylim(bottom=bottom_limit)
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.tick_params(axis='both', labelsize=13, width=2.5, length=10.0)

plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, "roi_searchlight_rsa_k_comparison_by_k.svg")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\nPlot saved to: {save_path}")
print(f"Total Execution Time: {time.time() - start_time:.2f} seconds.")