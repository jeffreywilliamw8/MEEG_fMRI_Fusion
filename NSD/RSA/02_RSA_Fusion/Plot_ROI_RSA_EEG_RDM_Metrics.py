"""
Group-average RSA fusion time courses compared across EEG RDM metrics (Pearson-r, crossnobis,
decoding_accuracy)
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
eeg_rdm_metrics = ['pearsonr', 'crossnobis', 'decoding_accuracy']
k = 100  # searchlight neighbourhood size

roi_groups = {
    'V1': ['V1v', 'V1d'],
    'V4': ['hV4'],
    'ventral': ['ventral'],
}
area_labels = ['V1', 'V4', 'ventral']

metric_labels = {
    'pearsonr': 'Pearson r',
    'crossnobis': 'Crossnobis',
    'decoding_accuracy': 'Decoding accuracy',
}
# one color per metric
metric_colors = {
    'pearsonr': "#7570B3",         # purple
    'crossnobis': "#D47827",       # orange
    'decoding_accuracy': "#3BA33B", # green
}

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


def get_subject_dir(metric, subject):
    """Resolves the per-subject aggregated searchlight results directory for a given EEG RDM
    metric, at the fixed searchlight size k defined above."""
    return os.path.join(
        f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/'
        f'searchlight_fusion/eeg_rdm_metric-{metric}/n_neighbours-{k}/aggregated_results',
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
# Data aggregation: aggregated[metric][area] -> (n_subjects, n_time)
# =============================================================================
print(">>> Aggregating ROI-averaged RSA fusion time courses across EEG RDM metrics <<<")

aggregated = {metric: {area: [] for area in area_labels} for metric in eeg_rdm_metrics}

for metric in eeg_rdm_metrics:
    for subject in tqdm(subject_list, desc=f'metric={metric}'):
        subject_dir = get_subject_dir(metric, subject)
        try:
            data_lh = np.load(os.path.join(
                subject_dir, f'subject-{subject}_lh_hemisphere_timecourse.npy'))
            data_rh = np.load(os.path.join(
                subject_dir, f'subject-{subject}_rh_hemisphere_timecourse.npy'))
        except FileNotFoundError:
            print(f"  Missing data for metric={metric}, subject {subject}")
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
                aggregated[metric][area].append(np.mean(sub_roi_corrs, axis=1))  # avg across vertices

for metric in eeg_rdm_metrics:
    for area in area_labels:
        aggregated[metric][area] = np.array(aggregated[metric][area])  # (n_subjects, n_time)

# =============================================================================
# Plot: 1x3 subplots (V1, V4, ventral), one curve per metric, group-average + 95% CI shaded area +
# peak marker + peak-latency-with-CI legend + significance lane, per subplot.
# =============================================================================
print("\nPlotting group-average RSA fusion time courses across EEG RDM metrics...")

fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=False)

for a_idx, area in enumerate(area_labels):
    ax = axes[a_idx]

    print(f"\n>>> {area} <<<")

    # 1. Compute all per-metric stats once (mean, CI ribbon, cluster significance, peak + CI),
    # before touching the axes -- this panel's own row_gap/y-scaling (below) depends on them.
    metric_stats = {}
    for m_idx, metric in enumerate(eeg_rdm_metrics):
        area_data = aggregated[metric][area]
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

        print(f"  {metric_labels[metric]}: peak latency = {obs_peak:.0f}ms "
              f"[95% CI: {low:.0f}-{high:.0f}ms]")
        if np.any(sig_mask):
            sig_times = times[sig_mask]
            print(f"  {metric_labels[metric]}: significant from {sig_times.min():.0f}ms to "
                  f"{sig_times.max():.0f}ms")
        else:
            print(f"  {metric_labels[metric]}: no significant time points")

        metric_stats[metric] = {
            'm_group': m_group, 'ci_low': ci_low, 'ci_high': ci_high, 'sig_mask': sig_mask,
            'obs_peak': obs_peak, 'peak_val': peak_val, 'low': low, 'high': high,
            'color': metric_colors[metric], 'm_idx': m_idx,
        }

    if not metric_stats:
        ax.axis('off')
        continue

    # 2. This panel's own dynamic y-scaling and significance-lane spacing -- since sharey=False,
    # every ROI gets a row_gap sized to its OWN data, not a figure-wide constant.
    local_max_y = max(np.max(s['ci_high']) for s in metric_stats.values())
    row_gap = local_max_y * 0.05

    # 3. Plot each metric's curve, ribbon, peak marker, and staggered significance lane.
    for metric, s in metric_stats.items():
        leg_text = f"{metric_labels[metric]}: {s['obs_peak']:.0f}ms [{s['low']:.0f}-{s['high']:.0f}ms]"
        ax.plot(times, s['m_group'], color=s['color'], lw=7.0, label=leg_text, zorder=3)
        ax.fill_between(times, s['ci_low'], s['ci_high'], color=s['color'], alpha=0.15, zorder=2)
        ax.scatter(s['obs_peak'], s['peak_val'], color=s['color'], s=180, edgecolors='white',
                   linewidth=1.2, zorder=5)

        # Significance marker -- staggered lane below y=0, one per metric.
        sig_y = -row_gap * (s['m_idx'] + 1)
        if np.any(s['sig_mask']):
            ax.scatter(times[s['sig_mask']], [sig_y] * np.sum(s['sig_mask']),
                       color=s['color'], s=30, marker='s', alpha=0.8, edgecolors='none', zorder=3)

    ax.set_title(area, fontweight='bold', fontsize=20, pad=12)
    ax.axvline(0, color='black', lw=2.5, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=2.5, alpha=0.2)
    ax.set_xlim(-100, 600)
    ax.set_xticks([0, 200, 400, 600])
    # Only the bottom is pinned (to fit the significance lanes) -- top stays auto/dynamic.
    bottom_limit = -row_gap * (len(eeg_rdm_metrics) + 1.5)
    ax.set_ylim(bottom=bottom_limit)
    ax.set_xlabel('Time (ms)', fontsize=16)
    if a_idx == 0:
        ax.set_ylabel("Spearman's R", fontsize=16)
    #ax.legend(loc='upper right', frameon=False, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.tick_params(axis='both', labelsize=13, width=2.5, length=10.0)

plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, "roi_searchlight_rsa_eeg_rdm_metrics.svg")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\nPlot saved to: {save_path}")
print(f"Total Execution Time: {time.time() - start_time:.2f} seconds.")