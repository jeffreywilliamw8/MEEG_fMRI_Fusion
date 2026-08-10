"""
This script plots the variance partitioning results for specific ROIs (V1, V4, ventral) across time, 
comparing the contributions of vision DNN features (VDNN) and language model features (LLM) to the fMRI responses. 
It aggregates data across subjects, computes statistics including confidence intervals and significance testing, and generates plots for each ROI.
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
subject_list = [1, 4, 5, 6, 7, 8]

roi_groups = {
    'V1': ['V1v', 'V1d'],
    'V4': ['hV4'],
    'ventral': ['ventral']
}

rois = ['V1', 'V4', 'ventral']

n_bootstraps = 10000

# Mapping keys to their curve identity within each ROI's subplot -- fixed colors held constant
# across every subplot (blue for VDNN, orange for LLM), same convention as the encoding version.
partitions = {
    'unique_vision': {'title': 'VDNN', 'color': "#63a1cc"},   # steel blue
    'unique_language': {'title': 'LLM', 'color': "#fd9f25"},  # orange
}
maxes = [0.017, 0.0035, 0.005]

# Grey used for the VDNN-vs-LLM difference significance lane -- one shared shade, dark enough to
# read clearly on a white background, distinct from both curve colors.
DIFF_COLOR = "#595959"

# Source data: whole-brain, aggregated across fMRI splits (per subject/hemisphere/variance_type)
base_results_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/variance_partitioning/wb'
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()
n_timepoints = len(times)
post_stim_mask = times >= 0  # for the mean/SD post-stimulus summary below


def select_vertices(data, mask, noise_ceilings, threshold=0.2):
    """Select vertices belonging to the ROI (mask) AND passing the noise-ceiling threshold."""
    condition1 = mask != 0
    condition2 = noise_ceilings >= threshold
    combined_condition = condition1 & condition2
    return data[:, combined_condition]


# --- Data Aggregation ---
# Structural layout: aggregated_data[partition_key][roi_idx] -> shape: (n_subjects, n_time)
print(">>> Aggregating Variance Partitioning Data (whole-brain, searchlight, noise-ceiling filtered) <<<")

aggregated_data = {part: [[] for _ in rois] for part in partitions.keys()}

berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')

for subject in tqdm(subject_list):
    metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)
    wb_noise_ceilings_lh = metadata['fmri']['lh_ncsnr']
    wb_noise_ceilings_rh = metadata['fmri']['rh_ncsnr']

    for part in partitions.keys():
        # Load this subject's whole-brain time course ONCE per variance_type
        try:
            path_lh = os.path.join(base_results_dir, f'subject-{subject}', f'{part}_left.npy')
            path_rh = os.path.join(base_results_dir, f'subject-{subject}', f'{part}_right.npy')
            data_lh = 1000*np.load(path_lh)
            data_rh = 1000*np.load(path_rh)
        except FileNotFoundError:
            print(f"  Missing data for subject {subject}, variance_type {part}")
            continue

        for roi_idx, roi in enumerate(rois):
            sub_rois = roi_groups[roi]
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

                if sr == 0:
                    sub_roi_corrs = data_concat
                else:
                    sub_roi_corrs = np.concatenate([sub_roi_corrs, data_concat], axis=1)

            if sub_roi_corrs is not None:
                subject_timecourse = np.mean(sub_roi_corrs, axis=1)  # average across vertices, scale to %
                aggregated_data[part][roi_idx].append(subject_timecourse)

# Cast lists into clean arrays
for part in partitions.keys():
    for roi_idx in range(len(rois)):
        aggregated_data[part][roi_idx] = np.array(aggregated_data[part][roi_idx])


def ci95_across_subjects(area_data):
    """
    95% confidence interval of the across-subject mean at each timepoint, via the
    standard normal-theory formula (t-critical value * SEM).
    """
    n_subs = area_data.shape[0]
    s_err = sem(area_data, axis=0)
    t_crit = t_dist.ppf(0.975, df=n_subs - 1)
    return s_err * t_crit


def print_significance_summary(label, clusters, times, indent="  "):
    """Prints the overall significant time window (first-to-last significant timepoint, pooled
    across all significant clusters) and, if there is more than one significant cluster, the
    start/end of each individual cluster too."""
    if len(clusters) == 0:
        print(f"{indent}{label}: no significant time points")
        return

    all_idx = np.concatenate([np.asarray(cluster_idx) for cluster_idx, _, _ in clusters])
    sig_times = times[all_idx]
    print(f"{indent}{label}: significant from {sig_times.min():.0f}ms to {sig_times.max():.0f}ms")

    if len(clusters) > 1:
        for c_i, (cluster_idx, _, _) in enumerate(clusters):
            c_times = times[np.asarray(cluster_idx)]
            print(f"{indent}  Cluster {c_i + 1}: {c_times.min():.0f}ms to {c_times.max():.0f}ms")


# =============================================================================
# Compute all stats once per ROI, and print everything -- then plot below.
# =============================================================================
print("\n>>> Computing stats per ROI <<<")
stats_cache = {}   # stats_cache[roi] = {'model_stats': {...}, 'row_gap': ..., 'local_max_y': ...}
diff_cache = {}    # diff_cache[roi] = {'sig_mask': ...} or None

part_keys = list(partitions.keys())  # [vision, language] -- order matters for the VDNN-LLM diff

for roi_idx, roi in enumerate(rois):
    print(f"\nPlotting Panel: {roi}")
    print(f">>> Peak latencies (95% CI, bootstrap over subjects) -- {roi} <<<")

    all_means = [np.mean(aggregated_data[p][roi_idx], axis=0) for p in partitions
                 if len(aggregated_data[p][roi_idx]) > 0]
    all_cis = [ci95_across_subjects(aggregated_data[p][roi_idx]) for p in partitions
               if len(aggregated_data[p][roi_idx]) > 0]
    local_max_y = max([np.max(m + c) for m, c in zip(all_means, all_cis)]) if all_means else 0.5

    # Row spacing for the staggered significance lanes below y=0 (one lane per curve, plus one
    # extra lane for the VDNN-vs-LLM difference test)
    row_gap = local_max_y * 0.05

    model_stats = {}
    for p_idx, (part_key, config) in enumerate(partitions.items()):
        area_data = aggregated_data[part_key][roi_idx]
        if len(area_data) == 0:
            continue

        n_subs = len(area_data)
        m_group = np.mean(area_data, axis=0)
        s_err = sem(area_data, axis=0)
        ci_err = ci95_across_subjects(area_data)  # ribbon CI: uncertainty in the correlation value itself

        # 1. Cluster Permutation Test
        cluster_results = sign_permutation_cluster_test(area_data, n_permutations=10000)
        sig_mask = np.zeros(n_timepoints, dtype=bool)
        for cluster_idx, _, _ in cluster_results['significant_clusters']:
            sig_mask[cluster_idx] = True

        print_significance_summary(config['title'], cluster_results['significant_clusters'], times)

        # 2. Bootstrap Peak Latency CI: uncertainty in the peak's time index, obtained by
        # resampling subjects and re-finding the argmax each time 
        boot_peaks = []
        for _ in range(n_bootstraps):
            res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
            boot_peaks.append(times[np.argmax(np.mean(area_data[res_idx], axis=0))])

        low, high = np.percentile(boot_peaks, [2.5, 97.5])
        obs_peak = times[np.argmax(m_group)]

        print(f"  {config['title']}: peak latency = {obs_peak:.0f}ms [95% CI: {low:.0f}-{high:.0f}ms]")

        # Mean +/- SD post-stimulus R2: per-subject post-stimulus mean first, then the mean and SD
        # of those subject-level values (SD across subjects, NOT across time).
        post_stim_vals = area_data[:, post_stim_mask].mean(axis=1)
        mean_val = np.mean(post_stim_vals)
        sd_val = np.std(post_stim_vals, ddof=1)
        print(f"  {config['title']}: mean post-stimulus = {mean_val:.4f} +/- {sd_val:.4f} (SD across subjects)")

        leg_text = f"{config['title']}: {obs_peak:.0f}ms [{low:.0f}-{high:.0f}ms]"
        model_stats[config['title']] = {
            'm_group': m_group, 'ci_err': ci_err, 'sig_mask': sig_mask,
            'color': config['color'], 'obs_peak': obs_peak, 'peak_val': np.max(m_group),
            'leg_text': leg_text, 'p_idx': p_idx,
        }

    # 4. VDNN vs LLM difference: cluster-based sign-permutation test on the per-subject
    # difference (VDNN - LLM).
    vdnn_data = aggregated_data[part_keys[0]][roi_idx]
    llm_data = aggregated_data[part_keys[1]][roi_idx]
    diff_info = None
    if len(vdnn_data) > 0 and len(llm_data) > 0 and len(vdnn_data) == len(llm_data):
        diff_data = vdnn_data - llm_data
        diff_cluster_results = sign_permutation_cluster_test(diff_data, n_permutations=10000)
        diff_sig_mask = np.zeros(n_timepoints, dtype=bool)
        for cluster_idx, _, _ in diff_cluster_results['significant_clusters']:
            diff_sig_mask[cluster_idx] = True

        print_significance_summary("VDNN vs LLM difference", diff_cluster_results['significant_clusters'], times)
        diff_info = {'sig_mask': diff_sig_mask}
    else:
        print("  VDNN vs LLM difference: skipped (missing or mismatched subject data)")

    stats_cache[roi] = {'model_stats': model_stats, 'row_gap': row_gap, 'local_max_y': local_max_y}
    diff_cache[roi] = diff_info


# =============================================================================
# Plotting -- one subplot per ROI, VDNN and LLM curves, staggered significance lanes below y=0
# (one lane per curve, plus one for the VDNN-vs-LLM difference test).
# =============================================================================
print("\n>>> Rendering plot <<<")
fig, axes = plt.subplots(len(rois), 1, figsize=(12, 6 * len(rois)), sharex=False)

for roi_idx, roi in enumerate(rois):
    ax = axes[roi_idx]
    cache = stats_cache[roi]
    model_stats = cache['model_stats']
    row_gap = cache['row_gap']
    n_lanes = len(partitions) + 1

    for title, s in model_stats.items():
        # 3. Curve and Variance Ribbon Plotting --
        ax.plot(times, s['m_group'], color=s['color'], lw=8.0, label=s['leg_text'], zorder=3)
        ax.fill_between(times, s['m_group'] - s['ci_err'], s['m_group'] + s['ci_err'],
                         color=s['color'], alpha=0.20, zorder=2)

        # Peak Markers with Errorbars
        ax.scatter(s['obs_peak'], s['peak_val'], color=s['color'], s=260, edgecolors='white', zorder=5)

        # Significance Markers -- staggered lanes BELOW the y=0 line (one lane per curve)
        sig_y = -row_gap * (s['p_idx'] + 1)
        if np.any(s['sig_mask']):
            ax.scatter(times[s['sig_mask']], [sig_y] * np.sum(s['sig_mask']),
                       color=s['color'], s=40, marker='s', alpha=0.8, edgecolors='none', zorder=3)

    diff_info = diff_cache[roi]
    if diff_info is not None and np.any(diff_info['sig_mask']):
        diff_sig_y = -row_gap * n_lanes
        ax.scatter(times[diff_info['sig_mask']], [diff_sig_y] * np.sum(diff_info['sig_mask']),
                   color=DIFF_COLOR, s=40, marker='s', alpha=0.8, edgecolors='none', zorder=3)

    # Subplot Aesthetic Configuration
    ax.set_title(roi, fontweight='bold', fontsize=26, pad=15)
    ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=3, alpha=0.2)
    ax.set_xlim(-100, 600)

    bottom_limit = -row_gap * (n_lanes + 1.5)
    top_limit = 0.25
    ax.set_ylim(bottom=bottom_limit, top=top_limit)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.tick_params(axis='both', labelsize=26, width=3.0, length=14.0)

plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, "roi_wise_rsa_vdnn_llm_variance_partitioning.svg")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {save_path}")

print(f"\nExecution Complete! Total Execution Time: {time.time() - start_time:.2f} seconds.")