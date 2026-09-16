import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utils import sign_permutation_cluster_test, get_eeg_times
import time

# Start time
start_time = time.time()

# --- Configuration ---
subject_list = [1, 4, 5, 6, 7, 8]
hemis = ['lh', 'rh']

roi_groups = {
    'V1': ['V1v', 'V1d'],
    'V4': ['hV4'],
    'ventral': ['ventral']
}

area_labels = ['V1', 'V4', 'ventral']

n_bootstraps = 10000

# Mapping internal saved partial correlation dictionary keys to their curve identity within each
# ROI's subplot -- fixed colors held constant across every subplot (blue for VDNN, orange for LLM).
partitions = {
    'vision_partial_correlation': {'title': 'VDNN', 'color': "#63a1cc"},   # steel blue
    'language_partial_correlation': {'title': 'LLM', 'color': "#fd9f25"}, # orange
}

# Grey used for the VDNN-vs-LLM difference significance bar
DIFF_COLOR = "#595959"


base_results_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/partial_correlation'
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()
n_timepoints = len(times)
post_stim_mask = times >= 0 

# --- Data Aggregation ---
# Structural layout: aggregated_data[partition_key][Area_idx] -> shape: (n_subjects, n_time)
aggregated_data = {part: [[] for _ in area_labels] for part in partitions}

print(">>> Aggregating Encoding Partial Correlation ROI Data <<<")

for a_idx, area in enumerate(area_labels):
    sub_rois = roi_groups[area]

    for subject in subject_list:
        # Dictionary to hold the raw (n_time, n_vertices) arrays across sub-ROIs and hemispheres
        pooled_vertices = {part: [] for part in partitions}

        for sub_roi in sub_rois:
            for hemi in hemis:
                file_path_even = os.path.join(base_results_dir, f'subject-{subject}', f'{sub_roi}_{hemi}_cv_split-even.npy')
                file_path_odd = os.path.join(base_results_dir, f'subject-{subject}', f'{sub_roi}_{hemi}_cv_split-odd.npy')


                if os.path.exists(file_path_even):
                    # Load the unified partial correlation dictionary file
                    results_dict_even = np.load(file_path_even, allow_pickle=True).item()
                    results_dict_odd = np.load(file_path_odd, allow_pickle=True).item()

                    for part in partitions:
                        # shape: (n_time, n_vertices)
                        data_even = results_dict_even[part]
                        data_odd = results_dict_odd[part]
                        data = (data_even + data_odd) / 2.0  # Average across even/odd splits

                        pooled_vertices[part].append(data)

        # For each partition type, concatenate across vertices and take the spatial mean
        for part in partitions:
            if len(pooled_vertices[part]) > 0:
                # Concatenate all hemispheres and sub-ROIs along the vertex dimension (axis=1)
                all_area_vertices = np.concatenate(pooled_vertices[part], axis=1)
                # Average across the combined vertex pool to get a clean 1D timecourse
                subject_timecourse = np.mean(all_area_vertices, axis=1)
                aggregated_data[part][a_idx].append(subject_timecourse)

# Cast lists into clean numpy arrays
for part in partitions:
    for a_idx in range(len(area_labels)):
        aggregated_data[part][a_idx] = np.array(aggregated_data[part][a_idx])


def ci95_across_subjects(area_data, n_bootstraps=10000):
    """
    95% confidence interval of the across-subject mean at each timepoint, via percentile
    bootstrap: resample subjects with replacement n_bootstraps times, recompute the mean
    across the resampled subjects at each timepoint, then take the 2.5th/97.5th percentiles
    of that bootstrap distribution.
    """
    n_subs = area_data.shape[0]
    boot_means = np.zeros((n_bootstraps, area_data.shape[1]))
    for i in range(n_bootstraps):
        res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
        boot_means[i] = np.mean(area_data[res_idx], axis=0)
    low, high = np.percentile(boot_means, [2.5, 97.5], axis=0)
    return low, high


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
# Compute all stats once per ROI -- curves, cluster tests, peak latencies, post-stimulus
# mean/SD, and the VDNN-vs-LLM difference test -- printing everything here, then plot below.
# =============================================================================
print("\n>>> Computing stats per ROI <<<")
stats_cache = {}   # stats_cache[area] = {'model_stats': {...}, 'row_gap': ..., 'local_max_y': ...}
diff_cache = {}    # diff_cache[area] = {'sig_mask': ...} or None

part_keys = list(partitions.keys())  # [vision, language] -- order matters for the VDNN-LLM diff

for a_idx, area in enumerate(area_labels):
    print(f"\nPlotting Panel: {area}")
    print(f">>> Peak latencies (95% CI, bootstrap over subjects) -- {area} <<<")

    # Calculate unique local y-limits to maximize tracking resolution inside this subplot
    # (using the 95% CI upper bound). Bootstrap once per partition and reuse below, rather
    # than recomputing it.
    all_cis = [ci95_across_subjects(aggregated_data[p][a_idx]) for p in partitions
               if len(aggregated_data[p][a_idx]) > 0]  # each: (low, high) arrays

    local_max_y = max(np.max(high) for low, high in all_cis) if all_cis else 0.1

    # Row spacing for the staggered significance lanes below y=0 (one lane per curve, plus one
    # extra lane for the VDNN-vs-LLM difference test)
    row_gap = local_max_y * 0.05

    model_stats = {}
    for p_idx, (part_key, config) in enumerate(partitions.items()):
        area_data = aggregated_data[part_key][a_idx]
        if len(area_data) == 0:
            continue

        n_subs = len(area_data)
        m_group = np.mean(area_data, axis=0)
        ci_low, ci_high = ci95_across_subjects(area_data)  # shaded area CI: uncertainty in the correlation value itself

        # 1. Cluster Permutation Test
        cluster_results = sign_permutation_cluster_test(area_data, n_permutations=10000)
        sig_mask = np.zeros(n_timepoints, dtype=bool)
        for cluster_idx, _, _ in cluster_results['significant_clusters']:
            sig_mask[cluster_idx] = True

        print_significance_summary(config['title'], cluster_results['significant_clusters'], times)

        # 2. Bootstrap Peak Latency CI
        boot_peaks = []
        for _ in range(n_bootstraps):
            res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
            boot_peaks.append(times[np.argmax(np.mean(area_data[res_idx], axis=0))])

        low, high = np.percentile(boot_peaks, [2.5, 97.5])
        obs_peak = times[np.argmax(m_group)]

        print(f"  {config['title']}: peak latency = {obs_peak:.0f}ms [95% CI: {low:.0f}-{high:.0f}ms]")

        # Mean +/- SD post-stimulus correlation: per-subject post-stimulus mean first, then the
        # mean and SD of those subject-level values (SD across subjects, NOT across time).
        post_stim_vals = area_data[:, post_stim_mask].mean(axis=1)
        mean_val = np.mean(post_stim_vals)
        sd_val = np.std(post_stim_vals, ddof=1)
        print(f"  {config['title']}: mean post-stimulus = {mean_val:.4f} +/- {sd_val:.4f} (SD across subjects)")

        leg_text = f"{config['title']}: {obs_peak:.0f}ms [{low:.0f}-{high:.0f}ms]"
        model_stats[config['title']] = {
            'm_group': m_group, 'ci_low': ci_low, 'ci_high': ci_high, 'sig_mask': sig_mask,
            'color': config['color'], 'obs_peak': obs_peak, 'peak_val': np.max(m_group),
            'leg_text': leg_text, 'p_idx': p_idx,
        }

    # 4. VDNN vs LLM difference: cluster-based sign-permutation test on the per-subject
    # difference (VDNN - LLM).
    vdnn_data = aggregated_data[part_keys[0]][a_idx]
    llm_data = aggregated_data[part_keys[1]][a_idx]
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

    stats_cache[area] = {'model_stats': model_stats, 'row_gap': row_gap, 'local_max_y': local_max_y}
    diff_cache[area] = diff_info


# =============================================================================
# Plotting -- one subplot per ROI, VDNN and LLM curves, significance bars below y=0
# (one bar per curve, plus one for the VDNN-vs-LLM difference test).
# =============================================================================
print("\n>>> Rendering plot <<<")
fig, axes = plt.subplots(1, len(area_labels), figsize=(6 * len(area_labels), 6), sharex=False)

for a_idx, area in enumerate(area_labels):
    ax = axes[a_idx]
    cache = stats_cache[area]
    model_stats = cache['model_stats']
    row_gap = cache['row_gap']
    n_lanes = len(partitions) + 1

    for title, s in model_stats.items():
        # 3. Curve and Variance Ribbon Plotting
        ax.plot(times, s['m_group'], color=s['color'], lw=8.0, label=s['leg_text'], zorder=3)
        ax.fill_between(times, s['ci_low'], s['ci_high'],
                         color=s['color'], alpha=0.20, zorder=2)

        # Peak Markers with Errorbars
        ax.scatter(s['obs_peak'], s['peak_val'], color=s['color'], s=500, edgecolors='white', zorder=5)

        # Significance Markers -- staggered lanes BELOW the y=0 line (one lane per curve)
        sig_y = -row_gap * (s['p_idx'] + 1)
        if np.any(s['sig_mask']):
            ax.scatter(times[s['sig_mask']], [sig_y] * np.sum(s['sig_mask']),
                       color=s['color'], s=40, marker='s', alpha=0.8, edgecolors='none', zorder=3)

    diff_info = diff_cache[area]
    if diff_info is not None and np.any(diff_info['sig_mask']):
        diff_sig_y = -row_gap * n_lanes
        ax.scatter(times[diff_info['sig_mask']], [diff_sig_y] * np.sum(diff_info['sig_mask']),
                   color=DIFF_COLOR, s=40, marker='s', alpha=0.8, edgecolors='none', zorder=3)

    # Subplot Aesthetic Configuration
    ax.set_title(area, fontweight='bold', fontsize=22, pad=15)
    ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=3, alpha=0.2)
    ax.set_xlim(-100, 600)

    # Adjust y-limit constraints smoothly to support standard correlation ranges
    bottom_limit = -row_gap * (n_lanes + 1.5)
    top_limit = 0.25
    ax.set_ylim(bottom=bottom_limit, top=top_limit)

    #ax.legend(loc='upper right', frameon=False, fontsize=18, ncol=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.tick_params(axis='both', labelsize=26, width=3.0, length=14.0)
    #ax.legend(loc='upper right', frameon=False, fontsize=26)

# Global Figure Layout settings
plt.suptitle('', fontweight='bold', fontsize=26, y=1.01)

plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, "roi_wise_vdnn_llm_partial_correlation.svg")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {save_path}")

# =============================================================================
# Individual-subject grid, 6 rows (subjects) x 3 columns (ROIs).
# =============================================================================
print("\n>>> Plotting individual-subject VDNN/LLM partial correlation grid <<<")
fig, axes = plt.subplots(len(subject_list), len(area_labels),
                          figsize=(7 * len(area_labels), 4.5 * len(subject_list)),
                          sharex=False, sharey=True)

for s_idx, subject in enumerate(subject_list):
    for a_idx, area in enumerate(area_labels):
        ax = axes[s_idx, a_idx]

        for part_key in part_keys:
            area_data = aggregated_data[part_key][a_idx]
            if len(area_data) == 0 or s_idx >= len(area_data):
                continue
            config = partitions[part_key]
            curve = area_data[s_idx, :]
            obs_peak = times[np.argmax(curve)]
            ax.plot(times, curve, color=config['color'], lw=5, zorder=3)

        ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', lw=3, alpha=0.2)
        ax.set_xlim(-100, 600)
        ax.set_xticks([0, 200, 400, 600])
        #ax.set_yticks([0.0, 0.1, 0.2, 0.3])
        #ax.legend(loc='upper right', frameon=False, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(3.0)
        ax.spines['bottom'].set_linewidth(3.0)
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=26, width=3.0, length=14.0,
                        labelbottom=False, labelleft=False)

        if s_idx == 0:
            ax.set_title(area, fontweight='bold', fontsize=16, pad=10)
        if a_idx == 0:
            ax.set_ylabel(f'Participant {subject}', fontweight='bold', fontsize=13)

plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, "roi_wise_vdnn_llm_partial_correlation_individual_participants.svg")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved to: {save_path}")

print(f"\nSuccess! Total Execution Time: {time.time() - start_time:.2f} seconds")