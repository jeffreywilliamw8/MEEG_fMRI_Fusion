"""
This script aggregates the ROI-level correlation results from the second phase of the Joint EEG-Feature Encoding Fusion (JEFE) analysis
for features extracted from different layers of the AlexNet model. It computes the best-predicting layer for each ROI based 
on mean post-stimulus correlation values and performs statistical analyses, including cluster-based permutation tests and bootstrap
confidence intervals for peak latencies.
"""



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
n_bootstraps = 10000


roi_groups = {
    'V1': ['V1v', 'V1d'],
    'V4': ['hV4'],
    'ventral': ['ventral']
}
area_labels = ['V1', 'V4', 'ventral']

alexnet_layers = [
    'features.2',    # Conv1 + Pool
    'features.5',    # Conv2 + Pool
    'features.7',    # Conv3
    'features.9',    # Conv4
    'features.12',   # Conv5 + Pool
    'classifier.2',  # FC6
    'classifier.5',  # FC7
    'classifier.6'   # FC8 (Output)
]
layer_display_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC6', 'FC7', 'FC8']
n_layers = len(alexnet_layers)

# Layer-depth colormap: dark purple (shallow) -> blue -> green (deep). Replaces plasma,
# whose bright yellow endpoint (deepest layer) was nearly invisible on a white background.
# Every stop here (purple, blue, green) stays readable on white, and the blue midpoint
# doubles as a nod to the project's EEG-blue convention.

layer_colors = [
    "#0C076E",  # Conv1
    "#5121A0",  # Conv2
    "#4C95BA",  # Conv3 
    "#16B28B",  # Conv4
    "#D4AC0D",  # Conv5
    "#D17C20",  # FC6 
    "#B23492",  # FC7
    "#CE1414"  # FC8 
]


# Base results root directory for AlexNet hierarchy
# Adjust the root directory naming convention below if your folder path differs
base_results_root = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/jefe_phase_2/roi/layerwise_alexnet'

PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()
n_timepoints = len(times)

# --- Data Aggregation ---
# Dictionary structure: processed_data[layer_key][area_name] -> numpy array of shape (n_subjects, n_time)
data = {layer: {} for layer in alexnet_layers}

print(">>> Aggregating ROI Data across AlexNet Layers <<<")
for layer in alexnet_layers:
    # Build path targeting the specific layer subfolder
    base_dir = os.path.join(base_results_root, 'layer-'+layer)

    for area in area_labels:
        sub_rois = roi_groups[area]
        subject_area_corrs = []

        for subject in subject_list:
            pooled_vertices = []

            for sub_roi in sub_rois:
                try:
                    path_lh = os.path.join(base_dir, f'subject-{subject}', f'{sub_roi}_lh.npy')
                    path_rh = os.path.join(base_dir, f'subject-{subject}', f'{sub_roi}_rh.npy')

                    if os.path.exists(path_lh) and os.path.exists(path_rh):
                        data_lh = np.load(path_lh)
                        data_rh = np.load(path_rh)

                        #data_lh = np.nan_to_num(data_lh, nan=0.0)
                        #data_rh = np.nan_to_num(data_rh, nan=0.0)

                        # Merge hemispheres
                        roi_concat = np.concatenate([data_lh, data_rh], axis=1)
                        pooled_vertices.append(roi_concat)
                except FileNotFoundError:
                    continue

            if len(pooled_vertices) > 0:
                # Merge all vertices belonging to this section (e.g. V1v + V1d)
                all_section_vertices = np.concatenate(pooled_vertices, axis=1)
                # Average across the combined spatial vertex pool
                subject_area_corrs.append(np.mean(all_section_vertices, axis=1))

        data[layer][area] = np.array(subject_area_corrs)


def bootstrap_ci_curve(area_data, n_bootstraps=10000, ci=95):
    """
    Bootstrap (over subjects) percentile confidence interval of the mean CORRELATION
    VALUE at each timepoint -- this is what the shaded ribbon around each curve shows.

    Vectorized: draws all n_bootstraps resamples of subjects at once (n_bootstraps x
    n_subs index array), averages each resample's curve, then takes percentiles across
    resamples at every timepoint in one shot, rather than looping in Python.

    This is a distinct quantity from the bootstrap CI used for peak LATENCY further
    below: that one resamples subjects and re-finds the argmax (a CI over a discrete
    time index), while this one resamples subjects and keeps the whole curve (a CI over
    the correlation value itself, at every timepoint).
    """
    n_subs = area_data.shape[0]
    res_idx = np.random.randint(0, n_subs, size=(n_bootstraps, n_subs))
    boot_means = area_data[res_idx].mean(axis=1)  # (n_bootstraps, n_time)
    lower_pct, upper_pct = (100 - ci) / 2, 100 - (100 - ci) / 2
    ci_low, ci_high = np.percentile(boot_means, [lower_pct, upper_pct], axis=0)
    return ci_low, ci_high


# =============================================================================
# Best-predicting-layer selection (per ROI)
#
# Selection rule: for each subject, summarize a layer's performance as that
# subject's own mean correlation across post-stimulus time (t >= 0). The overall
# best layer for an ROI is then the layer with the highest mean of those
# per-subject summaries (i.e. averaged across time AND across subjects) -- not
# the layer with the single highest peak, which is a noisier, single-timepoint
# statistic.
#
# This also runs (once) the cluster-permutation significance test and the
# bootstrap-over-subjects peak-latency CI for whichever layer wins, so both the
# terminal report and the plot's "best layer" highlight are driven by the exact
# same selection and the exact same permutation/bootstrap draw.
# =============================================================================
def compute_best_layer_results(data, alexnet_layers, layer_display_names, area_labels, times,
                                n_bootstraps=10000, n_permutations=10000):
    post_mask = times >= 0
    results = {}

    for area in area_labels:
        best_layer = None
        best_group_score = -np.inf
        best_area_data = None

        for layer in alexnet_layers:
            area_data = data[layer][area]
            if area_data is None or len(area_data) == 0:
                continue
            subject_scores = np.mean(area_data[:, post_mask], axis=1)  # per-subject, post-onset mean
            group_score = np.mean(subject_scores)                      # averaged across subjects
            if group_score > best_group_score:
                best_group_score = group_score
                best_layer = layer
                best_area_data = area_data

        if best_area_data is None:
            results[area] = None
            continue

        n_subs = best_area_data.shape[0]
        m_group = np.mean(best_area_data, axis=0)
        obs_peak = times[np.argmax(m_group)]

        boot_peaks = []
        for _ in range(n_bootstraps):
            res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
            boot_peaks.append(times[np.argmax(np.mean(best_area_data[res_idx], axis=0))])
        ci_low, ci_high = np.percentile(boot_peaks, [2.5, 97.5])

        cluster_results = sign_permutation_cluster_test(best_area_data, n_permutations=n_permutations)
        sig_mask = np.zeros(len(times), dtype=bool)
        for cluster_idx, _, _ in cluster_results['significant_clusters']:
            sig_mask[cluster_idx] = True

        layer_idx = alexnet_layers.index(best_layer)
        results[area] = {
            'layer': best_layer,
            'layer_idx': layer_idx,
            'layer_display': layer_display_names[layer_idx],
            'group_score': best_group_score,
            'peak_latency': obs_peak,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'sig_mask': sig_mask,
        }

    return results


print("\n>>> Determining overall best-predicting AlexNet layer per ROI <<<")
best_layer_results = compute_best_layer_results(
    data, alexnet_layers, layer_display_names, area_labels, times, n_bootstraps=n_bootstraps
)

for area in area_labels:
    r = best_layer_results[area]
    if r is None:
        print(f"{area}: no data found, skipping.")
        continue
    print(f"{area}: best layer = {r['layer_display']} (mean post-onset r = {r['group_score']:.4f}), "
          f"peak latency = {r['peak_latency']:.0f}ms [95% CI: {r['ci_low']:.0f}-{r['ci_high']:.0f}ms]")


# --- Plotting: one subplot per ROI, one curve per layer ---
# Per-layer significance bars sit in staggered lanes BELOW y=0 (all 8 layers, own color each).
# The overall best-predicting layer (per ROI, from compute_best_layer_results above) gets its
# own highlighted significance bar ABOVE the curves, plus a small labeled callout, so the
# "winning" layer is immediately visible without needing to cross-reference the terminal output.
print("\n>>> Plotting layerwise AlexNet JEFE results <<<")

fig, axes = plt.subplots(1, len(area_labels), figsize=(9 * len(area_labels), 8), sharex=False)
if len(area_labels) == 1:
    axes = [axes]

for a_idx, area in enumerate(area_labels):
    ax = axes[a_idx]

    all_means, all_ci_highs = [], []
    for layer in alexnet_layers:
        d = data[layer][area]
        if d is not None:
            _, ci_high = bootstrap_ci_curve(d, n_bootstraps=n_bootstraps)
            all_means.append(np.mean(d, axis=0))
            all_ci_highs.append(ci_high)
    global_max_y = max([np.max(c) for c in all_ci_highs]) if all_ci_highs else 0.06

    row_gap = global_max_y * 0.035
    best = best_layer_results.get(area)

    for l_idx, layer in enumerate(alexnet_layers):
        area_data = data[layer][area]
        if area_data is None:
            print(f"  {area} / {layer}: no data found, skipping.")
            continue

        m_group = np.mean(area_data, axis=0)
        ci_low, ci_high = bootstrap_ci_curve(area_data, n_bootstraps=n_bootstraps)
        color = layer_colors[l_idx]

        # Reuse the best layer's precomputed sig_mask (same permutation draw as the terminal
        # report / highlight bar below) instead of re-running the test a second time.
        if best is not None and l_idx == best['layer_idx']:
            sig_mask = best['sig_mask']
        else:
            cluster_results = sign_permutation_cluster_test(area_data, n_permutations=10000)
            sig_mask = np.zeros(n_timepoints, dtype=bool)
            for cluster_idx, _, _ in cluster_results['significant_clusters']:
                sig_mask[cluster_idx] = True

        ax.plot(times, m_group, color=color, lw=4.0, label=layer_display_names[l_idx], zorder=3)
        #ax.fill_between(times, ci_low, ci_high, color=color, alpha=0.12, zorder=2)

        # Per-layer significance bar: staggered lane below y=0
        sig_y = -row_gap * (l_idx + 1)
        if np.any(sig_mask):
            ax.scatter(times[sig_mask], [sig_y] * np.sum(sig_mask),
                       color=color, s=14, marker='s', alpha=0.8, edgecolors='none', zorder=3)

    # Best-layer highlight: significance bar above the curves + labeled callout
    if best is not None:
        best_color = layer_colors[best['layer_idx']]
        top_y = global_max_y * 1.08
        if np.any(best['sig_mask']):
            ax.scatter(times[best['sig_mask']], [top_y] * np.sum(best['sig_mask']),
                       color=best_color, s=50, marker='o', edgecolors='none',
                       alpha=0.8, zorder=3)


    ax.set_title(area, fontweight='bold', fontsize=22, pad=15)
    ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=3, alpha=0.2)
    xticks = [-100, 0, 200, 400, 600]
    ax.set_xticks(ticks=xticks)
    ax.set_xlim(-100, 600)
    bottom_limit = -row_gap * (n_layers + 1.5)
    ax.set_ylim(bottom=bottom_limit, top=0.38)

    #ax.legend(loc='upper right', frameon=False, fontsize=18, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.tick_params(axis='both', labelsize=26, width=3.0, length=14.0)
    ax.tick_params(axis='both', labelsize=16)

plt.tight_layout()
save_path = os.path.join(PLOTS_DIR, "roi_enc_layerwise_alexnet_fusion.svg")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {save_path}")
print(f"Total Execution time: {time.time() - start_time:.2f} seconds.")