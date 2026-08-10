"""
This script aggregates results for the AlexNet layer-wise commonality analysis into 3 ROIs (V1, V4 ventral),
performs the statistical analyses and plots the participant-averaged results

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import os
from tqdm import tqdm
from scipy.stats import sem
from utils import sign_permutation_cluster_test, get_eeg_times
from berg import BERG
import time

# Start time
start_time = time.time()

# --- Configuration ---
subject_list = [1, 4, 5, 6, 7, 8]
n_bootstraps = 10000

# ROI grouping: 3 panels -- V1-3 combined, hV4, ventral

roi_groups = {
    'V1': ['V1v', 'V1d', 'V2v', 'V2d'],
    'V4': ['hV4'],
    'ventral': ['ventral']
}
area_labels = ['V1-3', 'V4', 'ventral']
#area_labels = ['V1', 'V2', 'V3', 'hV4', 'ventral']

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

# Width of the bins used to determine the best performing layer over time.
BIN_WIDTH_MS = 20

# Layer-depth colormap: dark purple (shallow) -> blue -> green (deep). Replaces plasma,
# whose bright yellow endpoint (deepest layer) was nearly invisible on a white background.
# Every stop here (purple, blue, green) stays readable on white, and the blue midpoint
# doubles as a nod to the project's EEG-blue convention.
layer_cmap = LinearSegmentedColormap.from_list('depth_purple_to_green', ['#3B0F70', '#2C6E8C', '#5FA935'])



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

import matplotlib.colors as mcolors

# Discrete Matplotlib Colormap
alexnet_cmap = mcolors.ListedColormap(layer_colors)
# Pathing -- layerwise AlexNet RSA commonality analysis (whole-brain, aggregated across fMRI splits)
base_results_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/commonality_analysis/layerwise_alexnet/wb'
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()
n_timepoints = len(times)


def select_vertices(data, mask, noise_ceilings, threshold=0.2):
    """Select vertices belonging to the ROI (mask) AND passing the noise-ceiling threshold."""
    condition1 = mask != 0
    condition2 = noise_ceilings >= threshold
    combined_condition = condition1 & condition2
    return data[:, combined_condition]


# --- Data Aggregation: data[layer][area] -> (n_subjects, n_time) ---
print(">>> Aggregating layerwise AlexNet RSA data (Averaging sub-ROIs) <<<")

data = {layer: {} for layer in alexnet_layers}

for layer in alexnet_layers:
    print(f"Processing layer: {layer}")
    for area in area_labels:
        sub_rois = roi_groups[area]
        subject_area_corrs = []

        for subject in subject_list:
            sub_roi_corrs = None
            for sr, sub_roi in enumerate(sub_rois):
                try:
                    path_lh = os.path.join(base_results_dir, f'subject-{subject}', f'layer-{layer}', 'correlations_left.npy')
                    path_rh = os.path.join(base_results_dir, f'subject-{subject}', f'layer-{layer}', 'correlations_right.npy')

                    data_lh = 1000*np.load(path_lh)
                    data_rh = 1000*np.load(path_rh)

                    berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
                    metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)

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

            if sub_roi_corrs is not None:
                subject_area_corrs.append(np.mean(sub_roi_corrs, axis=1))  # average across vertices

        data[layer][area] = np.array(subject_area_corrs) if len(subject_area_corrs) > 0 else None


def bootstrap_ci_curve(area_data, n_bootstraps=10000, ci=95):
    """
    Bootstrap (over subjects) percentile confidence interval of the mean correlation
    at each timepoint -- this is what the shaded area around each curve shows.
    """
    n_subs = area_data.shape[0]
    res_idx = np.random.randint(0, n_subs, size=(n_bootstraps, n_subs))
    boot_means = area_data[res_idx].mean(axis=1)  # (n_bootstraps, n_time)
    lower_pct, upper_pct = (100 - ci) / 2, 100 - (100 - ci) / 2
    ci_low, ci_high = np.percentile(boot_means, [lower_pct, upper_pct], axis=0)
    return ci_low, ci_high


# =============================================================================
# Best-performing-layer selection (per ROI) -- SINGLE overall best performing layer. Used only
# for the terminal report now (the plot no longer highlights this one fixed layer -- see
# compute_best_performing_layer_bins below for the per-bin version that IS plotted).
# Selection rule: for each subject, summarize a layer's performance as that subject's own mean
# correlation across post-stimulus time (t >= 0). The overall best performing layer for an ROI
# is then the layer with the highest mean of those per-subject summaries.
# =============================================================================
def compute_best_performing_layer_results(data, alexnet_layers, layer_display_names, area_labels, times,
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


def compute_best_performing_layer_bins(times, model_stats, bin_width_ms=20):
    """
    For each `bin_width_ms` bin -- starting at the first timepoint where ANY layer is
    significant, and tiling forward to the end of the epoch -- restricts candidates to layers
    for which EVERY time point inside the bin is significant (that layer's own cluster-based
    test), then picks the best performing layer among those candidates as the one with the
    higher bin-averaged group score. If no layer is significant across the WHOLE bin, the bin
    has no winner and is not flagged as significant. Returns a list of per-bin dicts, or [] if
    no layer is ever significant anywhere. Generic over however many "models" are in
    model_stats -- used here with all 8 AlexNet layers, but the same function as in the
    VDNN/LLM scripts.
    """
    model_titles = list(model_stats.keys())
    combined_sig = np.zeros(len(times), dtype=bool)
    for title in model_titles:
        combined_sig |= model_stats[title]['sig_mask']

    if not np.any(combined_sig):
        return []

    t_start = times[np.where(combined_sig)[0][0]]
    t_end = times[-1]

    bins = []
    edge = t_start
    while edge < t_end:
        in_bin = (times >= edge) & (times < edge + bin_width_ms)
        if np.any(in_bin):
            # Only layers significant at EVERY time point within the bin are eligible.
            fully_sig_titles = [title for title in model_titles
                                 if np.all(model_stats[title]['sig_mask'][in_bin])]
            if fully_sig_titles:
                bin_means = {title: model_stats[title]['m_group'][in_bin].mean()
                             for title in fully_sig_titles}
                winner = max(bin_means, key=bin_means.get)
                bins.append({'start': edge, 'end': edge + bin_width_ms, 'winner': winner,
                             'significant': True})
            else:
                bins.append({'start': edge, 'end': edge + bin_width_ms, 'winner': None,
                             'significant': False})
        edge += bin_width_ms
    return bins


def merge_significant_bins(bins):
    """Merges consecutive, adjoining bins that share the same significant winner into contiguous
    windows, for compact reporting/plotting."""
    merged = []
    current = None
    for b in bins:
        if not b['significant']:
            if current is not None:
                merged.append(current)
            current = None
            continue
        if current is not None and current['winner'] == b['winner'] and current['end'] == b['start']:
            current['end'] = b['end']
        else:
            if current is not None:
                merged.append(current)
            current = {'winner': b['winner'], 'start': b['start'], 'end': b['end']}
    if current is not None:
        merged.append(current)
    return merged


print("\n>>> Determining overall best performing AlexNet layer per ROI <<<")
best_performing_layer_results = compute_best_performing_layer_results(
    data, alexnet_layers, layer_display_names, area_labels, times, n_bootstraps=n_bootstraps
)

for area in area_labels:
    r = best_performing_layer_results[area]
    if r is None:
        print(f"{area}: no data found, skipping.")
        continue
    print(f"{area}: best performing layer = {r['layer_display']} (mean post-onset R = {r['group_score']:.4f}), "
          f"peak latency = {r['peak_latency']:.0f}ms [95% CI: {r['ci_low']:.0f}-{r['ci_high']:.0f}ms]")


# =============================================================================
# Precompute per-layer curves/CI/significance for ALL layers, ALL ROIs (reusing the overall
# best performing layer's already-computed sig_mask where applicable, so it's never tested
# twice), then determine the best performing layer per 20ms bin from those cached per-layer
# stats.
# =============================================================================
print("\n>>> Precomputing per-layer curves and significance for all ROIs <<<")
layer_stats_cache = {}
row_gap_cache = {}
global_max_y_cache = {}

for area in area_labels:
    all_means, all_ci_highs = [], []
    for layer in alexnet_layers:
        d = data[layer][area]
        if d is not None:
            _, ci_high = bootstrap_ci_curve(d, n_bootstraps=n_bootstraps)
            all_means.append(np.mean(d, axis=0))
            all_ci_highs.append(ci_high)
    global_max_y = max([np.max(c) for c in all_ci_highs]) if all_ci_highs else 0.06
    row_gap = global_max_y * 0.035
    best = best_performing_layer_results.get(area)

    layer_stats = {}
    for l_idx, layer in enumerate(alexnet_layers):
        area_data = data[layer][area]
        if area_data is None:
            print(f"  {area} / {layer}: no data found, skipping.")
            continue

        m_group = np.mean(area_data, axis=0)
        ci_low, ci_high = bootstrap_ci_curve(area_data, n_bootstraps=n_bootstraps)

        if best is not None and l_idx == best['layer_idx']:
            sig_mask = best['sig_mask']
        else:
            cluster_results = sign_permutation_cluster_test(area_data, n_permutations=10000)
            sig_mask = np.zeros(n_timepoints, dtype=bool)
            for cluster_idx, _, _ in cluster_results['significant_clusters']:
                sig_mask[cluster_idx] = True

        layer_stats[layer] = {
            'm_group': m_group, 'ci_low': ci_low, 'ci_high': ci_high, 'sig_mask': sig_mask,
            'color': layer_colors[l_idx], 'l_idx': l_idx, 'display': layer_display_names[l_idx],
        }

    layer_stats_cache[area] = layer_stats
    row_gap_cache[area] = row_gap
    global_max_y_cache[area] = global_max_y

print(f"\n>>> Determining best performing AlexNet layer per {BIN_WIDTH_MS}ms bin <<<")
bins_cache = {}
for area in area_labels:
    layer_stats = layer_stats_cache[area]
    model_stats_for_bins = {layer: {'m_group': s['m_group'], 'sig_mask': s['sig_mask']}
                             for layer, s in layer_stats.items()}
    bins_info = compute_best_performing_layer_bins(times, model_stats_for_bins, bin_width_ms=BIN_WIDTH_MS)
    merged_windows = merge_significant_bins(bins_info)
    print(f"{area}:")
    if not merged_windows:
        print("  no significant best performing layer bins")
    else:
        for w in merged_windows:
            display = layer_stats[w['winner']]['display']
            print(f"  {display} best performing & significant from {w['start']:.0f}ms to {w['end']:.0f}ms")
    bins_cache[area] = merged_windows


# =============================================================================
# Plotting: one subplot per ROI, one curve per layer. Per-layer significance bars sit in
# staggered lanes BELOW y=0 (all 8 layers, own color each).
#
# The best performing layer is determined PER 20ms BIN (not a single fixed layer for the whole
# epoch) and drawn as colored bar segments -- only where a layer is significant across the
# WHOLE bin -- side by side along the top, so the winner is allowed to change over time. The
# single overall best performing layer printed in the terminal.
# =============================================================================
def render_figure(save_name):
    print("\n>>> Plotting layerwise AlexNet RSA results (best performing layer per bin) <<<")
    fig, axes = plt.subplots(1, len(area_labels), figsize=(9 * len(area_labels), 8), sharex=False)
    if len(area_labels) == 1:
        axes = [axes]

    for a_idx, area in enumerate(area_labels):
        ax = axes[a_idx]
        layer_stats = layer_stats_cache[area]
        row_gap = row_gap_cache[area]
        global_max_y = global_max_y_cache[area]

        for layer, s in layer_stats.items():
            ax.plot(times, s['m_group'], color=s['color'], lw=7.0, label=s['display'], zorder=3)
            #ax.fill_between(times, s['ci_low'], s['ci_high'], color=s['color'], alpha=0.15, zorder=2)

            sig_y = -row_gap * (s['l_idx'] + 1)
            if np.any(s['sig_mask']):
                ax.scatter(times[s['sig_mask']], [sig_y] * np.sum(s['sig_mask']),
                           color=s['color'], s=20, marker='s', alpha=0.8, edgecolors='none', zorder=3)

        top_bar_y = global_max_y * 1.08
        for w in bins_cache[area]:
            color = layer_stats[w['winner']]['color']
            ax.plot([w['start'], w['end']], [top_bar_y, top_bar_y], color=color, lw=8.0,
                    solid_capstyle='butt', zorder=4)

        ax.set_title(area, fontweight='bold', fontsize=22, pad=15)
        ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', lw=3, alpha=0.2)
        xticks = [-100, 0, 200, 400, 600]
        ax.set_xticks(ticks=xticks)
        ax.set_xlim(-100, 600)
        bottom_limit = -row_gap * (n_layers + 1.5)
        top_limit = global_max_y * 1.35
        ax.set_ylim(bottom=bottom_limit, top=top_limit)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(8.0)
        ax.spines['bottom'].set_linewidth(8.0)
        ax.tick_params(axis='both', labelsize=26, width=6.0, length=18.0)
        ax.tick_params(axis='both', labelsize=18)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


render_figure("roi_rsa_layerwise_alexnet_fusion.svg")

print(f"\nExecution complete! Total Time: {time.time() - start_time:.2f}s")