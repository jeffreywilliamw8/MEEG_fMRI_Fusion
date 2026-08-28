"""
This script aggregates the ROI-level correlation results from the second phase of the Stimulus Feature Encoding Fusion (SFEF) analysis
for features extracted from different layers of the AlexNet model. It computes the best-predicting layer for each ROI based 
on mean post-stimulus correlation values and performs statistical analyses, including cluster-based permutation tests and bootstrap
confidence intervals for peak latencies.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from scipy.stats import sem
from utils import sign_permutation_cluster_test, get_eeg_times, compute_best_performing_layer_results, bootstrap_ci_curve, compute_best_performing_layer_bins, compute_subject_layer_bins, merge_significant_bins, get_bin_window
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

# Width of the bins used to determine the best performing layer over time.
BIN_WIDTH_MS = 20

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


base_results_root = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/stimulus_feature_encoding_fusion/phase_2/layerwise_alexnet/roi'

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

        data[layer][area] = np.array(subject_area_corrs) if len(subject_area_corrs) > 0 else None


print("\n>>> Determining overall best performing AlexNet layer per ROI <<<")
best_performing_layer_results = compute_best_performing_layer_results(
    data, alexnet_layers, layer_display_names, area_labels, times, n_bootstraps=n_bootstraps
)

for area in area_labels:
    r = best_performing_layer_results[area]
    if r is None:
        print(f"{area}: no data found, skipping.")
        continue
    print(f"{area}: best performing layer = {r['layer_display']} (mean post-onset r = {r['group_score']:.4f}), "
          f"peak latency = {r['peak_latency']:.0f}ms [95% CI: {r['ci_low']:.0f}-{r['ci_high']:.0f}ms]")


# =============================================================================
# Precompute per-layer curves/CI/significance for ALL layers, ALL ROIs (reusing the overall
# best performing layer's already-computed sig_mask where applicable, so it's never tested
# twice), then determine the best performing layer per 20ms
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
bin_window_cache = {}
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
    bin_window_cache[area] = get_bin_window(times, model_stats_for_bins)


def get_last_significant_time(times, model_stats):
    """
    Last timepoint where ANY layer's group-level cluster test is significant -- the sibling
    endpoint to get_bin_window's t_start (first significant timepoint). Used below to bound the
    "best layer regardless of significance" bins to the group's overall significant window,
    rather than running all the way to the end of the epoch.
    """
    combined_sig = np.zeros(len(times), dtype=bool)
    for s in model_stats.values():
        combined_sig |= s['sig_mask']
    if not np.any(combined_sig):
        return None
    return times[np.where(combined_sig)[0][-1]]


# =============================================================================
# Best performing layer per bin, 
# =============================================================================
print(f"\n>>> Determining best performing AlexNet layer per {BIN_WIDTH_MS}ms bin, ALL LAYERS (no significance gating) <<<")
allsig_bins_cache = {}
for area in area_labels:
    layer_stats = layer_stats_cache[area]
    model_stats_for_bins = {layer: {'m_group': s['m_group'], 'sig_mask': s['sig_mask']}
                             for layer, s in layer_stats.items()}
    t_start, _ = bin_window_cache[area]
    t_end_sig = get_last_significant_time(times, model_stats_for_bins)

    print(f"{area}:")
    if t_start is None or t_end_sig is None:
        print("  no group-level significance found, skipping")
        allsig_bins_cache[area] = []
        continue

    group_layer_curves = {layer: s['m_group'] for layer, s in layer_stats.items()}
    bins_info = compute_subject_layer_bins(times, group_layer_curves, t_start, t_end_sig, bin_width_ms=BIN_WIDTH_MS)
    merged_windows = merge_significant_bins(bins_info)
    if not merged_windows:
        print("  no bins in significant window")
    else:
        for w in merged_windows:
            display = layer_stats[w['winner']]['display']
            print(f"  {display} best performing from {w['start']:.0f}ms to {w['end']:.0f}ms "
                  f"(regardless of per-layer significance)")
    allsig_bins_cache[area] = merged_windows


# =============================================================================
# Per-subject binning
# =============================================================================
print(f"\n>>> Determining best performing AlexNet layer per {BIN_WIDTH_MS}ms bin, PER SUBJECT <<<")
subject_curves_cache = {}
subject_bins_cache = {}
for area in area_labels:
    t_start, t_end = bin_window_cache[area]
    for s_idx, subject in enumerate(subject_list):
        layer_curves = {}
        for layer in alexnet_layers:
            area_data = data[layer][area]
            if area_data is None:
                continue
            layer_curves[layer] = area_data[s_idx, :]
        subject_curves_cache[(subject, area)] = layer_curves

        if t_start is None or len(layer_curves) == 0:
            subject_bins_cache[(subject, area)] = []
            continue

        bins_info = compute_subject_layer_bins(times, layer_curves, t_start, t_end, bin_width_ms=BIN_WIDTH_MS)
        subject_bins_cache[(subject, area)] = merge_significant_bins(bins_info)

    print(f"{area}: computed per-subject bins for {len(subject_list)} subjects "
          f"(window starts at {t_start:.0f}ms)" if t_start is not None else
          f"{area}: no group-level significance found, per-subject bins skipped")


# =============================================================================
# Plotting: one subplot per ROI, one curve per layer. Per-layer significance bars in
# staggered horizontal bars y=0 (all 8 layers, own color each).
# =============================================================================

maxes = [0.3, 0.5, 1.0]

def render_figure(save_name):
    print("\n>>> Plotting layerwise AlexNet SFEF results (best performing layer per bin) <<<")
    fig, axes = plt.subplots(1, len(area_labels), figsize=(9 * len(area_labels), 8), sharex=False)
    if len(area_labels) == 1:
        axes = [axes]

    for a_idx, area in enumerate(area_labels):
        ax = axes[a_idx]
        layer_stats = layer_stats_cache[area]
        row_gap = row_gap_cache[area]
        global_max_y = global_max_y_cache[area]

        for layer, s in layer_stats.items():
            ax.plot(times, s['m_group'], color=s['color'], lw=4.0, label=s['display'], zorder=3)
            #ax.fill_between(times, s['ci_low'], s['ci_high'], color=s['color'], alpha=0.15, zorder=2)

            sig_y = -row_gap * (s['l_idx'] + 1)
            if np.any(s['sig_mask']):
                ax.scatter(times[s['sig_mask']], [sig_y] * np.sum(s['sig_mask']),
                           color=s['color'], s=14, marker='s', alpha=0.8, edgecolors='none', zorder=3)

        top_bar_y = global_max_y * 1.08
        for w in bins_cache[area]:
            color = layer_stats[w['winner']]['color']
            ax.plot([w['start'], w['end']], [top_bar_y, top_bar_y], color=color, lw=12.0,
                    solid_capstyle='butt', zorder=4)

        # Second bar, sitting just above the significance-gated one: best performing layer per
        # bin with all layers eligible (no significance gating), over the [first-significant,
        # last-significant] window 
        top_bar_y_allsig = global_max_y * 1.20
        for w in allsig_bins_cache[area]:
            color = layer_stats[w['winner']]['color']
            ax.plot([w['start'], w['end']], [top_bar_y_allsig, top_bar_y_allsig], color=color, lw=12.0,
                    solid_capstyle='butt', zorder=4)

        ax.set_title(area, fontweight='bold', fontsize=22, pad=15)
        ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', lw=3, alpha=0.2)
        xticks = [-100, 0, 200, 400, 600]
        ax.set_xticks(ticks=xticks)
        ax.set_xlim(-100, 600)
        bottom_limit = -row_gap * (n_layers + 1.5)
        top_limit = global_max_y * 1.5
        #top_limit = maxes[a_idx]
        ax.set_ylim(bottom=bottom_limit, top=top_limit)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(3.0)
        ax.spines['bottom'].set_linewidth(3.0)
        ax.tick_params(axis='both', labelsize=26, width=3.0, length=14.0)
        ax.tick_params(axis='both', labelsize=18)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


render_figure("roi_enc_layerwise_alexnet_fusion_best_layer_bin.svg")


# =============================================================================
# Plotting: individual-subject grid, 6 rows (subjects) x 3 columns (ROIs)
# =============================================================================
def render_subject_figure(save_name):
    print("\n>>> Plotting layerwise AlexNet SFEF results, individual subjects <<<")
    n_rows, n_cols = len(subject_list), len(area_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), sharex=False)

    for s_idx, subject in enumerate(subject_list):
        for a_idx, area in enumerate(area_labels):
            ax = axes[s_idx, a_idx]
            layer_curves = subject_curves_cache[(subject, area)]
            merged_windows = subject_bins_cache[(subject, area)]

            if len(layer_curves) == 0:
                ax.axis('off')
                continue

            curve_max = max(c.max() for c in layer_curves.values())
            curve_min = min(c.min() for c in layer_curves.values())
            top_limit = curve_max * 1.35 if curve_max > 0 else curve_max + 0.01
            bottom_limit = min(0, curve_min) - 0.05 * (curve_max - min(0, curve_min))
            top_bar_y = curve_max * 1.08

            for l_idx, layer in enumerate(alexnet_layers):
                if layer not in layer_curves:
                    continue
                ax.plot(times, layer_curves[layer], color=layer_colors[l_idx], lw=1.6, zorder=3)

            for w in merged_windows:
                color = layer_colors[alexnet_layers.index(w['winner'])]
                ax.plot([w['start'], w['end']], [top_bar_y, top_bar_y], color=color, lw=4.0,
                        solid_capstyle='butt', zorder=4)

            ax.axvline(0, color='black', lw=1.5, linestyle='--', alpha=0.5)
            ax.axhline(0, color='black', lw=1.5, alpha=0.2)
            ax.set_xticks([-100, 0, 200, 400, 600])
            ax.set_xlim(-100, 600)
            ax.set_ylim(bottom=bottom_limit, top=top_limit)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', labelsize=9)

            if s_idx == 0:
                ax.set_title(area, fontweight='bold', fontsize=16, pad=10)
            if a_idx == 0:
                ax.set_ylabel(f'Sub-{subject}', fontweight='bold', fontsize=13)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


render_subject_figure("roi_enc_layerwise_alexnet_fusion_individual_subjects_best_layer_bin.svg")

print(f"\nTotal Execution time: {time.time() - start_time:.2f} seconds.")