import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import os
from tqdm import tqdm
from scipy.stats import sem
from utils import sign_permutation_cluster_test, get_eeg_times, compute_best_performing_layer_results, bootstrap_ci_curve, compute_best_performing_layer_bins, compute_subject_layer_bins, merge_significant_bins, get_bin_window
from berg import BERG
from matplotlib.ticker import MaxNLocator
import time

# Start time
start_time = time.time()

# --- Configuration ---
subject_list = [1, 4, 5, 6, 7, 8]
n_bootstraps = 10000

# ROI grouping: 3 panels -- V1, V4, ventral

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


import matplotlib.colors as mcolors

# Pathing -- layerwise AlexNet RSA commonality analysis (whole-brain, aggregated across fMRI splits)
base_results_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/commonality_analysis/layerwise_alexnet/eeg_rdm_metric-pearsonr/wb'
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
# stats. Doing this once here means the terminal report and the plot rendered below reuse the
# exact same numbers.
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


# =============================================================================
# Per-subject binning: same bin edges as the group (t_start from bin_window_cache, tiling to
# the end of the epoch), but computed on each subject's OWN raw curves individually, with no
# significance gating (single subject -- nothing to test). This is what the individual-subject
# figure below highlights, to check whether the group-level layer-progression pattern also
# shows up subject by subject.
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
# Plotting: one subplot per ROI, one curve per layer. Per-layer significance bars sit in
# staggered lanes BELOW y=0 (all 8 layers, own color each).
#
# The best performing layer is determined PER 20ms BIN (not a single fixed layer for the whole
# epoch) and drawn as colored bar segments -- only where a layer is significant across the
# WHOLE bin -- side by side along the top, so the winner is allowed to change over time. The
# single overall best performing layer (computed above) is no longer plotted; it's still
# reported in the terminal.
# =============================================================================

maxes = [0.4, 0.6, 1.0]

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

        ax.set_title(area, fontweight='bold', fontsize=22, pad=15)
        ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', lw=3, alpha=0.2)
        xticks = [-100, 0, 200, 400, 600]
        ax.set_xticks(ticks=xticks)
        ax.set_xlim(-100, 600)
        bottom_limit = -row_gap * (n_layers + 1.5)
        bottom_limit = -0.01
        #top_limit = global_max_y * 1.35
        top_limit = maxes[a_idx]
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


render_figure("roi_rsa_layerwise_alexnet_fusion.svg")


# =============================================================================
# Plotting: individual-subject grid, 6 rows (subjects) x 3 columns (ROIs). Each panel shows
# that single subject's own 8 layer curves (no CI ribbon -- can't bootstrap over n=1) plus a
# top color bar for the per-bin winning layer, using the SAME bin edges as the group plot
# (starting at the group's significance onset, tiling to the end of the epoch). There is no
# below-curve significance lane here, since single-subject data has no significance test to
# plot. This is purely a visual check for whether the group-level best-performing-layer
# progression also holds up subject by subject.
# =============================================================================
def render_subject_figure(save_name):
    print("\n>>> Plotting layerwise AlexNet RSA results, individual subjects <<<")
    n_rows, n_cols = len(subject_list), len(area_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), sharex=False)

    post_stim_mask = times >= 0

    for s_idx, subject in enumerate(subject_list):
        for a_idx, area in enumerate(area_labels):
            ax = axes[s_idx, a_idx]
            layer_curves = subject_curves_cache[(subject, area)]
            merged_windows = subject_bins_cache[(subject, area)]

            if len(layer_curves) == 0:
                ax.axis('off')
                continue

            # Best-performing layer for this subject/ROI: highest post-stimulus-period
            # time-averaged score, plus that layer's own peak latency. Printout + marker only --
            # does not affect the per-bin winning-layer color bar below, which is unchanged.
            best_layer, best_score = None, -np.inf
            for layer, curve in layer_curves.items():
                score = curve[post_stim_mask].mean()
                if score > best_score:
                    best_layer, best_score = layer, score
            best_curve = layer_curves[best_layer]
            best_peak_idx = np.argmax(best_curve)
            best_peak_latency = times[best_peak_idx]
            best_peak_val = best_curve[best_peak_idx]
            print(f"Sub-{subject}, {area}: best layer = {best_layer} "
                  f"(mean post-stimulus score = {best_score:.4f}), "
                  f"peak latency = {best_peak_latency:.0f}ms")

            curve_max = max(c.max() for c in layer_curves.values())
            curve_min = min(c.min() for c in layer_curves.values())
            top_limit = curve_max * 1.35 if curve_max > 0 else curve_max + 0.01
            bottom_limit = min(0, curve_min) - 0.05 * (curve_max - min(0, curve_min))
            top_bar_y = curve_max * 1.08

            for l_idx, layer in enumerate(alexnet_layers):
                if layer not in layer_curves:
                    continue
                ax.plot(times, layer_curves[layer], color=layer_colors[l_idx], lw=3, zorder=3)

            # Peak marker for the best-performing layer, colored to match its curve.
            best_color = layer_colors[alexnet_layers.index(best_layer)]
            ax.scatter(best_peak_latency, best_peak_val, color=best_color, s=180,
                       edgecolor='white', linewidth=1.2, zorder=5)

            for w in merged_windows:
                color = layer_colors[alexnet_layers.index(w['winner'])]
                ax.plot([w['start'], w['end']], [top_bar_y, top_bar_y], color=color, lw=4.0,
                        solid_capstyle='butt', zorder=4)

            ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
            ax.axhline(0, color='black', lw=3, alpha=0.2)
            ax.set_xticks([0, 200, 400, 600])
            ax.set_xlim(-100, 600)
            ax.set_ylim(bottom=bottom_limit, top=top_limit)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_xlabel('')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', labelsize=11, labelbottom=False, labelleft=False, width=3.0, length=12.0)

            if s_idx == 0:
                ax.set_title(area, fontweight='bold', fontsize=16, pad=10)
            if a_idx == 0:
                ax.set_ylabel(f'Sub-{subject}', fontweight='bold', fontsize=13)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


render_subject_figure("roi_rsa_layerwise_alexnet_fusion_individual_participants.svg")

print(f"\nExecution complete! Total Time: {time.time() - start_time:.2f}s")