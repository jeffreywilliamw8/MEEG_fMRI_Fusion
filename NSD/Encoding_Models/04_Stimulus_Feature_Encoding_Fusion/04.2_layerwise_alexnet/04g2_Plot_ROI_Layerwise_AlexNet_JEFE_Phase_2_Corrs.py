import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
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
layer_colors = [
    "#B23434",  # Conv1
    "#B29234",  # Conv2
    "#73B234",  # Conv3
    "#34B292",  # Conv4
    "#1A9BE0",  # Conv5
    "#3453B2",  # FC6
    "#7334B2",  # FC7
    "#B23492",  # FC8
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
                    path_lh_even = os.path.join(base_dir, f'subject-{subject}', f'{sub_roi}_lh_cv_split-even.npy')
                    path_lh_odd = os.path.join(base_dir, f'subject-{subject}', f'{sub_roi}_lh_cv_split-odd.npy')

                    path_rh_even = os.path.join(base_dir, f'subject-{subject}', f'{sub_roi}_rh_cv_split-even.npy')
                    path_rh_odd = os.path.join(base_dir, f'subject-{subject}', f'{sub_roi}_rh_cv_split-odd.npy')

                    if os.path.exists(path_lh_even) and os.path.exists(path_rh_odd):
                        data_lh = (np.load(path_lh_even)+np.load(path_lh_odd))/2.0 # Averaging results across even and odd splits
                        data_rh = (np.load(path_rh_even)+np.load(path_rh_odd))/2.0


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
    resamples at every timepoint in one shot
    """
    n_subs = area_data.shape[0]
    res_idx = np.random.randint(0, n_subs, size=(n_bootstraps, n_subs))
    boot_means = area_data[res_idx].mean(axis=1)  # (n_bootstraps, n_time)
    lower_pct, upper_pct = (100 - ci) / 2, 100 - (100 - ci) / 2
    ci_low, ci_high = np.percentile(boot_means, [lower_pct, upper_pct], axis=0)
    return ci_low, ci_high


# =============================================================================
# Best-performing-layer selection (per ROI)
#
# Selection rule: for each subject, summarize a layer's performance as that
# subject's own mean correlation across post-stimulus time (t >= 0). The overall
# best performing layer for an ROI is then the layer with the highest mean of those
# per-subject summaries (i.e. averaged across time AND across subjects)
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


def get_bin_window(times, model_stats):
    """
    The shared [t_start, t_end) binning window: t_start is the earliest timepoint where ANY
    layer's group-level cluster test is significant, t_end is the end of the epoch. This is the
    exact same combined_sig computation used inside compute_best_performing_layer_bins, pulled
    out separately (without touching that function) so individual-subject bins can start at
    exactly the same group-determined onset -- i.e. "the time found significant across all 6
    subjects" -- rather than each subject picking their own onset.
    """
    combined_sig = np.zeros(len(times), dtype=bool)
    for s in model_stats.values():
        combined_sig |= s['sig_mask']
    if not np.any(combined_sig):
        return None, None
    t_start = times[np.where(combined_sig)[0][0]]
    t_end = times[-1]
    return t_start, t_end


def compute_subject_layer_bins(times, layer_curves, t_start, t_end, bin_width_ms=20):
    """
    Single-subject analog of compute_best_performing_layer_bins. There is no significance test
    for a single subject (n=1, nothing to bootstrap/permute), so every layer is an eligible
    candidate in every bin -- the winner is simply whichever layer has the highest bin-averaged
    correlation, no significance gating at all. Binning starts at the group-level t_start (shared
    across every subject and every bin, from get_bin_window above) and tiles forward to t_end
    (end of epoch), same bin edges the group plot uses.
    """
    layer_titles = list(layer_curves.keys())
    bins = []
    edge = t_start
    while edge < t_end:
        in_bin = (times >= edge) & (times < edge + bin_width_ms)
        if np.any(in_bin):
            bin_means = {title: layer_curves[title][in_bin].mean() for title in layer_titles}
            winner = max(bin_means, key=bin_means.get)
            bins.append({'start': edge, 'end': edge + bin_width_ms, 'winner': winner, 'significant': True})
        edge += bin_width_ms
    return bins


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
def render_figure(save_name):
    print("\n>>> Plotting layerwise AlexNet JEFE results (best performing layer per bin) <<<")
    fig, axes = plt.subplots(1, len(area_labels), figsize=(9 * len(area_labels), 8), sharex=False)
    if len(area_labels) == 1:
        axes = [axes]

    for a_idx, area in enumerate(area_labels):
        ax = axes[a_idx]
        layer_stats = layer_stats_cache[area]
        row_gap = row_gap_cache[area]
        global_max_y = global_max_y_cache[area]

        for layer, s in layer_stats.items():
            ax.plot(times, s['m_group'], color=s['color'], lw=3.0, label=s['display'], zorder=3)
            ax.fill_between(times, s['ci_low'], s['ci_high'], color=s['color'], alpha=0.12, zorder=2)

            sig_y = -row_gap * (s['l_idx'] + 1)
            if np.any(s['sig_mask']):
                ax.scatter(times[s['sig_mask']], [sig_y] * np.sum(s['sig_mask']),
                           color=s['color'], s=14, marker='s', alpha=0.8, edgecolors='none', zorder=3)

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
        ax.set_ylim(bottom=bottom_limit, top=0.38)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(8.0)
        ax.spines['bottom'].set_linewidth(8.0)
        ax.tick_params(axis='both', labelsize=26, width=6.0, length=18.0)
        ax.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


render_figure("roi_enc_layerwise_alexnet_fusion.svg")


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
    print("\n>>> Plotting layerwise AlexNet JEFE results, individual subjects <<<")
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
                ax.plot([w['start'], w['end']], [top_bar_y, top_bar_y], color=color, lw=5.0,
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


render_subject_figure("roi_enc_layerwise_alexnet_fusion_individual_subjects.svg")

print(f"\nTotal Execution time: {time.time() - start_time:.2f} seconds.")