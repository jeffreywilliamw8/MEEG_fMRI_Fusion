"""
This script aggregates the ROI-level correlation results from the second phase of the Joint EEG-Feature Encoding Fusion (JEFE) analysis
for features extracted from different layers of the AlexNet model. It computes the best performing layer for each ROI based
on mean post-stimulus correlation values and performs statistical analyses, including bootstrap confidence intervals for peak latencies.

Significance testing: per-timepoint one-sample t-test on Fisher-z-transformed Pearson
correlations (one-tailed), against a population mean of 0, Benjamini-Hochberg FDR-corrected
across timepoints (alpha=0.05, method='fdr_bh') -- replaces the sign-permutation cluster test
used in the original version of this script (kept untouched as
04g_Plot_ROI_Layerwise_AlexNet_JEFE_Phase_2_Corrs.py). The best-performing-layer selection
logic itself (overall winner + per-20ms-bin winner) is unchanged; only the underlying
significance test that feeds it has changed.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from scipy.stats import sem, ttest_1samp
from statsmodels.stats.multitest import multipletests
from utils import get_eeg_times
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


def fisher_z(r):
    """Fisher z-transform (arctanh) of a Pearson correlation coefficient, clipped just inside
    (-1, 1) so an exact +/-1 value doesn't produce +/-inf."""
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def ttest_fdr_sig_mask(data, alpha=0.05, alternative='greater'):
    """
    Per-timepoint one-sample t-test against a population mean of 0, Benjamini-Hochberg
    FDR-corrected across timepoints (alpha=0.05, method='fdr_bh'). Replaces the cluster-based
    sign-permutation test used previously.
    data: (n_subjects, n_time) -- already Fisher-z-transformed.
    """
    n_time = data.shape[1]
    pvals = np.array([ttest_1samp(data[:, t], popmean=0, alternative=alternative).pvalue
                       for t in range(n_time)])
    reject, _, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    return reject


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
# Best-performing-layer selection (per ROI)
#
# Selection rule: for each subject, summarize a layer's performance as that
# subject's own mean correlation across post-stimulus time (t >= 0). The overall
# best performing layer for an ROI is then the layer with the highest mean of those
# per-subject summaries (i.e. averaged across time AND across subjects) -- not
# the layer with the single highest peak, which is a noisier, single-timepoint
# statistic.
#
# This also runs (once) the per-timepoint t-test + BH-FDR significance test and the
# bootstrap-over-subjects peak-latency CI for whichever layer wins, so the
# terminal report is driven by the exact same selection and the exact same
# test/bootstrap draw. This overall best performing layer is no longer
# highlighted in the plot (see compute_best_performing_layer_bins below for the
# per-bin version that IS plotted) but is still printed to the terminal.
# =============================================================================
def compute_best_performing_layer_results(data, alexnet_layers, layer_display_names, area_labels, times,
                                           n_bootstraps=10000):
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

        sig_mask = ttest_fdr_sig_mask(fisher_z(best_area_data), alpha=0.05, alternative='greater')

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
    for which EVERY time point inside the bin is significant (that layer's own significance
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
            sig_mask = ttest_fdr_sig_mask(fisher_z(area_data), alpha=0.05, alternative='greater')

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


render_figure("roi_enc_layerwise_alexnet_fusion_w_t_test_stats.svg")

print(f"\nTotal Execution time: {time.time() - start_time:.2f} seconds.")