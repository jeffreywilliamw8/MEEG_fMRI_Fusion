"""
Stream-wise box + swarm plot of whole-brain split-half reliability, comparing Encoding vs RSA.


Per subject, per hemisphere, the aggregated split-half files store an array of shape
(n_timepoints=359, n_shuffles=5, 2 halves, n_vertices=163842) -- see
Plot_WB_Split_Half_Reliability.py for the full rationale. In short: for each of 5 shuffles, the
vertex-wise Spearman correlation between the two halves' post-stimulus time courses is computed
(rank+center+cosine trick, fully vectorized), then averaged across shuffles -> one whole-brain
reliability value per vertex per subject, for each of Encoding and RSA.

Streams: the 7 BERG fsaverage ROI dict keys directly (NOT combined into coarser groups) --
'early', 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal'.

Per subject, per stream: vertices are filtered by (stream ROI mask) AND (ncsnr >= 0.2), then
averaged into a SINGLE value per subject -- so each box (and each swarm dot) represents one of
the N=6 per-subject values.

Stats: paired t-test (Encoding vs RSA) per stream, same subjects on both sides, Bonferroni
corrected across the 7 streams. Prints raw p, corrected p, and a significant/not verdict per
stream so significance stars can be added manually (e.g. in Inkscape).

Uses seaborn (boxplot + swarmplot) for the plot; pandas for the small per-stream/per-subject
dataframe that feeds it.
"""

import os
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import rankdata, ttest_rel
from utils import get_eeg_times
from berg import BERG
import pandas as pd
import seaborn as sns
import time

start_time = time.time()

# =============================================================================
# Configuration
# =============================================================================
subject_list = [1, 4, 5, 6, 7, 8]
N_NEIGHBOURS = 100
NCSNR_THRESHOLD = 0.2
N_VERT_PER_HEMI = 163842

STREAMS = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal']
analyses = ['Encoding', 'RSA']
analysis_colors = {'Encoding': '#1b9e77', 'RSA': '#7570b3'} 

eeg_times = get_eeg_times()
post_stim_mask = eeg_times >= 0

# Pathing -- matches the aggregation scripts' own save locations.
rsa_base_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/split_half_reliability/n_neighbours-{N_NEIGHBOURS}'
rsa_output_dir = f'{rsa_base_dir}/aggregated_results'
rsa_hemispheres = ['lh_hemisphere', 'rh_hemisphere']

encoding_base_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion_split_half_reliability/whole_brain'

plots_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/plots'
os.makedirs(plots_dir, exist_ok=True)


# =============================================================================
# Split-half reliability computation (same as Plot_WB_Split_Half_Reliability.py)
# =============================================================================
def rank_center(x):
    ranked = rankdata(x, axis=0)
    return ranked - ranked.mean(axis=0, keepdims=True)


def spearman_across_time(tc_a, tc_b):
    a = rank_center(tc_a)
    b = rank_center(tc_b)
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    denom[denom == 0] = np.nan
    return ((a * b).sum(axis=0) / denom).astype(np.float32)


def reorder_to_canonical(data, n_timepoints=None, n_shuffles=5, n_halves=2):
    """ The on-disk axis order isn't guaranteed to match between the Encoding and RSA aggregation pipelines (Encoding's
    turned out to be shuffle-first), so this detects axes by their expected sizes and reorders
    with np.moveaxis regardless of how a given file was actually saved."""
    if n_timepoints is None:
        n_timepoints = post_stim_mask.shape[0]

    shape = data.shape
    expected = {n_timepoints: 'time', n_shuffles: 'shuffle', n_halves: 'half'}
    axis_roles = {}
    for axis, size in enumerate(shape):
        if size in expected and expected[size] not in axis_roles:
            axis_roles[expected[size]] = axis
    remaining = [a for a in range(len(shape)) if a not in axis_roles.values()]

    if len(axis_roles) != 3 or len(remaining) != 1:
        raise ValueError(
            f"Could not unambiguously identify the time/shuffle/half axes in shape {shape} "
            f"(expected sizes: time={n_timepoints}, shuffle={n_shuffles}, half={n_halves}); "
            f"got axis_roles={axis_roles}.")

    order = [axis_roles['time'], axis_roles['shuffle'], axis_roles['half'], remaining[0]]
    if order != [0, 1, 2, 3]:
        print(f"    [reorder_to_canonical] on-disk shape {shape} -> canonical order {order}")
    return np.moveaxis(data, order, [0, 1, 2, 3])


def compute_split_half_reliability(data):
    """data: whole-brain split-half array for one subject (both hemispheres already
    concatenated along the vertex axis). Returns a single (n_vertices,) reliability map."""
    data = reorder_to_canonical(data)
    data_post = data[post_stim_mask]
    n_shuffles = data_post.shape[1]
    shuffle_maps = np.stack([
        spearman_across_time(data_post[:, s, 0, :], data_post[:, s, 1, :])
        for s in range(n_shuffles)
    ], axis=0)
    return np.nanmean(shuffle_maps, axis=0)


def load_rsa_split_half(subject):
    sub_name = f"subject-{subject}"
    hemi_arrays = []
    for hemi in rsa_hemispheres:
        file_path = os.path.join(rsa_output_dir, sub_name, f"{sub_name}_{hemi}_split_half_timecourse.npy")
        hemi_arrays.append(np.load(file_path))
    return np.concatenate(hemi_arrays, axis=-1)


def load_encoding_split_half(subject):
    sub_dir = os.path.join(encoding_base_dir, f'subject-{subject}')
    data_l = np.load(os.path.join(sub_dir, 'correlations_left.npy'))
    data_r = np.load(os.path.join(sub_dir, 'correlations_right.npy'))
    return np.concatenate([data_l, data_r], axis=-1)


loaders = {'Encoding': load_encoding_split_half, 'RSA': load_rsa_split_half}

# =============================================================================
# Per-subject whole-brain reliability maps
# =============================================================================
per_subject_maps = {name: [] for name in analyses}
for name in analyses:
    print(f"\n>>> Computing per-subject split-half reliability -- {name} <<<")
    for subject in tqdm(subject_list):
        data = loaders[name](subject)
        reliability_map = compute_split_half_reliability(data)
        per_subject_maps[name].append(reliability_map)
        print(f"  Subject {subject}: mean reliability = {np.nanmean(reliability_map):.4f}")

# =============================================================================
# Per-subject, per-stream masks (ROI + noise-ceiling)
# =============================================================================
print(f"\n>>> Building per-subject, per-stream masks (ROI + noise-ceiling >= {NCSNR_THRESHOLD}) <<<")
berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')

stream_masks = {}
for subject in subject_list:
    metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)
    wb_noise_ceilings_lh = metadata['fmri']['lh_ncsnr']
    wb_noise_ceilings_rh = metadata['fmri']['rh_ncsnr']

    stream_masks[subject] = {}
    for stream in STREAMS:
        roi_idx_lh = metadata['fmri']['lh_fsaverage_rois'][stream]
        roi_mask_lh = np.zeros(N_VERT_PER_HEMI, dtype=bool)
        roi_mask_lh[roi_idx_lh] = True

        roi_idx_rh = metadata['fmri']['rh_fsaverage_rois'][stream]
        roi_mask_rh = np.zeros(N_VERT_PER_HEMI, dtype=bool)
        roi_mask_rh[roi_idx_rh] = True

        combined_lh = roi_mask_lh & (wb_noise_ceilings_lh >= NCSNR_THRESHOLD)
        combined_rh = roi_mask_rh & (wb_noise_ceilings_rh >= NCSNR_THRESHOLD)
        stream_masks[subject][stream] = np.concatenate([combined_lh, combined_rh])

# =============================================================================
# Per-subject, per-stream, per-analysis aggregate reliability (single value per subject)
# =============================================================================
print("\n>>> Aggregating per-subject, per-stream mean reliability (Encoding vs RSA) <<<")
# stream_values[stream][analysis] -> list of per-subject means, aligned with subject_list
stream_values = {stream: {name: [] for name in analyses} for stream in STREAMS}

for stream in STREAMS:
    for s_idx, subject in enumerate(subject_list):
        mask = stream_masks[subject][stream]
        n_pass = int(mask.sum())
        if n_pass == 0:
            print(f"  WARNING: subject {subject}, stream '{stream}': 0 vertices pass the ROI + "
                  f"ncsnr filter -- skipping.")
            for name in analyses:
                stream_values[stream][name].append(np.nan)
            continue
        for name in analyses:
            vals = per_subject_maps[name][s_idx][mask]
            stream_values[stream][name].append(np.nanmean(vals))

# =============================================================================
# Paired stats: Encoding vs RSA within each stream, Bonferroni corrected across streams
# =============================================================================
print(f"\n>>> Encoding vs RSA paired t-test per stream (Bonferroni-corrected across "
      f"{len(STREAMS)} streams) <<<")
stream_pvals = {}
for stream in STREAMS:
    enc_vals = np.array(stream_values[stream]['Encoding'])
    rsa_vals = np.array(stream_values[stream]['RSA'])
    valid = ~np.isnan(enc_vals) & ~np.isnan(rsa_vals)
    if valid.sum() < 2:
        print(f"  {stream:<12} skipped (insufficient paired data)")
        continue
    _, p_val = ttest_rel(enc_vals[valid], rsa_vals[valid])
    stream_pvals[stream] = p_val

stream_names = list(stream_pvals.keys())
raw_pvals = np.array([stream_pvals[s] for s in stream_names])
corrected_pvals = np.minimum(raw_pvals * len(raw_pvals), 1.0)

for stream, raw_p, corr_p in zip(stream_names, raw_pvals, corrected_pvals):
    verdict = "SIGNIFICANT" if corr_p < 0.05 else "not significant"
    print(f"  {stream:<12} raw p = {raw_p:.4f}   Bonferroni-corrected p = {corr_p:.4f}   -> {verdict}")

# =============================================================================
# Overall summary, computed SEPARATELY for Encoding and RSA: a single grand mean collapsing
# BOTH subjects and streams, plus the SD across subjects of each subject's own mean across all
# 7 streams (i.e. the SD describes how much subjects vary once stream identity has been averaged
# away, not raw across-cell scatter).
# =============================================================================
print("\n>>> Overall split-half reliability (averaged across subjects AND streams) <<<")
for name in analyses:
    value_matrix = np.array([stream_values[stream][name] for stream in STREAMS]).T  # (n_subjects, n_streams)
    grand_mean = np.nanmean(value_matrix)
    per_subject_stream_mean = np.nanmean(value_matrix, axis=1)  # (n_subjects,)
    sd_across_subjects = np.nanstd(per_subject_stream_mean, ddof=1)
    print(f"  {name}: {grand_mean:.4f} +/- {sd_across_subjects:.4f} (SD across subjects)")

# =============================================================================
# Plotting: box + swarm, one panel, 2 boxes (+ overlaid subject dots) per stream
# =============================================================================
print("\n>>> Plotting stream-wise box + swarm (Encoding vs RSA) <<<")
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

df = pd.DataFrame([
    {'Stream': stream, 'Analysis': name, 'Reliability': val}
    for stream in STREAMS
    for name in analyses
    for val in stream_values[stream][name]
    if not np.isnan(val)
])

fig, ax = plt.subplots(figsize=(16, 8))
sns.boxplot(data=df, x='Stream', y='Reliability', hue='Analysis', order=STREAMS,
            hue_order=['Encoding', 'RSA'], palette=analysis_colors, showfliers=False,
            width=0.6, ax=ax)
sns.swarmplot(data=df, x='Stream', y='Reliability', hue='Analysis', order=STREAMS,
              hue_order=['Encoding', 'RSA'], palette=analysis_colors, dodge=True, size=6,
              edgecolor='black', linewidth=0.5, ax=ax)

# Box + swarm each add their own 'Encoding'/'RSA' legend handles -- keep only the first pair.
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:len(analyses)], labels[:len(analyses)], frameon=False, fontsize=16,
          loc='upper right')

ax.set_xlabel('')
ax.set_ylabel('Split-Half Reliability (Spearman)', fontsize=20)
ax.tick_params(axis='both', labelsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
save_path = os.path.join(plots_dir, 'wb_split_half_reliability_streamwise_boxswarm.svg')
fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False, format='svg')
plt.close(fig)
print(f"Box+swarm plot saved to: {save_path}")

print(f"\nExecution Complete! Total Time: {time.time() - start_time:.2f} seconds.")