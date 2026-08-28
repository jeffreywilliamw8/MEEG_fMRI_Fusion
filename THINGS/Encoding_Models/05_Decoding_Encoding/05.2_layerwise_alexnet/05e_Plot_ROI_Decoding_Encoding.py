"""
Plots ROI, layer-wise AlexNet Decoding-Encoding (DE) fusion correlation time
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from utils import get_meg_times

start_time = time.time()

# --- Configuration ---
fmri_subjects = [1, 2, 3]
halves = [1, 2]

area_labels = ['V1', 'hV4', 'IT']
area_colors = [
    "#480758",  # V1 (Deep Purple)
    "#468fc3",  # hV4 (Steel Blue)
    "#ea8e16",  # IT (Orange)
]

alexnet_layers = [
    'features.2',    # Conv1 + Pool
    'features.5',    # Conv2 + Pool
    'features.7',    # Conv3
    'features.9',    # Conv4
    'features.12',   # Conv5 + Pool
    'classifier.2',  # FC6
    'classifier.5',  # FC7
    'classifier.6',  # FC8 (Output)
]
layer_display_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC6', 'FC7', 'FC8']

layer_colors = [
    "#0C076E",  # Conv1
    "#5121A0",  # Conv2
    "#4C95BA",  # Conv3
    "#16B28B",  # Conv4
    "#D4AC0D",  # Conv5
    "#D17C20",  # FC6
    "#B23492",  # FC7
    "#CE1414",  # FC8
]

# Pathing -- must match THINGS_ROI_AlexNet_Layerwise_DE_Fusion.py's
# --corrs_save_dir default and save-directory layout.
base_results_dir = ('/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/'
                     'correlations/decoding_encoding_fusion/roi/layerwise_alexnet')
data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/THINGS/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector ---
times = get_meg_times(data_dir)
n_timepoints = len(times)


# =============================================================================
# Data Aggregation
#
# For each subject, average the two DE half runs (--half 1 and --half 2)
# together
# =============================================================================
def load_subject_roi_layer_curve(subject, roi, layer):
    curves = []
    for half in halves:
        path = os.path.join(base_results_dir, f'fmri_sub-{subject:02d}', f'roi-{roi}',
                             f'half-{half}', f'layer-{layer}', 'correlations.npy')
        if os.path.exists(path):
            d = np.load(path)  # (n_time, n_voxels)
            curves.append(d.mean(axis=1))  # average across ROI voxels -> (n_time,)
        else:
            print(f"  Missing: {path}")

    if len(curves) == 0:
        return None
    if len(curves) == 1:
        print(f"  NOTE: only one half found for sub-{subject:02d} / {roi} / {layer} -- "
              f"using it alone (no half-1/half-2 averaging).")
    return np.mean(curves, axis=0)


print(">>> Aggregating ROI data across AlexNet layers (DE fusion) <<<")
# data[layer][roi] -> (n_subjects_found, n_time)
data = {layer: {} for layer in alexnet_layers}
for layer in alexnet_layers:
    for roi in area_labels:
        subject_curves = []
        for subject in fmri_subjects:
            curve = load_subject_roi_layer_curve(subject, roi, layer)
            if curve is not None:
                subject_curves.append(curve)
        data[layer][roi] = np.array(subject_curves)


print("\n>>> Peak latency (observed; no significance stats) per ROI, per layer <<<")
for roi in area_labels:
    print(f"{roi}:")
    for l_idx, layer in enumerate(alexnet_layers):
        d = data[layer][roi]
        if d is None or len(d) == 0:
            print(f"  {layer_display_names[l_idx]}: no data found, skipping.")
            continue
        m_group = np.mean(d, axis=0)
        peak = times[np.argmax(m_group)]
        print(f"  {layer_display_names[l_idx]}: peak latency = {peak:.0f} ms "
              f"(mean r = {np.max(m_group):.4f})")


# =============================================================================
# Group figure: one subplot per ROI, one mean curve per layer. No ribbon
# (matching the SFEF layerwise script's own group figure), no significance
# markers.
# =============================================================================
def render_figure(save_name):
    print("\n>>> Plotting group-average layerwise AlexNet DE fusion results <<<")
    fig, axes = plt.subplots(1, len(area_labels), figsize=(9 * len(area_labels), 8), sharex=False)
    if len(area_labels) == 1:
        axes = [axes]

    for a_idx, roi in enumerate(area_labels):
        ax = axes[a_idx]
        all_means = []

        for l_idx, layer in enumerate(alexnet_layers):
            d = data[layer][roi]
            if d is None or len(d) == 0:
                continue
            m_group = np.mean(d, axis=0)
            all_means.append(m_group)
            ax.plot(times, m_group, color=layer_colors[l_idx], lw=3.0,
                    label=layer_display_names[l_idx], zorder=3)

        if all_means:
            global_max_y = max(np.max(m) for m in all_means)
            global_min_y = min(np.min(m) for m in all_means)
        else:
            global_max_y, global_min_y = 0.06, -0.02

        ax.set_title(roi, fontweight='bold', fontsize=22, pad=15)
        ax.set_xlabel('Time (ms)', fontsize=20)
        if a_idx == 0:
            ax.set_ylabel("Pearson's r", fontsize=20)
        ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', lw=3, alpha=0.2)
        ax.set_xticks([-100, 0, 200, 400, 600])
        ax.set_xlim(-100, 600)
        bottom_limit = min(global_min_y - 0.02, -0.02)
        ax.set_ylim(bottom=bottom_limit, top=global_max_y * 1.15)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(3.0)
        ax.spines['bottom'].set_linewidth(3.0)
        ax.tick_params(axis='both', labelsize=16, width=3.0, length=12.0)

        if a_idx == len(area_labels) - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


render_figure("roi_layerwise_alexnet_de_fusion.svg")


# =============================================================================
# Individual-subject figure: rows = subjects, columns = ROIs, each
# panel shows that subject's own raw per-layer curves. No stats, no CI.
# =============================================================================
def render_individual_subjects_figure(save_name):
    print("\n>>> Plotting individual-subject layerwise AlexNet DE fusion results <<<")
    n_subs = len(fmri_subjects)
    fig, axes = plt.subplots(n_subs, len(area_labels), figsize=(9 * len(area_labels), 5 * n_subs),
                              sharex=False)
    axes = np.array(axes).reshape(n_subs, len(area_labels))

    for s_idx, subject in enumerate(fmri_subjects):
        for a_idx, roi in enumerate(area_labels):
            ax = axes[s_idx, a_idx]

            for l_idx, layer in enumerate(alexnet_layers):
                d = data[layer][roi]
                if d is None or s_idx >= len(d):
                    continue
                curve = d[s_idx, :]
                ax.plot(times, curve, color=layer_colors[l_idx], lw=2.0,
                        label=layer_display_names[l_idx], zorder=3)

            if a_idx == 0:
                ax.set_ylabel(f"fMRI sub-{subject:02d}", fontweight='bold', fontsize=16)
            if s_idx == 0:
                ax.set_title(roi, fontweight='bold', fontsize=20, pad=15)

            ax.axvline(0, color='black', lw=1.5, linestyle='--', alpha=0.5)
            ax.axhline(0, color='black', lw=1.5, alpha=0.2)
            ax.set_xticks([-100, 0, 200, 400, 600])
            ax.set_xlim(-100, 600)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', labelsize=11)

            if s_idx == 0 and a_idx == len(area_labels) - 1:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


render_individual_subjects_figure("roi_layerwise_alexnet_de_fusion_individual_subjects.svg")

print(f"\nExecution complete! Total Time: {time.time() - start_time:.2f} seconds.")