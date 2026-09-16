"""
ROI plots of the layerwise AlexNet searchlight commonality analysis: averages
the per-voxel commonality over each ROI's voxels, one curve per AlexNet layer
per ROI subplot. Group figure (mean across fMRI subjects) plus an
individual-subject grid.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from utils import get_roi_voxel_idx

start_time = time.time()

# --- Configuration ---
fmri_subjects = [1, 2, 3]
area_labels = ['V1', 'hV4', 'IT']

alexnet_layers = [
    'features.2', 'features.5', 'features.7', 'features.9', 'features.12',
    'classifier.2', 'classifier.5', 'classifier.6'
]
layer_display_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC6', 'FC7', 'FC8']
layer_colors = ["#0C076E", "#5121A0", "#4C95BA", "#16B28B",
                "#D4AC0D", "#D17C20", "#B23492", "#CE1414"]

EEG_RDM_METRIC = 'pearsonr'   # 'pearsonr' | 'crossnobis' | 'decoding_accuracy'
METADATA_DIR = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/fMRI/prepared'
RADIUS = 10.0
NCSNR_THRESHOLD = 20.0

data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'
base_results_dir = (
    f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/commonality_analysis/'
    f'layerwise_alexnet/eeg_rdm_metric-{EEG_RDM_METRIC}/radius-{RADIUS}'
)
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/THINGS/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

meg_metadata_file = os.path.join(METADATA_DIR, 'meg_P1_metadata.npy')
meg_metadata = np.load(meg_metadata_file, allow_pickle=True).item()
raw_times = meg_metadata['meg']['times']  # seconds
times = 1000 * raw_times[raw_times <= 0.6]  # -100 to +600 ms
# =============================================================================
# Aggregation: data[layer][roi] -> (n_subjects, n_time)
# =============================================================================
print(">>> Aggregating commonality ROI data across AlexNet layers <<<")
roi_voxels = {(s, roi): get_roi_voxel_idx(data_dir, s, roi, NCSNR_THRESHOLD)
              for s in fmri_subjects for roi in area_labels}

data = {layer: {roi: [] for roi in area_labels} for layer in alexnet_layers}

for subject in fmri_subjects:
    for layer in alexnet_layers:
        path = os.path.join(base_results_dir, f'subject-{subject:02d}', f'layer-{layer}.npy')
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
        r2 = np.load(path)  # (n_time, n_voxels)

        for roi in area_labels:
            voxel_idx = roi_voxels[(subject, roi)]
            if len(voxel_idx) == 0:
                continue
            data[layer][roi].append(r2[:, voxel_idx].mean(axis=1))
        del r2

for layer in alexnet_layers:
    for roi in area_labels:
        data[layer][roi] = np.array(data[layer][roi])

# =============================================================================
# Console printouts
# =============================================================================
print("\n>>> Peak latency per ROI, per layer <<<")
for roi in area_labels:
    print(f"{roi}:")
    for l_idx, layer in enumerate(alexnet_layers):
        d = data[layer][roi]
        if len(d) == 0:
            print(f"  {layer_display_names[l_idx]}: no data found.")
            continue
        m_group = np.mean(d, axis=0)
        print(f"  {layer_display_names[l_idx]}: peak latency = {times[np.argmax(m_group)]:.0f} ms "
              f"(mean commonality = {np.max(m_group):.5f})")


def render_group_figure(save_name):
    fig, axes = plt.subplots(1, len(area_labels), figsize=(9 * len(area_labels), 8), sharex=False)
    if len(area_labels) == 1:
        axes = [axes]

    for a_idx, roi in enumerate(area_labels):
        ax = axes[a_idx]
        all_means = []

        for l_idx, layer in enumerate(alexnet_layers):
            d = data[layer][roi]
            if len(d) == 0:
                continue
            m_group = np.mean(d, axis=0)
            all_means.append(m_group)
            ax.plot(times, m_group, color=layer_colors[l_idx], lw=3.0,
                    label=layer_display_names[l_idx], zorder=3)

        if all_means:
            global_max_y = max(np.max(m) for m in all_means)
            global_min_y = min(np.min(m) for m in all_means)
        else:
            global_max_y, global_min_y = 0.01, -0.005

        ax.set_title(roi, fontweight='bold', fontsize=22, pad=15)
        ax.set_xlabel('Time (ms)', fontsize=20)
        if a_idx == 0:
            ax.set_ylabel('R2 score', fontsize=20)
        ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', lw=3, alpha=0.2)
        ax.set_xticks([-100, 0, 200, 400, 600])
        ax.set_xlim(-100, 600)
        ax.set_ylim(bottom=min(global_min_y * 1.15, -0.001), top=global_max_y * 1.15)
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


def render_individual_subjects_figure(save_name):
    n_subs = len(fmri_subjects)
    fig, axes = plt.subplots(n_subs, len(area_labels),
                              figsize=(9 * len(area_labels), 5 * n_subs), sharex=False)
    axes = np.array(axes).reshape(n_subs, len(area_labels))

    for s_idx, subject in enumerate(fmri_subjects):
        for a_idx, roi in enumerate(area_labels):
            ax = axes[s_idx, a_idx]

            for l_idx, layer in enumerate(alexnet_layers):
                d = data[layer][roi]
                if len(d) == 0 or s_idx >= len(d):
                    continue
                ax.plot(times, d[s_idx, :], color=layer_colors[l_idx], lw=2.0,
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


render_group_figure(f"roi_layerwise_alexnet_commonality_{EEG_RDM_METRIC}.svg")
render_individual_subjects_figure(
    f"roi_layerwise_alexnet_commonality_{EEG_RDM_METRIC}_individual_subjects.svg")

print(f"\nExecution complete! Total Time: {time.time() - start_time:.2f} seconds.")