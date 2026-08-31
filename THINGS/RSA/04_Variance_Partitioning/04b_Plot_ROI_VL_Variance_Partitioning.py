"""
ROI plots of the searchlight VDNN vs LLM RSA variance partitioning: averages
each partition (unique VDNN, unique LLM, shared) over each ROI's voxels, one
curve per partition per ROI subplot. Group figure (mean across fMRI subjects
with SEM ribbon) plus an individual-subject grid. 
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

from utils import get_roi_voxel_idx

start_time = time.time()

# --- Configuration ---
fmri_subjects = [1, 2, 3]
area_labels = ['V1', 'hV4', 'IT']

partitions = {
    'unique_vision': {'title': 'Unique VDNN', 'color': "#63a1cc"},
    'unique_language': {'title': 'Unique LLM', 'color': "#fd9f25"}
}

EEG_RDM_METRIC = 'pearsonr'   # 'pearsonr' | 'crossnobis' | 'decoding_accuracy'
RADIUS = 10.0
NCSNR_THRESHOLD = 0.0

data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'
base_results_dir = (
    f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/variance_partitioning/'
    f'eeg_rdm_metric-{EEG_RDM_METRIC}/radius-{RADIUS}'
)
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/THINGS/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

times = 1000*np.load(os.path.join(data_dir, 'meg_times.npy')) # pre-saved file of MEG times from -100 to +800 ms

# =============================================================================
# Aggregation: aggregated_data[partition][roi] -> (n_subjects, n_time)
# =============================================================================
print(">>> Aggregating variance partitioning ROI data <<<")
roi_voxels = {(s, roi): get_roi_voxel_idx(data_dir, s, roi, NCSNR_THRESHOLD)
              for s in fmri_subjects for roi in area_labels}

aggregated_data = {part: {roi: [] for roi in area_labels} for part in partitions}

for subject in fmri_subjects:
    path = os.path.join(base_results_dir, f'subject-{subject:02d}.npy')
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        continue
    results = np.load(path, allow_pickle=True).item()

    for part in partitions:
        arr = results[part]  # (n_time, n_voxels)
        for roi in area_labels:
            voxel_idx = roi_voxels[(subject, roi)]
            if len(voxel_idx) == 0:
                continue
            aggregated_data[part][roi].append(arr[:, voxel_idx].mean(axis=1))
    del results

for part in partitions:
    for roi in area_labels:
        aggregated_data[part][roi] = np.array(aggregated_data[part][roi])

# =============================================================================
# Console printouts
# =============================================================================
print("\n>>> Peak latency (observed; no significance stats) per ROI, per partition <<<")
for roi in area_labels:
    print(f"{roi}:")
    for part, config in partitions.items():
        d = aggregated_data[part][roi]
        if len(d) == 0:
            print(f"  {config['title']}: no data found.")
            continue
        m_group = np.mean(d, axis=0)
        print(f"  {config['title']}: peak latency = {times[np.argmax(m_group)]:.0f} ms "
              f"(mean R2 = {np.max(m_group):.5f})")


def render_group_figure(save_name):
    fig, axes = plt.subplots(1, len(area_labels), figsize=(6 * len(area_labels), 8), sharex=False)
    if len(area_labels) == 1:
        axes = [axes]

    for a_idx, roi in enumerate(area_labels):
        ax = axes[a_idx]
        all_means, all_sems = [], []

        for part, config in partitions.items():
            d = aggregated_data[part][roi]
            if len(d) == 0:
                continue
            m_group = np.mean(d, axis=0)
            sem_err = sem(d, axis=0)
            all_means.append(m_group)
            all_sems.append(sem_err)

            obs_peak = times[np.argmax(m_group)]
            ax.plot(times, m_group, color=config['color'], lw=8.0,
                    label=f"{config['title']}: {obs_peak:.0f} ms", zorder=3)
            ax.fill_between(times, m_group - sem_err, m_group + sem_err,
                            color=config['color'], alpha=0.20, zorder=2)
            ax.scatter(obs_peak, np.max(m_group), color=config['color'], s=400,
                       edgecolors='white', zorder=5)

        if all_means:
            global_max_y = max(np.max(m + s) for m, s in zip(all_means, all_sems))
            global_min_y = min(np.min(m - s) for m, s in zip(all_means, all_sems))
        else:
            global_max_y, global_min_y = 0.01, -0.005

        ax.set_title(roi, fontweight='bold', fontsize=22, pad=15)
        ax.set_xlabel('Time (ms)', fontsize=20)
        if a_idx == 0:
            ax.set_ylabel('Explained variance (R2)', fontsize=20)
        ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', lw=3, alpha=0.2)
        ax.set_xticks([-100, 0, 200, 400, 600, 800])
        ax.set_xlim(-100, 800)
        ax.set_ylim(bottom=min(global_min_y * 1.15, -0.001), top=global_max_y * 1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(3.0)
        ax.spines['bottom'].set_linewidth(3.0)
        ax.tick_params(axis='both', labelsize=16, width=3.0, length=12.0)
        ax.legend(loc='upper right', frameon=False, fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


def render_individual_subjects_figure(save_name):
    n_subs = len(fmri_subjects)
    fig, axes = plt.subplots(n_subs, len(area_labels),
                              figsize=(6 * len(area_labels), 5 * n_subs), sharex=False)
    axes = np.array(axes).reshape(n_subs, len(area_labels))

    for s_idx, subject in enumerate(fmri_subjects):
        for a_idx, roi in enumerate(area_labels):
            ax = axes[s_idx, a_idx]

            for part, config in partitions.items():
                d = aggregated_data[part][roi]
                if len(d) == 0 or s_idx >= len(d):
                    continue
                curve = d[s_idx, :]
                obs_peak = times[np.argmax(curve)]
                ax.plot(times, curve, color=config['color'], lw=2.0,
                        label=f"{config['title']}: {obs_peak:.0f}ms", zorder=3)

            if a_idx == 0:
                ax.set_ylabel(f"fMRI sub-{subject:02d}", fontweight='bold', fontsize=16)
            if s_idx == 0:
                ax.set_title(roi, fontweight='bold', fontsize=20, pad=15)

            ax.axvline(0, color='black', lw=1.5, linestyle='--', alpha=0.5)
            ax.axhline(0, color='black', lw=1.5, alpha=0.2)
            ax.set_xticks([-100, 0, 200, 400, 600, 800])
            ax.set_xlim(-100, 800)
            ax.legend(loc='upper right', frameon=False, fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


render_group_figure(f"roi_rsa_variance_partitioning_{EEG_RDM_METRIC}.svg")
render_individual_subjects_figure(
    f"roi_rsa_variance_partitioning_{EEG_RDM_METRIC}_individual_subjects.svg")

print(f"\nExecution complete! Total Time: {time.time() - start_time:.2f} seconds.")