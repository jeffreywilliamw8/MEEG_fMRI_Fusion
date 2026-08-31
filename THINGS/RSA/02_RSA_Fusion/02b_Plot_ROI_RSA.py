"""
ROI plots of the searchlight MEG-fMRI RSA fusion: averages the per-voxel
searchlight correlations over each ROI's voxels, one curve per ROI. Group
figure (mean across fMRI subjects with SEM ribbon) plus an individual-subject
companion figure.
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
area_colors = ["#480758", "#468fc3", "#ea8e16"]

EEG_RDM_METRIC = 'pearsonr'   # 'pearsonr' | 'crossnobis' | 'decoding_accuracy'
RADIUS = 10.0
NCSNR_THRESHOLD = 0.0

data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'
base_results_dir = (
    f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/correlations/searchlight_fusion/aggregated_results'
    f'eeg_rdm_metric-{EEG_RDM_METRIC}/radius-{RADIUS}'
)
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/THINGS/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

times = 1000*np.load(os.path.join(data_dir, 'meg_times.npy')) # pre-saved file of MEG times from -100 to +800 ms
n_timepoints = len(times)

# =============================================================================
# Aggregation: one (n_time, n_voxels) matrix per subject, assembled from the
# per-timepoint files, then averaged within each ROI's voxels.
# =============================================================================
print(">>> Aggregating searchlight fusion ROI data <<<")
area_data_list = [[] for _ in area_labels]

for subject in fmri_subjects:
    subject_dir = os.path.join(base_results_dir, f'subject-{subject:02d}')

    timecourse = None
    missing = 0
    for t in range(n_timepoints):
        path = os.path.join(subject_dir, f'time_point_{t:04d}.npy')
        if not os.path.exists(path):
            missing += 1
            continue
        corrs = np.load(path)
        if timecourse is None:
            timecourse = np.full((n_timepoints, corrs.shape[0]), np.nan, dtype=np.float32)
        timecourse[t] = corrs

    if timecourse is None:
        print(f"  fMRI sub-{subject:02d}: no timepoint files found, skipping.")
        continue
    if missing:
        print(f"  fMRI sub-{subject:02d}: {missing}/{n_timepoints} timepoint files missing.")

    for a_idx, roi in enumerate(area_labels):
        voxel_idx = get_roi_voxel_idx(data_dir, subject, roi, NCSNR_THRESHOLD)
        if len(voxel_idx) == 0:
            print(f"  fMRI sub-{subject:02d}, {roi}: no voxels above threshold, skipping.")
            continue
        area_data_list[a_idx].append(np.nanmean(timecourse[:, voxel_idx], axis=1))
        print(f"  fMRI sub-{subject:02d}, {roi}: {len(voxel_idx)} voxels")

    del timecourse

area_data_list = [np.array(d) for d in area_data_list]

# =============================================================================
# Console printouts
# =============================================================================
print("\n>>> Peak latency per ROI <<<")
for a_idx, roi in enumerate(area_labels):
    d = area_data_list[a_idx]
    if len(d) == 0:
        print(f"{roi}: no data found.")
        continue
    m_group = np.mean(d, axis=0)
    print(f"{roi}: peak latency = {times[np.argmax(m_group)]:.0f} ms "
          f"(mean r = {np.max(m_group):.4f})")


def render_group_figure(save_name):
    plt.figure(figsize=(14, 8))
    ax = plt.gca()

    all_means, all_sems = [], []
    for a_idx, d in enumerate(area_data_list):
        if len(d) == 0:
            continue
        m_group = np.mean(d, axis=0)
        sem_err = sem(d, axis=0)
        all_means.append(m_group)
        all_sems.append(sem_err)

        obs_peak = times[np.argmax(m_group)]
        ax.plot(times, m_group, color=area_colors[a_idx], lw=12.0,
                label=f"{area_labels[a_idx]}: {obs_peak:.0f} ms", zorder=3)
        ax.fill_between(times, m_group - sem_err, m_group + sem_err,
                        color=area_colors[a_idx], alpha=0.20, zorder=2)
        ax.scatter(obs_peak, np.max(m_group), color=area_colors[a_idx], s=600,
                   edgecolors='white', zorder=5)

    global_max_y = max(np.max(m + s) for m, s in zip(all_means, all_sems))
    global_min_y = min(np.min(m - s) for m, s in zip(all_means, all_sems))

    ax.set_title('MEG-fMRI Searchlight RSA Fusion (ROI)', fontweight='bold', fontsize=28, pad=40)
    ax.set_xlabel('Time (ms)', fontsize=28)
    ax.set_ylabel("Spearman's r", fontsize=28)
    ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=3, alpha=0.2)
    ax.set_xlim(-100, 800)
    ax.set_ylim(bottom=min(global_min_y - 0.02, -0.02), top=global_max_y * 1.15)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.tick_params(axis='both', labelsize=26, width=3.0, length=12.0)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


def render_individual_subjects_figure(save_name, n_cols=3):
    n_subs = len(fmri_subjects)
    n_rows = int(np.ceil(n_subs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), sharex=False)
    axes = np.array(axes).reshape(-1)

    for s_idx, subject in enumerate(fmri_subjects):
        ax = axes[s_idx]
        for a_idx, d in enumerate(area_data_list):
            if len(d) == 0 or s_idx >= len(d):
                continue
            curve = d[s_idx, :]
            obs_peak = times[np.argmax(curve)]
            ax.plot(times, curve, color=area_colors[a_idx], lw=2.5,
                    label=f"{area_labels[a_idx]}: {obs_peak:.0f}ms", zorder=3)

        ax.set_title(f'fMRI sub-{subject:02d}', fontweight='bold', fontsize=16, pad=10)
        ax.axvline(0, color='black', lw=1.5, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', lw=1.5, alpha=0.2)
        ax.set_xlim(-100, 600)
        ax.legend(loc='upper right', frameon=False, fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=11)

    for extra_ax in axes[n_subs:]:
        extra_ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


render_group_figure(f"roi_searchlight_rsa_fusion_{EEG_RDM_METRIC}.svg")
render_individual_subjects_figure(
    f"roi_searchlight_rsa_fusion_{EEG_RDM_METRIC}_individual_subjects.svg")

print(f"\nExecution complete! Total Time: {time.time() - start_time:.2f} seconds.")