"""
Plots the ROI-level VDNN vs LLM partial correlation results from the SFEF
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

from utils import get_meg_times

start_time = time.time()

# --- Configuration ---
fmri_subjects = [1, 2, 3]
halves = [1, 2]

area_labels = ['V1', 'hV4', 'IT']

partitions = {
    'vision_partial_correlation': {'title': 'VDNN', 'color': "#63a1cc"},    # steel blue
    'language_partial_correlation': {'title': 'LLM', 'color': "#fd9f25"},  # orange
}

# Pathing -- must match THINGS_ROI_SFEF_Partial_Correlation.py's
# --corrs_save_dir default and save-directory layout.
base_results_dir = ('/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/'
                     'correlations/stimulus_feature_encoding_fusion/partial_correlation/roi')
data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/THINGS/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector ---
times = get_meg_times(data_dir)
n_timepoints = len(times)


# =============================================================================
# Data Aggregation: average each subject's --half 1 and --half 2 runs
# (SFEF's "even/odd" equivalent -- see module docstring).
# =============================================================================
def load_subject_roi_curve(subject, roi, part_key):
    curves = []
    for half in halves:
        path = os.path.join(base_results_dir, f'fmri_sub-{subject:02d}', f'roi-{roi}',
                             f'half-{half}', 'partial_correlations.npy')
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True).item()
            curves.append(d[part_key].mean(axis=1))  # average across ROI voxels -> (n_time,)
        else:
            print(f"  Missing: {path}")

    if len(curves) == 0:
        return None
    if len(curves) == 1:
        print(f"  NOTE: only one half found for sub-{subject:02d} / {roi} / {part_key} -- "
              f"using it alone (no half-1/half-2 averaging).")
    return np.mean(curves, axis=0)


print(">>> Aggregating ROI SFEF partial correlation data <<<")
# aggregated_data[part_key][roi] -> (n_subjects_found, n_time)
aggregated_data = {part: {} for part in partitions}
for part_key in partitions:
    for roi in area_labels:
        subject_curves = []
        for subject in fmri_subjects:
            curve = load_subject_roi_curve(subject, roi, part_key)
            if curve is not None:
                subject_curves.append(curve)
        aggregated_data[part_key][roi] = np.array(subject_curves)


def sem_across_subjects(area_data):
    """SEM ribbon -- uncertainty in the correlation value at each timepoint."""
    return sem(area_data, axis=0)


# =============================================================================
# Console report: observed peak latency per ROI/category (no stats, no CI).
# =============================================================================
print("\n>>> Peak latency (observed; no significance stats) per ROI, per category <<<")
for roi in area_labels:
    print(f"{roi}:")
    for part_key, config in partitions.items():
        d = aggregated_data[part_key][roi]
        if d is None or len(d) == 0:
            print(f"  {config['title']}: no data found, skipping.")
            continue
        m_group = np.mean(d, axis=0)
        peak = times[np.argmax(m_group)]
        print(f"  {config['title']}: peak latency = {peak:.0f} ms (mean r = {np.max(m_group):.4f})")


# =============================================================================
# Group figure: one subplot per ROI, VDNN + LLM mean curves with SEM ribbons.
# =============================================================================
def render_figure(save_name):
    print("\n>>> Plotting group-average SFEF VDNN/LLM partial correlation results <<<")
    fig, axes = plt.subplots(1, len(area_labels), figsize=(9 * len(area_labels), 8), sharex=False)
    if len(area_labels) == 1:
        axes = [axes]

    for a_idx, roi in enumerate(area_labels):
        ax = axes[a_idx]
        all_means, all_sems = [], []

        for part_key, config in partitions.items():
            d = aggregated_data[part_key][roi]
            if d is None or len(d) == 0:
                continue
            m_group = np.mean(d, axis=0)
            sem_err = sem_across_subjects(d)
            all_means.append(m_group)
            all_sems.append(sem_err)

            obs_peak = times[np.argmax(m_group)]
            leg_text = f"{config['title']}: {obs_peak:.0f} ms"

            ax.plot(times, m_group, color=config['color'], lw=8.0, label=leg_text, zorder=3)
            ax.fill_between(times, m_group - sem_err, m_group + sem_err,
                            color=config['color'], alpha=0.20, zorder=2)

            peak_val = np.max(m_group)
            ax.scatter(obs_peak, peak_val, color=config['color'], s=400,
                       edgecolors='white', zorder=5)

        if all_means:
            global_max_y = max(np.max(m + s) for m, s in zip(all_means, all_sems))
            global_min_y = min(np.min(m - s) for m, s in zip(all_means, all_sems))
        else:
            global_max_y, global_min_y = 0.06, -0.02

        ax.set_title(roi, fontweight='bold', fontsize=22, pad=15)
        ax.set_xlabel('Time (ms)', fontsize=20)
        if a_idx == 0:
            ax.set_ylabel("Partial r", fontsize=20)
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

        ax.legend(loc='upper right', frameon=False, fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


render_figure("roi_sfef_vdnn_llm_partial_correlation.svg")


# =============================================================================
# Individual-subject figure: rows = subjects, columns = ROIs
# =============================================================================
def render_individual_subjects_figure(save_name):
    print("\n>>> Plotting individual-subject SFEF VDNN/LLM partial correlation results <<<")
    n_subs = len(fmri_subjects)
    fig, axes = plt.subplots(n_subs, len(area_labels), figsize=(9 * len(area_labels), 5 * n_subs),
                              sharex=False)
    axes = np.array(axes).reshape(n_subs, len(area_labels))

    for s_idx, subject in enumerate(fmri_subjects):
        for a_idx, roi in enumerate(area_labels):
            ax = axes[s_idx, a_idx]

            for part_key, config in partitions.items():
                d = aggregated_data[part_key][roi]
                if d is None or s_idx >= len(d):
                    continue
                curve = d[s_idx, :]
                obs_peak = times[np.argmax(curve)]
                leg_text = f"{config['title']}: {obs_peak:.0f}ms"
                ax.plot(times, curve, color=config['color'], lw=2.0, label=leg_text, zorder=3)

            if a_idx == 0:
                ax.set_ylabel(f"fMRI sub-{subject:02d}", fontweight='bold', fontsize=16)
            if s_idx == 0:
                ax.set_title(roi, fontweight='bold', fontsize=20, pad=15)

            ax.axvline(0, color='black', lw=1.5, linestyle='--', alpha=0.5)
            ax.axhline(0, color='black', lw=1.5, alpha=0.2)
            ax.set_xticks([-100, 0, 200, 400, 600])
            ax.set_xlim(-100, 600)
            ax.legend(loc='upper right', frameon=False, fontsize=11)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


render_individual_subjects_figure("roi_sfef_vdnn_llm_partial_correlation_individual_subjects.svg")

print(f"\nExecution complete! Total Time: {time.time() - start_time:.2f} seconds.")
