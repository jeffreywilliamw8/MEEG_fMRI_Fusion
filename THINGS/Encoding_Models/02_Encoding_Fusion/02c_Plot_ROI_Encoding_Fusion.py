"""
Plots ROI correlation time courses for the THINGS-MEG-fMRI encoding fusion.
Assumes ROI results have already been aggregated to whole-ROI correlation time courses,
shape = (n_time_points, n_voxels)

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import sem
from utils import get_meg_times
import time

start_time = time.time()

# --- Configuration ---
fmri_subjects = [1, 2, 3]

area_labels = ['V1', 'hV4', 'IT']

area_colors = [
    "#480758",  # V1 (Deep Purple)
    "#468fc3",  # hV4 (Steel Blue)
    "#ea8e16",  # IT (Orange)
]

# Pathing
base_results_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/correlations/roi_encoding_fusion'
data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/THINGS/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector ---
times = get_meg_times(data_dir)

# --- Data Aggregation ---
# area_data_list[i]: (n_subjects, n_time) -- voxel-averaged correlations for ROI i
area_data_list = []

print(">>> Aggregating ROI Data <<<")

for roi in area_labels:
    subject_roi_corrs = []

    for subject in fmri_subjects:
        path = os.path.join(base_results_dir, f'fmri_sub-{subject:02d}', f'roi-{roi}', 'correlations.npy')
        try:
            data = np.load(path) 
        except FileNotFoundError:
            print(f"Missing: {path}")
            continue

        print(f"Loaded correlations for fMRI sub-{subject:02d}, ROI {roi}: shape = {data.shape}")
        subject_roi_corrs.append(np.mean(data, axis=1))  # average across voxels -> (n_time,)

    area_data_list.append(np.array(subject_roi_corrs))  # (n_subjects, n_time)


def sem_across_subjects(area_data):
    """
    Standard error of the across-subject mean at each timepoint. This is the
    error metric used for the shaded area around each curve
    """
    return sem(area_data, axis=0)


# --- Plotting ---
def plot_roi_results(data_list, title, filename):
    plt.figure(figsize=(14, 8))
    ax = plt.gca()

    all_means = [np.mean(d, axis=0) for d in data_list]
    all_sems = [sem_across_subjects(d) for d in data_list]
    global_max_y = max([np.max(m + s) for m, s in zip(all_means, all_sems)])
    global_min_y = min([np.min(m - s) for m, s in zip(all_means, all_sems)])

    print("\n>>> Peak latency (observed; no bootstrap CI -- too few subjects) per ROI <<<")

    for i, area_data in enumerate(data_list):
        m_group = np.mean(area_data, axis=0)
        sem_err = sem_across_subjects(area_data) 
        color = area_colors[i]

        obs_peak = times[np.argmax(m_group)]
        print(f"{area_labels[i]}: peak latency = {obs_peak:.0f} ms")

        leg_text = f"{area_labels[i]}: {obs_peak:.0f} ms"

        ax.plot(times, m_group, color=color, lw=12.0, label=leg_text, zorder=3)
        ax.fill_between(times, m_group - sem_err, m_group + sem_err, color=color, alpha=0.20, zorder=2)

        # Peak marker
        peak_val = np.max(m_group)
        ax.scatter(obs_peak, peak_val, color=color, s=600, edgecolors='white', zorder=5)

    # Styling
    ax.set_title(f'{title}', fontweight='bold', fontsize=28, pad=40)
    ax.set_xlabel('Time (ms)', fontsize=28)
    ax.set_ylabel("Pearson's r", fontsize=28)
    ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=3, alpha=0.2)
    ax.set_xlim(-100, 600)

    # No staggered significance lanes below y=0 (no significance stats here) --
    # bottom limit is just a small margin below the lowest SEM band.
    bottom_limit = min(global_min_y - 0.02, -0.02)
    ax.set_ylim(bottom=bottom_limit, top=global_max_y * 1.15)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=18)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.tick_params(axis='both', labelsize=26, width=3.0, length=12.0)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")


# =============================================================================
# Individual-subject companion plot: no stats, no CI (nothing to bootstrap/permute
# over a single subject), just each subject's own raw ROI curves overlaid in their
# own panel, with a legend giving each ROI's own peak latency for that subject.
# Purely a visual check for whether the group-level pattern holds up subject by
# subject.
# =============================================================================
def plot_roi_results_individual_subjects(data_list, filename, n_cols=3):
    n_subs = len(fmri_subjects)
    n_rows = int(np.ceil(n_subs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), sharex=False)
    axes = np.array(axes).reshape(-1)

    print("\n>>> Peak latency per ROI, per subject (individual-subject plot) <<<")
    for s_idx, subject in enumerate(fmri_subjects):
        ax = axes[s_idx]
        print(f"fMRI sub-{subject:02d}:")
        for a_idx, area_data in enumerate(data_list):
            if s_idx >= len(area_data):
                continue
            curve = area_data[s_idx, :]
            obs_peak = times[np.argmax(curve)]
            print(f"  {area_labels[a_idx]}: peak latency = {obs_peak:.0f}ms")
            leg_text = f"{area_labels[a_idx]}: {obs_peak:.0f}ms"
            ax.plot(times, curve, color=area_colors[a_idx], lw=2.5, label=leg_text, zorder=3)

        ax.set_title(f'fMRI Participant-{subject:02d}', fontweight='bold', fontsize=16, pad=10)
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
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


# Run the plots
plot_roi_results(area_data_list, "MEG-fMRI Encoding Fusion (ROI)", "roi_encoding_fusion.svg")
plot_roi_results_individual_subjects(area_data_list, "roi_encoding_fusion_individual_subjects.svg")

print(f"Execution complete! Total Time: {time.time() - start_time:.2f} seconds.")