"""
Plots ROI correlation time courses for the THINGS MEG-fMRI encoding fusion, restricted to V1,
V4 (hV4), and IT. Data loading -- raw meg2fmri_fusion correlation time courses, per subject,
masked to each area's ROI vertices above the noise-ceiling threshold and averaged across them --
is adapted verbatim from an earlier diagnostic script. The plotting logic follows the canonical
group/individual-subject ROI plot style used elsewhere in this project, with one change: the
group shaded_area is a 95% percentile-bootstrap CI across subjects (not SEM), and peak latency gets
its own bootstrap CI reported in the legend -- matching the NSD/BMD ROI plots. No significance
stats here (only 3 fMRI subjects).
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()

# --- Configuration ---
subject_list = ['01', '02', '03']
area_labels = ['V1', 'V4', 'IT']
areas = [['V1'], ['hV4'], ['IT']]  # metadata ROI key(s) underlying each displayed area label
area_colors = ["#480758", "#468fc3", "#ea8e16"]  # V1, V4, IT

N_BOOTSTRAPS = 10000
NCSNR_THRESHOLD = 20.0
target_len = 141 # 141 time points -> -100 to 600ms

# --- Paths ---
BASE_DIR = '/home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models'
METADATA_DIR = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/fMRI/prepared'
meg_metadata_dir = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/MEG/prepared'
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/THINGS/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

#  f'/home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/correlations/meg2fmri_fusion/sub-{subject}_correlation_time_courses.npy'

# --- Time Vector ---
meg_metadata_file = os.path.join(meg_metadata_dir, 'meg_P1_metadata.npy')
meg_metadata = np.load(meg_metadata_file, allow_pickle=True).item()
times = 1000 * meg_metadata['meg']['times']
times = times[:target_len]
XLIM = (-100, times.max())

# =============================================================================
# Data loading 
# =============================================================================
print(">>> Aggregating ROI Data <<<")

area_data_list = []
for area_rois in areas:
    subject_matrices = []
    for subject in subject_list:
        metadata = np.load(f'{METADATA_DIR}/fmri_{subject}_metadata.npy', allow_pickle=True).item()
        data = np.load(f'{BASE_DIR}/correlations/meg2fmri_fusion/sub-{subject}_correlation_time_courses.npy')[:target_len, :]

        roi_averages = []
        for roi_name in area_rois:
            idx = metadata['roi'][roi_name]
            nc = metadata['encoding_model']['noise_ceiling_testset'][idx]
            valid = np.where(nc > NCSNR_THRESHOLD)[0]
            roi_averages.append(np.mean(data[:, idx][:, valid], axis=1))

        subject_matrices.append(np.mean(roi_averages, axis=0))
    area_data_list.append(np.stack(subject_matrices))


def ci95_across_subjects(area_data, n_bootstraps=N_BOOTSTRAPS):
    """
    95% confidence interval of the across-subject mean at each timepoint, via percentile
    bootstrap: resample subjects with replacement n_bootstraps times, recompute the mean
    across the resampled subjects at each timepoint, then take the 2.5th/97.5th percentiles
    of that bootstrap distribution.
    """
    n_subs = area_data.shape[0]
    boot_means = np.zeros((n_bootstraps, area_data.shape[1]))
    for i in range(n_bootstraps):
        res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
        boot_means[i] = np.mean(area_data[res_idx], axis=0)
    low, high = np.percentile(boot_means, [2.5, 97.5], axis=0)
    return low, high


# =============================================================================
# Plotting -- canonical group figure (bootstrap 95% CI shaded_area + peak marker with bootstrap
# peak-latency CI in the legend, no significance stats) plus individual-subject companion figure
# =============================================================================
def plot_roi_results(data_list, title, filename):
    plt.figure(figsize=(14, 8))
    ax = plt.gca()

    all_cis = [ci95_across_subjects(d) for d in data_list]  # each: (low, high) arrays
    global_max_y = max(np.max(high) for _, high in all_cis)
    global_min_y = min(np.min(low) for low, _ in all_cis)

    print("\n>>> Peak latency (95% CI, bootstrap over subjects) per ROI <<<")

    for i, area_data in enumerate(data_list):
        n_subs = area_data.shape[0]
        m_group = np.mean(area_data, axis=0)
        ci_low, ci_high = all_cis[i]
        color = area_colors[i]

        # Bootstrap peak latency CI
        boot_peaks = []
        for _ in range(N_BOOTSTRAPS):
            res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
            boot_peaks.append(times[np.argmax(np.mean(area_data[res_idx], axis=0))])
        low, high = np.percentile(boot_peaks, [2.5, 97.5])
        obs_peak = times[np.argmax(m_group)]

        print(f"{area_labels[i]}: peak latency = {obs_peak:.0f} ms [95% CI: {low:.0f}-{high:.0f} ms]")

        leg_text = f"{area_labels[i]}: {obs_peak:.0f} ms [{low:.0f}-{high:.0f} ms]"
        ax.plot(times, m_group, color=color, lw=6.0, label=leg_text, zorder=3)
        ax.fill_between(times, ci_low, ci_high, color=color, alpha=0.20, zorder=2)
        ax.scatter(obs_peak, np.max(m_group), color=color, s=600, edgecolors='white', zorder=5)

    ax.set_title('MEG-fMRI Encoding Fusion (ROI)', fontweight='bold', fontsize=28, pad=40)
    ax.set_xlabel('Time (ms)', fontsize=28)
    ax.set_ylabel("Pearson's r", fontsize=28)
    ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=3, alpha=0.2)
    ax.set_xlim(*XLIM)
    ax.set_ylim(bottom=min(global_min_y - 0.02, -0.02), top=global_max_y * 1.15)
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=18)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.tick_params(axis='both', labelsize=26, width=3.0, length=12.0)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


def plot_roi_results_individual_subjects(data_list, filename, n_cols=3):
    n_subs = len(subject_list)
    n_rows = int(np.ceil(n_subs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), sharex=False)
    axes = np.array(axes).reshape(-1)

    print("\n>>> Peak latency per ROI, per subject (individual-subject plot) <<<")
    for s_idx, subject in enumerate(subject_list):
        ax = axes[s_idx]
        print(f"fMRI sub-{subject}:")
        for a_idx, area_data in enumerate(data_list):
            if s_idx >= len(area_data):
                continue
            curve = area_data[s_idx, :]
            obs_peak = times[np.argmax(curve)]
            print(f"  {area_labels[a_idx]}: peak latency = {obs_peak:.0f}ms")
            ax.plot(times, curve, color=area_colors[a_idx], lw=2.5,
                    label=f"{area_labels[a_idx]}: {obs_peak:.0f}ms", zorder=3)

        ax.set_title(f'fMRI sub-{subject}', fontweight='bold', fontsize=16, pad=10)
        ax.axvline(0, color='black', lw=1.5, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', lw=1.5, alpha=0.2)
        ax.set_xlim(*XLIM)
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


plot_roi_results(area_data_list, "MEG-fMRI Encoding Fusion (ROI)", "roi_meg2fmri_encoding_fusion.svg")
plot_roi_results_individual_subjects(area_data_list, "roi_meg2fmri_encoding_fusion_individual_subjects.svg")

print(f"\nExecution complete! Total Time: {time.time() - start_time:.2f} seconds.")