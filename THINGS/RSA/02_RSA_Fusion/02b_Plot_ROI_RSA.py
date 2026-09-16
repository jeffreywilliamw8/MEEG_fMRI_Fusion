"""
Plots ROI correlation time courses for the THINGS MEG-fMRI RSA SEARCHLIGHT fusion, restricted to
V1, V4 (hV4), and IT
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
TMAX = 0.6  # seconds -- matches the tmax the searchlight RSA fusion was run with (141 timepoints)
EEG_RDM_METRIC = 'pearsonr'
RADIUS = 10.0  # searchlight radius (mm) -- must match what was actually run

#/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/correlations/searchlight_fusion/eeg_rdm_metric-pearsonr/radius-10.0/aggregated_results

# --- Paths ---
METADATA_DIR = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/fMRI/prepared'
meg_metadata_dir = '/scratch/jeffreykatab/Code/Encoding_Models/THINGS/MEG/prepared'
SEARCHLIGHT_DIR = (
    f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/correlations/searchlight_fusion/'
    f'eeg_rdm_metric-{EEG_RDM_METRIC}/radius-{RADIUS}/aggregated_results'
)
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/THINGS/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector ---
meg_metadata_file = os.path.join(meg_metadata_dir, 'meg_P1_metadata.npy')
meg_metadata = np.load(meg_metadata_file, allow_pickle=True).item()
raw_times = meg_metadata['meg']['times']  # seconds
times = 1000 * raw_times[raw_times <= TMAX]  # ms
n_time = len(times)
assert n_time == 141, (
    f"Expected 141 timepoints for tmax={TMAX}s (per the searchlight fusion run), got {n_time} -- "
    f"double check TMAX against what THINGS_Searchlight_MEG_fMRI_RSA_Fusion.py was actually run with."
)
XLIM = (-100, times.max())

# =============================================================================
# Data loading -- read each subject's whole-brain searchlight timecourse (already aggregated
# across time point splits), mask to each area's ROI voxels above the noise-ceiling threshold,
# and average across them (and across sub-ROIs, for areas with more than one).
# =============================================================================
print(f"Loading data for {len(area_labels)} areas across {len(subject_list)} subjects...")

area_data_list = []
for a_idx, area_rois in enumerate(areas):
    area_label = area_labels[a_idx]
    print(f"\n--- Processing Area: {area_label} ---")
    subject_matrices = []

    for subject in subject_list:
        meta_path = os.path.join(METADATA_DIR, f'fmri_{subject}_metadata.npy')
        if not os.path.exists(meta_path):
            print(f"    [!] Missing metadata: {meta_path}")
            continue

        metadata = np.load(meta_path, allow_pickle=True).item()
        noise_ceilings = metadata['encoding_model']['noise_ceiling_testset']

        data_path = os.path.join(SEARCHLIGHT_DIR, f'subject-{subject}', f'subject-{subject}_timecourse.npy')
        if not os.path.exists(data_path):
            print(f"    [!] Missing: {data_path}")
            continue

        try:
            data = np.load(data_path)  # (n_time, n_voxels) -- already time-truncated at aggregation
            if data.shape[0] != n_time:
                print(f"    [!] Warning: sub-{subject} timecourse has {data.shape[0]} timepoints, "
                      f"expected {n_time} -- skipping.")
                continue

            roi_averages = []
            for roi_name in area_rois:
                if roi_name not in metadata['roi']:
                    continue

                idx = metadata['roi'][roi_name]
                roi_data = data[:, idx]
                roi_nc = noise_ceilings[idx].astype(np.float32)

                # Filter by noise ceiling
                valid_voxels = np.where(roi_nc > NCSNR_THRESHOLD)[0]

                if len(valid_voxels) > 0:
                    # Average across voxels (axis 1) to get shape (n_time,)
                    roi_tc = np.mean(roi_data[:, valid_voxels], axis=1)
                    roi_averages.append(roi_tc)

            if roi_averages:
                # Average across multiple ROIs within the same area
                subj_avg = np.mean(roi_averages, axis=0)

                if subj_avg.ndim == 1 and subj_avg.shape == (n_time,):
                    subject_matrices.append(subj_avg)
                else:
                    print(f"    [!] Error: Sub-{subject} unexpected shape: {subj_avg.shape}")

        except Exception as e:
            print(f"    [!] Error processing sub-{subject}: {e}")

    if subject_matrices:
        area_array = np.stack(subject_matrices)  # (n_subjects, n_time)
        print(f"  [=>] SUCCESS: Area {area_label} matrix shape: {area_array.shape}")
        area_data_list.append(area_array)
    else:
        print(f"  [=>] FAILURE: No valid data for Area {area_label}")
        area_data_list.append(np.zeros((len(subject_list), n_time)))

print("\n--- Data Loading Complete. Proceeding to Bootstrap/Plotting... ---")


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
# Plotting -- canonical group figure (bootstrap 95% CI ribbon + peak marker with bootstrap
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

        # Bootstrap peak latency CI: resample subjects with replacement, re-find the argmax of
        # the resampled group mean each time -- uncertainty in the peak's TIME INDEX, distinct
        # from the ribbon CI above (which is about the correlation VALUE at each timepoint).
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

    ax.set_title('MEG-fMRI Searchlight RSA Fusion (ROI)', fontweight='bold', fontsize=28, pad=40)
    ax.set_xlabel('Time (ms)', fontsize=28)
    ax.set_ylabel("Spearman's R", fontsize=28)
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


plot_roi_results(area_data_list, "MEG-fMRI Searchlight RSA Fusion (ROI)", "roi_searchlight_rsa_fusion.svg")
plot_roi_results_individual_subjects(area_data_list, "roi_searchlight_rsa_fusion_individual_subjects.svg")

print(f"\nExecution complete! Total Time: {time.time() - start_time:.2f} seconds.")