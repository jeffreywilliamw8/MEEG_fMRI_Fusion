"""
This script aggregates the ROI-wise EEG-fMRI encoding fusion results, computes statistics, and generates plots for each ROI (V1, V4, PPA)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utils import sign_permutation_cluster_test, load_ncsnr, load_roi_indices
import time

# Start time
start_time = time.time()

# --- Configuration ---
subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Define the grouping: Area -> list of sub-ROI filenames
# Define the grouping hierarchy: Group Label -> List of sub-ROI file names
roi_groups = {
    'V1': ['V1v', 'V1d'],
    'hV4': ['hV4'],
    'PPA': ['PPA']
}

area_labels = ['V1', 'hV4', 'PPA']

area_colors = [
    "#480758",  # V1 (Deep Purple)
    "#468fc3",  # hV4 (Steel Blue)
    "#ea8e16" ,  # PPA (Orange)
]



N_BOOTSTRAPS = 10000
NCSNR_THRESHOLD = 20.0
N_VERT_PER_HEMI = 163842
EEG_TEMPORAL_RESOLUTION = 2

# Pathing -- whole-brain per-vertex correlation time courses 
base_results_dir = f'/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/encoding_fusion/whole_brain/eeg_temporal_resolution-{EEG_TEMPORAL_RESOLUTION}ms'
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


# Loading EEG time points (in ms)
metadata_dir = os.path.join('/scratch/giffordale95/projects/eeg_moments_dataset', 'derivatives', 'eeg', 'sub-01', 'sub-01_eeg_metadata.npy')
#metadata_dir = os.path.join('/scratch/giffordale95/projects/eeg_moments_dataset/derivatives/eeg/sub-01/sub-01_eeg_metadata.npy')

times = 1000*np.load(metadata_dir, allow_pickle=True).item()['times']
if EEG_TEMPORAL_RESOLUTION == 4:
    times = times[::2]

# --- Per-subject, per-sub-ROI vertex masks (ROI + noise-ceiling) ---
print(f">>> Building per-subject, per-sub-ROI masks (ROI + noise-ceiling >= {NCSNR_THRESHOLD}) <<<")
all_sub_rois = sorted(set(sub_roi for sub_rois in roi_groups.values() for sub_roi in sub_rois))

subject_masks = {}  # subject -> sub_roi -> boolean mask over concatenated [lh, rh] vertices
for subject in subject_list:
    ncsnr_lh = load_ncsnr(subject, 'left')
    ncsnr_rh = load_ncsnr(subject, 'right')

    subject_masks[subject] = {}
    for sub_roi in all_sub_rois:
        roi_idx_lh = load_roi_indices(subject, 'left', sub_roi)
        roi_mask_lh = np.zeros(N_VERT_PER_HEMI, dtype=bool)
        roi_mask_lh[roi_idx_lh] = True

        roi_idx_rh = load_roi_indices(subject, 'right', sub_roi)
        roi_mask_rh = np.zeros(N_VERT_PER_HEMI, dtype=bool)
        roi_mask_rh[roi_idx_rh] = True

        combined_lh = roi_mask_lh & (ncsnr_lh >= NCSNR_THRESHOLD)
        combined_rh = roi_mask_rh & (ncsnr_rh >= NCSNR_THRESHOLD)
        subject_masks[subject][sub_roi] = np.concatenate([combined_lh, combined_rh])


# --- Data Aggregation ---
area_data_list = []  # Will store [Area][Subject, Time]

print(">>> Aggregating Whole-Brain Data (masking vertices per area, averaging across them) <<<")

for area in tqdm(area_labels):
    sub_rois = roi_groups[area]
    subject_area_corrs = []

    for subject in subject_list:
        path_lh = os.path.join(base_results_dir, f'subject-{subject}', 'correlations_left.npy')
        path_rh = os.path.join(base_results_dir, f'subject-{subject}', 'correlations_right.npy')

        data_lh = np.load(path_lh)
        data_rh = np.load(path_rh)
        data_concat = np.concatenate([data_lh, data_rh], axis=1)  # (n_timepoints, n_vertices)

        # Union of this area's sub-ROI masks (e.g. V1v + V1d for 'V1'), each already
        # noise-ceiling-filtered.
        combined_mask = np.zeros(data_concat.shape[1], dtype=bool)
        for sub_roi in sub_rois:
            combined_mask |= subject_masks[subject][sub_roi]

        print(f"Loaded whole-brain correlations for Sub {subject}, Area {area} data shape = {data_concat[:, combined_mask].shape}")

        subject_area_corrs.append(np.mean(data_concat[:, combined_mask], axis=1))  # Averaging across vertices

    area_data_list.append(np.array(subject_area_corrs))

def ci95_across_subjects(area_data, N_BOOTSTRAPS=N_BOOTSTRAPS):
    """
    95% confidence interval of the across-subject mean at each timepoint, via percentile
    bootstrap: resample subjects with replacement N_BOOTSTRAPS times, recompute the mean
    across the resampled subjects at each timepoint, then take the 2.5th/97.5th percentiles
    of that bootstrap distribution.
    """
    n_subs = area_data.shape[0]
    boot_means = np.zeros((N_BOOTSTRAPS, area_data.shape[1]))
    for i in range(N_BOOTSTRAPS):
        res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
        boot_means[i] = np.mean(area_data[res_idx], axis=0)
    low, high = np.percentile(boot_means, [2.5, 97.5], axis=0)
    return low, high


# --- Plotting & Stats ---
def plot_roi_results(data_list, title, filename):
    n_timepoints = len(times)
    plt.figure(figsize=(26, 15))
    ax = plt.gca()

    # Calculate y-limit based on data (using the 95% CI half-width, not SEM)
    all_means = [np.mean(d, axis=0) for d in data_list]
    all_cis = [ci95_across_subjects(d) for d in data_list]
    global_max_y = max([np.max(m + c) for m, c in zip(all_means, all_cis)])

    # Row spacing for the staggered significance lanes below y=0 (one lane per area)
    row_gap = global_max_y * 0.02

    print("\n>>> Peak latency (95% CI, bootstrap over subjects) per ROI <<<")

    for i, area_data in enumerate(data_list):
        n_subs = len(subject_list)
        m_group = np.mean(area_data, axis=0)
        color = area_colors[i]

        # 1. Cluster Permutation Test
        cluster_results = sign_permutation_cluster_test(area_data, n_permutations=10000)
        sig_mask = np.zeros(n_timepoints, dtype=bool)
        for cluster_idx, _, _ in cluster_results['significant_clusters']:
            sig_mask[cluster_idx] = True

        # Significant time window: first and last significant timepoint, pooled across all
        # significant clusters
        if np.any(sig_mask):
            sig_times = times[sig_mask]
            print(f"{area_labels[i]}: significant from {sig_times.min():.0f} ms to {sig_times.max():.0f} ms")
        else:
            print(f"{area_labels[i]}: no significant time points")

        # 2. Bootstrap Peak Latency CI

        boot_peaks = []
        for _ in range(N_BOOTSTRAPS):
            res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
            boot_peaks.append(times[np.argmax(np.mean(area_data[res_idx], axis=0))])

        low, high = np.percentile(boot_peaks, [2.5, 97.5])
        obs_peak = times[np.argmax(m_group)]

        print(f"{area_labels[i]}: peak latency = {obs_peak:.0f} ms [95% CI: {low:.0f}-{high:.0f} ms]")

        # 3. Plotting
        leg_text = f"{area_labels[i]}: {obs_peak:.0f} ms [{low:.0f}-{high:.0f} ms]"

        ax.plot(times, m_group, color=color, lw=6.0, label=leg_text, zorder=3)
        ci_low, ci_high = ci95_across_subjects(area_data)
        ax.fill_between(times, ci_low, ci_high, color=color, alpha=0.20, zorder=2)

        # Peak Marker
        peak_val = np.max(m_group)
        ax.scatter(obs_peak, peak_val, color=color, s=600, edgecolors='white', zorder=5)
        #ax.errorbar(obs_peak, peak_val, xerr=[[obs_peak-low], [high-obs_peak]],
        #            fmt='none', ecolor='k', elinewidth=1, capsize=3, zorder=4)

        # Significance bars -- staggered lanes BELOW the y=0 line (one lane per area)
        sig_y = -row_gap * (i + 1)
        if np.any(sig_mask):
            ax.scatter(times[sig_mask], [sig_y] * np.sum(sig_mask),
                       color=color, s=50, marker='s', alpha=0.8, edgecolors='none', zorder=3)


    # Styling
    ax.set_title(f'{title}', fontweight='bold', fontsize=28, pad=40)
    ax.set_xlabel('Time (ms)', fontsize=28)
    ax.set_ylabel("Pearson's r", fontsize=28)
    ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=3, alpha=0.2)
    ax.set_xlim(-100, 3500)
    ax.set_xticks([-100, 0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
    bottom_limit = -row_gap * (len(area_labels) + 1.5)
    ax.set_ylim(bottom_limit, top=0.3)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])

    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=18)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.tick_params(axis='both', labelsize=26, width=6.0, length=24.0)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")


def plot_roi_results_individual_subjects(data_list, filename, n_cols=3):
    n_subs = len(subject_list)
    n_rows = int(np.ceil(n_subs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), sharex=False)
    axes = np.array(axes).reshape(-1)

    print("\n>>> Peak latency per ROI, per subject (individual-subject plot) <<<")
    for s_idx, subject in enumerate(subject_list):
        ax = axes[s_idx]
        print(f"Participant-{subject}:")
        for a_idx, area_data in enumerate(data_list):
            if s_idx >= len(area_data):
                continue
            curve = area_data[s_idx, :]
            obs_peak = times[np.argmax(curve)]
            peak_val = curve.max()
            print(f"  {area_labels[a_idx]}: peak latency = {obs_peak:.0f}ms")
            leg_text = f"{area_labels[a_idx]}: {obs_peak:.0f}ms"
            ax.plot(times, curve, color=area_colors[a_idx], lw=7, label=leg_text, zorder=3)
            ax.scatter(obs_peak, peak_val, color=area_colors[a_idx], s=150,
                       edgecolor='black', linewidth=1.5, zorder=4)

        ax.set_title(f'Participant-{subject}', fontweight='bold', fontsize=16, pad=10)
        ax.axvline(0, color='black', lw=3.0, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', lw=3.0, alpha=0.2)
        ax.set_xlim(-100, 3500)
        ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
        ax.set_yticks([0.0, 0.1, 0.2, 0.3])
        ax.legend(loc='upper right', frameon=False, fontsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=-0.03, top=0.3)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', labelsize=11, labelbottom=False, labelleft=False, width=3.0, length=12.0)

    for extra_ax in axes[n_subs:]:
        extra_ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


# Run the plots
plot_roi_results(area_data_list, "Encoding Fusion", "roi_encoding_fusion.svg")
plot_roi_results_individual_subjects(area_data_list, "roi_encoding_fusion_individual_participants.svg")

print(f"Execution complete! Total Time: {time.time() - start_time:.2f} seconds.")