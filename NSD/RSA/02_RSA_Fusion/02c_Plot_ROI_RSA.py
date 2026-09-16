import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utils import sign_permutation_cluster_test, get_eeg_times
from berg import BERG
import time


# Start time
start_time = time.time()

# --- Configuration ---
subject_list = [1,4,5,6,7,8]

# Define the grouping: Area -> list of sub-ROI filenames
# Define the grouping hierarchy: Group Label -> List of sub-ROI file names
roi_groups = {
    'V1': ['V1v', 'V1d'],
    'V4': ['hV4'],
    'ventral': ['ventral']
}

# Ordered labels and explicit high-contrast colors for each integrated group
area_labels = ['V1', 'V4', 'ventral']
area_colors = [
    "#480758",  # V1 (Deep Purple)
    "#468fc3",  # hV4 (Steel Blue)
    "#ea8e16" ,  # ventral (Orange)
]

n_bootstraps = 10000
N_NEIGHBOURS = 25
# Pathing
# f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/univariate_rsa/subject-{args.subject}'
base_results_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/searchlight_fusion/eeg_rdm_metric-pearsonr/n_neighbours-{N_NEIGHBOURS}/aggregated_results'
#base_results_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/searchlight_fusion/eeg_rdm_metric-pearsonr/n_neighbours-100/aggregated_resultssubject-{subject}/subject-{subject}_lh_hemisphere_timecourse.npy
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()

# Helper function to select vertices based on mask and noise ceiling threshold
def select_vertices(data, mask, noise_ceilings, threshold=0.2):
    # Selecting vertices from the whole-brain surface based on 2 conditions: belonging to the ROI and noise ceiling above the threshold
    # Condition 1: Non-zero values of the mask (selecting vertices belonging to the ROI)
    condition1 = mask != 0

    # Condition 2: Vertices with a noise ceiling greater than the threshold
    condition2 = noise_ceilings >= threshold

    # Combined conditions
    combined_condition = condition1 & condition2

    # Selecting the vertices that satisfy both conditions

    return data[:,combined_condition]

# --- Data Aggregation ---
area_data_list = [] # Will store [Area][Subject, Time]

print(">>> Aggregating ROI Data (Averaging sub-ROIs) <<<")

for area in area_labels:
    sub_rois = roi_groups[area]
    subject_area_corrs = []

    for subject in subject_list:
        sub_roi_corrs = []
        for sr, sub_roi in enumerate(sub_rois):
            try:
                # Direct load of pre-averaged npy files
                path_lh = os.path.join(base_results_dir, f'subject-{subject}', f'subject-{subject}_lh_hemisphere_timecourse.npy')
                path_rh = os.path.join(base_results_dir, f'subject-{subject}', f'subject-{subject}_rh_hemisphere_timecourse.npy')

                data_lh = np.load(path_lh)
                data_rh = np.load(path_rh)

                #print(f"Loaded data for subject {subject}, sub-ROI {sub_roi}: LH shape {data_lh.shape}, RH shape {data_rh.shape}")

                data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
                berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
                metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)
                # Available ROIS:
                # dict_keys(['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FBA-1', 'FBA-2', 'mTL-bodies', 'OFA', 'FFA-1',
                #  'FFA-2', 'mTL-faces', 'aTL-faces', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words', 'early',
                # 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal', 'nsdgeneral'])
                # Selecting the ROI indices
                roi_idx_lh = metadata['fmri']['lh_fsaverage_rois'][sub_roi]
                roi_mask_lh = np.zeros(163842, dtype=bool)
                roi_mask_lh[roi_idx_lh] = True

                roi_idx_rh = metadata['fmri']['rh_fsaverage_rois'][sub_roi]
                roi_mask_rh = np.zeros(163842, dtype=bool)
                roi_mask_rh[roi_idx_rh] = True

                wb_noise_ceilings_lh = metadata['fmri']['lh_ncsnr']
                wb_noise_ceilings_rh = metadata['fmri']['rh_ncsnr']

                roi_corrs_left = select_vertices(data_lh, roi_mask_lh, wb_noise_ceilings_lh)
                roi_corrs_right = select_vertices(data_rh, roi_mask_rh, wb_noise_ceilings_rh)

                data_concat = np.concatenate([roi_corrs_left, roi_corrs_right], axis=1)

                if sr == 0:
                    sub_roi_corrs = data_concat
                else:
                    sub_roi_corrs = np.concatenate([sub_roi_corrs, data_concat], axis=1)

            except FileNotFoundError:
                    print(f"Warning: Data file for subject {subject}, sub-ROI {sub_roi} not found !")
                    continue


        subject_area_corrs.append(np.mean(sub_roi_corrs, axis=1)) # Averaging across vertices

    area_data_list.append(np.array(subject_area_corrs))



def ci95_across_subjects(area_data, n_bootstraps=10000):
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


# --- Plotting & Stats ---
def plot_roi_results(data_list, title, filename):
    n_timepoints = len(times)
    plt.figure(figsize=(14, 8))
    ax = plt.gca()

    # Calculate y-limit based on data (using the 95% CI upper bound, not SEM). Bootstrap once
    # per ROI here and reuse the result inside the loop below, rather than recomputing it.
    all_means = [np.mean(d, axis=0) for d in data_list]
    all_cis = [ci95_across_subjects(d) for d in data_list]  # each: (low, high) arrays
    global_max_y = max(np.max(high) for _, high in all_cis)
    # Row spacing for the staggered significance lanes below y=0 (one lane per area)
    row_gap = global_max_y * 0.05

    print("\n>>> Peak latency (95% CI, bootstrap over subjects) per ROI <<<")

    for i, area_data in enumerate(data_list):
        n_subs = len(subject_list)
        m_group = np.mean(area_data, axis=0)
        ci_low, ci_high = all_cis[i]  # ribbon CI: uncertainty in the correlation value itself
        color = area_colors[i]

        # 1. Cluster Permutation Test
        cluster_results = sign_permutation_cluster_test(area_data, n_permutations=10000)
        sig_mask = np.zeros(n_timepoints, dtype=bool)
        for cluster_idx, _, _ in cluster_results['significant_clusters']:
            sig_mask[cluster_idx] = True

        # Significant time window: first and last significant timepoint, pooled across all
        # significant clusters (not a per-cluster breakdown) -- matching how significant windows
        # are reported elsewhere in this project (e.g. "significant between ~50ms and 420ms").
        if np.any(sig_mask):
            sig_times = times[sig_mask]
            print(f"{area_labels[i]}: significant from {sig_times.min():.0f}ms to {sig_times.max():.0f}ms")
        else:
            print(f"{area_labels[i]}: no significant time points")

        # 2. Bootstrap Peak Latency CI: uncertainty in the peak's TIME INDEX, obtained by
        # resampling subjects and re-finding the argmax each time -- not to be confused
        # with the ribbon's CI above, which is about the correlation value, not its timing.
        boot_peaks = []
        for _ in range(n_bootstraps):
            res_idx = np.random.choice(n_subs, size=n_subs, replace=True)
            boot_peaks.append(times[np.argmax(np.mean(area_data[res_idx], axis=0))])

        low, high = np.percentile(boot_peaks, [2.5, 97.5])
        obs_peak = times[np.argmax(m_group)]

        print(f"{area_labels[i]}: peak latency = {obs_peak:.0f}ms [95% CI: {low:.0f}-{high:.0f}ms]")

        # 3. Plotting
        leg_text = f"{area_labels[i]}: {obs_peak:.0f}ms [{low:.0f}-{high:.0f}ms]"

        ax.plot(times, m_group, color=color, lw=12.0, label=leg_text, zorder=3)
        ax.fill_between(times, ci_low, ci_high, color=color, alpha=0.20, zorder=2)

        # Peak Marker
        peak_val = np.max(m_group)
        ax.scatter(obs_peak, peak_val, color=color, s=600, edgecolors='white', zorder=5)
        #ax.errorbar(obs_peak, peak_val, xerr=[[obs_peak-low], [high-obs_peak]],
        #            fmt='none', ecolor='k', elinewidth=1, capsize=3, zorder=4)

        # Significance Dots -- staggered lanes BELOW the y=0 line (one lane per area)
        sig_y = -row_gap * (i + 1)
        if np.any(sig_mask):
            ax.scatter(times[sig_mask], [sig_y] * np.sum(sig_mask),
                       color=color, s=50, marker='s', alpha=0.8, edgecolors='none', zorder=3)

    # Styling
    ax.set_title(f'{title}', fontweight='bold', fontsize=28, pad=40)
    ax.set_xlabel('Time (ms)', fontsize=26)
    ax.set_ylabel("Spearman's R", fontsize=26)
    ax.axvline(0, color='black', lw=3, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', lw=3, alpha=0.2)
    ax.set_xlim(-100, 600)
    ax.set_yticks([0.00, 0.02, 0.04])
    bottom_limit = -row_gap * (len(area_labels) + 1.5)
    ax.set_ylim(bottom=bottom_limit, top=0.040)

    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.tick_params(axis='both', labelsize=26, width=3.0, length=14.0)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

# =============================================================================
# Individual-subject companion plot: 
# =============================================================================
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
        ax.set_xlim(-100, 600)
        ax.set_xticks([0, 200, 400, 600])
        ax.set_yticks([0.00, 0.02, 0.04, 0.06])
        ax.legend(loc='upper right', frameon=False, fontsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=-0.004, top=0.064)
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
plot_roi_results(area_data_list, "Searchlight RSA", "roi_sl_rsa_fusion_k25.svg")
#plot_roi_results_individual_subjects(area_data_list, "roi_sl_rsa_fusion_individual_participants.svg")

print(f"Execution complete! Total Time: {time.time() - start_time:.2f}s")