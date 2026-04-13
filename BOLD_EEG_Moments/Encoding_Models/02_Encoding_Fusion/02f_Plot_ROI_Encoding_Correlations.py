import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.ndimage import label
import pickle
from tqdm import tqdm
import time

# Start time
start_time = time.time()


parser = argparse.ArgumentParser()
parser.add_argument('--eeg_frequency', type=int, default=100)
args = parser.parse_args()


rois_masks_dir = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_roi_masks_hemi-{}.npy'
subject_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']   # List of subjects to process
eeg_path = '/scratch/giffordale95/projects/eeg_moments/dataset/preprocessed_data/eeg/sub-01/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-'+format(args.eeg_frequency, '04')+'/preprocessed_data.npy'
noise_ceiling_file = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_noiseceiling_space-fsaverage_task-test_hemi-{}_n-10.pkl'
times = np.load(eeg_path, allow_pickle=True).item()['times']  # Loading the time vector from the EEG data
post_stimulus_times = 1000*times[times >= 0]  # Selecting only the post-stimulus times
plots_dir = '/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/plots/encoding_fusion_correlations'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


# Function to bootstrap the mean correlation vector for each Area
def bootstrap_mean_vector(correlation_vectors, n_iterations=10000):
    mean_vectors = []

    for _ in range(n_iterations):
        # Resampling subjects with replacement
        sample_indices = np.random.choice(len(correlation_vectors), size=len(correlation_vectors), replace=True)
        sampled_vectors = correlation_vectors[sample_indices]

        # Calculating the mean of the sampled latencies
        mean_vector = np.mean(sampled_vectors, axis=0)
        mean_vectors.append(mean_vector)

    return np.array(mean_vectors)

def select_vertices(data, mask, noise_ceiling, threshold=30.):
    # Selecting vertices from the whole-brain surface based on 2 conditions: belonging to the ROI and noise ceiling above the threshold
    # Condition 1: Non-zero values of the mask (selecting vertices belonging to the ROI)
    condition1 = mask != 0

    # Condition 2: Vertices with a noise ceiling greater than the threshold
    threshold = 20.
    condition2 = noise_ceiling >= threshold

    # Combined conditions
    combined_condition = condition1 & condition2

    # Selecting the vertices that satisfy both conditions

    return data[:,combined_condition]

def sign_permutation_cluster_test(corr_timecourses, n_permutations=10000, p_thresh=0.01, alpha=0.05):
    """
    Perform a sign-permutation cluster test across subjects' correlation time courses.

    Parameters
    ----------
    corr_timecourses : list of np.ndarray
        Each array has shape (n_timepoints,)
    n_permutations : int
        Number of permutations for the null distribution.
    p_thresh : float
        Cluster-defining threshold (pointwise).
    alpha : float
        Cluster-level corrected significance threshold.

    Returns
    -------
    results : dict
        Contains:
        - 'observed_clusters': list of (indices, cluster_sum, p_value)
        - 'significant_clusters': same, filtered by p < alpha
        - 'average_onset_index': mean index of significant cluster onsets (or None)
    """
    corr_timecourses = np.array(corr_timecourses)  # shape: (n_subjects, n_timepoints)
    n_subj, n_time = corr_timecourses.shape

    # -------------------------
    # 1. Compute observed group mean
    # -------------------------
    mean_corr = np.mean(corr_timecourses, axis=0)

    # -------------------------
    # 2. Build null distribution via sign-flipping
    # -------------------------
    null_distrib = np.zeros((n_permutations, n_time))

    for i in range(n_permutations):
        signs = np.random.choice([-1, 1], size=(n_subj, 1))
        permuted = corr_timecourses * signs
        null_distrib[i, :] = np.mean(permuted, axis=0)

    # -------------------------
    # 3. Define cluster threshold (P < p_thresh)
    # -------------------------
    upper_thr = np.percentile(null_distrib, 100 * (1 - p_thresh / 2))
    lower_thr = np.percentile(null_distrib, 100 * (p_thresh / 2))

    # -------------------------
    # 4. Find clusters in observed data
    # -------------------------
    suprathreshold = (mean_corr > upper_thr) | (mean_corr < lower_thr)
    labeled, n_clusters = label(suprathreshold)

    clusters = []
    for i in range(1, n_clusters + 1):
        idx = np.where(labeled == i)[0]
        cluster_sum = np.sum(mean_corr[idx])
        clusters.append((idx, cluster_sum))

    # -------------------------
    # 5. Null distribution of cluster sums (max per permutation)
    # -------------------------
    max_cluster_sums = np.zeros(n_permutations)
    for i in range(n_permutations):
        suprathreshold_null = (null_distrib[i, :] > upper_thr) | (null_distrib[i, :] < lower_thr)
        labeled_null, n_null = label(suprathreshold_null)
        if n_null > 0:
            cluster_sums_null = [np.sum(null_distrib[i, np.where(labeled_null == j)[0]]) for j in range(1, n_null + 1)]
            max_cluster_sums[i] = np.max(cluster_sums_null)
        else:
            max_cluster_sums[i] = 0

    # -------------------------
    # 6. Compute corrected p-values for observed clusters
    # -------------------------
    results = []
    for idx, cluster_sum in clusters:
        p_val = (np.sum(max_cluster_sums >= np.abs(cluster_sum)) + 1) / (n_permutations + 1)
        results.append((idx, cluster_sum, p_val))

    # Filter significant clusters
    significant_clusters = [r for r in results if r[2] < alpha]

    # -------------------------
    # 7. Compute average onset index
    # -------------------------
    if significant_clusters:
        onsets = [r[0][0] for r in significant_clusters]  # first index of each cluster
        avg_onset = int(np.mean(onsets))
    else:
        avg_onset = None
    
    # -------------------------
    # 8. Compute per-subject onset list
    # -------------------------
    subject_onsets = []
    for subj_corr in corr_timecourses:
        subj_suprathreshold = (subj_corr > upper_thr) | (subj_corr < lower_thr)
        if np.any(subj_suprathreshold):
            onset_idx = int(np.argmax(subj_suprathreshold))  # first True
        else:
            onset_idx = None
        subject_onsets.append(onset_idx)

    return {
        "observed_clusters": results,
        "significant_clusters": significant_clusters,
        "onsets": subject_onsets,  # <--- list of 10 onset indices
        "mean_corr": mean_corr,
        "upper_thr": upper_thr,
        "lower_thr": lower_thr
    }


def plot_multiple_timecourses_with_significance(results_list, areas_correlations, time_axis, roi_labels, plots_dir,
                                                dot_height=0.02, color_sig="lawngreen", color_nonsig="lightgray"):
    """
    Plot multiple correlation time courses (e.g., for different ROIs) on separate subplots
    with significance dots, similar to plot_timecourse_with_significance().

    Parameters
    ----------
    results_list : list of dict
        Each dict is the output of sign_permutation_cluster_test() for one ROI.
    time_axis : np.ndarray
        Time points in ms (shared across ROIs).
    roi_labels : list of str
        Names of the ROIs (e.g., ['V1&V2', 'LOC', 'PPA', 'FFA']).
    plots_dir : str
        Directory to save the resulting figure.
    dot_height : float
        Vertical offset for significance dots relative to y-axis range.
    color_sig : str
        Color for significant dots.
    color_nonsig : str
        Color for non-significant dots.
    """

    n_rois = len(results_list)
    fig, axes = plt.subplots(n_rois, 1, figsize=(12, 2.5 * n_rois), sharex=True)

    if n_rois == 1:
        axes = [axes]  # Make iterable if only one subplot
    colors = ['blue', 'green', 'orange', 'purple']
    for i, (res, ax) in enumerate(zip(results_list, axes)):
        
        mean_corr = res["mean_corr"]
        bootstrapped_mean_corrs = bootstrap_mean_vector(np.array(areas_correlations[i]))
        lower_bound = np.percentile(bootstrapped_mean_corrs, 2.5, axis=0)
        upper_bound = np.percentile(bootstrapped_mean_corrs, 97.5, axis=0)
        n_time = len(mean_corr)

        # Build significance mask
        sig_mask = np.zeros(n_time, dtype=bool)
        for idx, _, _ in res["significant_clusters"]:
            sig_mask[idx] = True
        subject_onsets_indices = sign_permutation_cluster_test(areas_correlations[i][:, 20:])['onsets']
        subject_onsets = [post_stimulus_times[index].item() for index in subject_onsets_indices]
        mean_onset = np.mean(subject_onsets).item()
        # --- Plot main time course ---
        ax.plot(time_axis, mean_corr, color=colors[i], lw=2, label="Group average")
        ax.axvline(x=0, color='black', linestyle='--')
        ax.axhline(0, color="black", lw=1, ls="-")
        #ax.axvline(x=time_axis[np.where(sig_mask)[0][0]], color='gray', linestyle=':', label =f'{time_axis[np.where(sig_mask)[0][0]]:.2f} ms')
        ax.axvline(x=mean_onset, color='gray', linestyle=':', label =f'{mean_onset:.2f} ms')
        ax.fill_between(time_axis, lower_bound, upper_bound, color=colors[i], alpha=0.2)

        # Compute y position for dots
        y_min, y_max = ax.get_ylim()
        y_dot = y_max + (y_max - y_min) * dot_height

        # --- Plot significance dots ---
        ax.scatter(time_axis[sig_mask], np.full(np.sum(sig_mask), y_dot),
                   color=color_sig, s=15, label="Significant")
        ax.scatter(time_axis[~sig_mask], np.full(np.sum(~sig_mask), y_dot),
                   color=color_nonsig, s=10, alpha=0.4, label="Non-significant")

        # Make dots visible
        ax.set_ylim(y_min, y_dot + (y_max - y_min) * 0.05)

        # Titles and labels
        ax.set_ylabel("Pearson\'s R")
        ax.set_title(roi_labels[i])
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')

        if i == 0:
            ax.legend(loc='upper right', frameon=False)

    # Shared X label and layout
    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()

    save_path = os.path.join(plots_dir, 'V1V2-LOC-PPA-FFA_avg.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")


area1 = ['V1d', 'V1v', 'V2d', 'V2v']
area2 = ['LOC']
area3 = ['PPA']
area4 = ['FFA']
areas = [area1, area2, area3, area4]
areas_corrs = [[], [], [], []]
areas_lb = [[], [], [], []]
areas_ub = [[], [], [], []]
areas_correlations = []
areas_upper_bounds = []
areas_lower_bounds = []
for area, area_corrs, area_ub, area_lb in tqdm(zip(areas, areas_corrs, areas_ub, areas_lb)):
    for roi in area:
        
        # Loop through each subject and load the corresponding results
        subject_correlations_left = [] # Initializing an empty list to accumulate correlations for the current ROI (left hemisphere)
        subject_correlations_right = [] # Initializing an empty list to accumulate correlations for the current ROI (right hemisphere)
        subject_corrs = []  # Initializing a list to store the correlation data for each subject
        
        for subject in subject_list:
            accuracies_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/eeg2fmri_wb/100_Hz/eeg_channel_policy_append/fmri_sub-{subject}'
            data_dir_l = os.path.join(accuracies_dir, 'correlations_left.npy')
            data_dir_r = os.path.join(accuracies_dir, 'correlations_right.npy')
            data_left = np.load(data_dir_l)
            data_right = np.load(data_dir_r)

            mask_l = np.load(rois_masks_dir.format(subject, subject, 'left'), allow_pickle=True).item()[roi]
            mask_r = np.load(rois_masks_dir.format(subject, subject, 'right'), allow_pickle=True).item()[roi]

            noise_ceiling_left = pickle.load(open(noise_ceiling_file.format(subject, subject, 'left'), 'rb'))[1]
            noise_ceiling_right = pickle.load(open(noise_ceiling_file.format(subject, subject, 'right'), 'rb'))[1]


            roi_corrs_left = np.mean(select_vertices(data_left, mask_l, noise_ceiling_left, 30.), axis=1)
            roi_corrs_right = np.mean(select_vertices(data_right, mask_r, noise_ceiling_right, 30.), axis=1)

            corrs = np.mean([roi_corrs_left, roi_corrs_right], axis=0)
            subject_corrs.append(corrs)


        # After processing all subjects, average the correlations for the current ROI

        # Using standard deviation as lower and upper bounds for simplicity
        std = np.std(np.array(subject_corrs), axis=0)
        lower_bound = np.mean(np.array(subject_corrs), axis=0) - std
        upper_bound = np.mean(np.array(subject_corrs), axis=0) + std

        area_lb.append(lower_bound.flatten())
        area_ub.append(upper_bound.flatten())

        area_corrs.append(subject_corrs)  # Storing the mean correlation vector for the current ROI in area1
    if len(area_corrs) > 1:
        areas_correlations.append(np.mean(area_corrs, axis=0))  # Averaging the correlations across all ROIs in area1
        areas_lower_bounds.append(np.mean(area_lb, axis=0))
        areas_upper_bounds.append(np.mean(area_ub, axis=0))
    elif len(area_corrs) == 1:
        areas_correlations.append(np.array(area_corrs[0]))  # If only one ROI, just take its correlations
        areas_lower_bounds.append(np.array(area_lb[0]))
        areas_upper_bounds.append(np.array(area_ub[0]))
print("Length of areas correlations: ", len(areas_correlations))
print("Shape of each area correlation: ", [area.shape for area in areas_correlations])

roi_labels = ['V1&V2', 'LOC', 'PPA', 'FFA']
results_list =[sign_permutation_cluster_test(subject_corrs, n_permutations=10000) for subject_corrs in areas_correlations]

plot_multiple_timecourses_with_significance(results_list, areas_correlations, 1000*times, roi_labels, plots_dir)

# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Plots saved in: {plots_dir}")
print(f"Execution time: {execution_time:.2f} seconds.")
