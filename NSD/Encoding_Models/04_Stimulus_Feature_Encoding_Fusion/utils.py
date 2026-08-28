import os
import h5py
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.ndimage import label
from berg import BERG




def load_fmri_hemi_data(subject, hemisphere):

    data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
    train_filename = f'fmri_train_sub-{subject:02d}_hemi-{hemisphere}.npy'
    test_filename = f'fmri_test_sub-{subject:02d}_hemi-{hemisphere}.npy'
    fmri_train = np.load(os.path.join(data_dir, train_filename), allow_pickle=True).item()['fmri_train'].astype(np.float32)
    fmri_test_dict = np.load(os.path.join(data_dir, test_filename),allow_pickle=True).item()
    fmri_test = np.mean(fmri_test_dict['fmri_test'], axis=1, dtype=np.float32)
    del fmri_test_dict

    return fmri_train, fmri_test

def load_fmri_roi_data(subject, hemisphere, roi, nc_threshold=0.2):
    data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
    berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
    train_filename = f'fmri_train_sub-{subject:02d}_hemi-{hemisphere}.npy'
    fmri_train = np.load(os.path.join(data_dir, train_filename), allow_pickle=True).item()['fmri_train'].astype(np.float32)

    test_filename = f'fmri_test_sub-{subject:02d}_hemi-{hemisphere}.npy'
    fmri_test_dict = np.load(os.path.join(data_dir, test_filename),allow_pickle=True).item()
    fmri_test = np.mean(fmri_test_dict['fmri_test'], axis=1, dtype=np.float32)
    del fmri_test_dict


    metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)
    # Available ROIS:
    # dict_keys(['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FBA-1', 'FBA-2', 'mTL-bodies', 'OFA', 'FFA-1',
    #  'FFA-2', 'mTL-faces', 'aTL-faces', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words', 'early', 
    # 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal', 'nsdgeneral'])
    # Selecting the ROI indices
    roi_idx = metadata['fmri'][f'{hemisphere}_fsaverage_rois'][roi]
    roi_mask = np.zeros(fmri_train.shape[1], dtype=bool)
    roi_mask[roi_idx] = True
    wb_noise_ceilings = metadata['fmri'][f'{hemisphere}_ncsnr']
    nc_mask = np.zeros(fmri_train.shape[1], dtype=bool)
    nc_idx = np.where(wb_noise_ceilings >= nc_threshold)[0] # Selecting vertices above noise ceiling threshold of 20%
    nc_mask[nc_idx] = True
    valid_vertices = np.where(roi_mask & nc_mask)[0] # Selecting vertices above noise ceiling threshold of 20%
    
    fmri_train = fmri_train[:, valid_vertices].astype(np.float32)
    fmri_test = fmri_test[:, valid_vertices].astype(np.float32)

    del roi_mask, nc_mask
    return fmri_train, fmri_test


def get_eeg_times():
    # Get the time points # !!! Use official time points
    n_times = 615
    times = np.round(np.linspace(-200, 1000, n_times)).astype(int)
    # Account for the 50ms shift in the EEG responses # !!!
    shift = -50
    times = times + shift
    # Only select time points between -100ms and 600ms
    t_start = np.where(times == -100)[0][0]
    t_end = np.where(times == 600)[0][0]
    times = times[t_start:t_end+1]
    return times



def get_significance_mask(correlation_data, alpha=0.05, method='fdr_bh'):
    """
    Performs Fisher z-transform, 1-sample t-test, and Multiple Comparison Correction.
    
    Parameters:
    - correlation_data: array (n_subjects, n_timepoints)
    - alpha: significance level
    - method: 'fdr_bh' for Benjamini-Hochberg, 'bonferroni' for the strict way.
    """
    # 1. Fisher z-transform (handles the r-distribution skew)
    # Note: We use np.clip to avoid infinity if r is exactly 1.0
    z_data = np.arctanh(np.clip(correlation_data, -0.999, 0.999))
    
    # 2. Perform 1-sample t-test across subjects at each timepoint
    t_stats, p_values = stats.ttest_1samp(z_data, 0, axis=0)
    
    # 3. Correct for Multiple Comparisons
    # reject is a boolean mask, pvals_corrected are the adjusted p-values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
    
    return pvals_corrected < alpha


def sign_permutation_cluster_test(corr_timecourses, n_permutations=10000, p_thresh=0.05, alpha=0.05):
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
        "onsets": subject_onsets,  # <--- list of 6 onset indices
        "mean_corr": mean_corr,
        "upper_thr": upper_thr,
        "lower_thr": lower_thr
    }

