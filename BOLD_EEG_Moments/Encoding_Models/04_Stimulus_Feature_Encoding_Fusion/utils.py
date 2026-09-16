import numpy as np
import pickle
import os
from scipy.ndimage import label
from scipy import stats

def load_fmri_data(subject, hemisphere, roi, threshold=0.0):

    base_dir = ('/scratch/giffordale95/projects/eeg_moments_dataset/bold_moments_dataset/derivatives/versionB/fsaverage/GLM')
    betas_dir = os.path.join(base_dir, f'sub-{subject:02d}', 'prepared_betas')

    hemispheres = ['left', 'right'] if hemisphere == 'both' else [hemisphere]
    loaded = {}  # loaded[hemi] = (fmri_train, fmri_test)

    for hemi in hemispheres:

        fmri_file_train = os.path.join(
            betas_dir, f'sub-{subject:02d}_organized_betas_task-train_hemi-{hemi}_normalized.pkl')
        fmri_file_test = os.path.join(
            betas_dir, f'sub-{subject:02d}_organized_betas_task-test_hemi-{hemi}_normalized.pkl')
        rois_masks_file = os.path.join(
            betas_dir, f'sub-{subject:02d}_roi_masks_hemi-{hemi}.npy')
        noise_ceiling_file = os.path.join(
            betas_dir, f'sub-{subject:02d}_noiseceiling_space-fsaverage_task-test_hemi-{hemi}_n-10.pkl')

        with open(fmri_file_train, 'rb') as f:
            fmri_T = np.float32(pickle.load(f)[0])
        with open(fmri_file_test, 'rb') as f:
            fmri_t = np.float32(pickle.load(f)[0])
        with open(noise_ceiling_file, 'rb') as f:
            noise_ceiling = pickle.load(f)[1]

        # Selecting vertices from the whole-brain surface based on 2 conditions: belonging to the
        # ROI and noise ceiling above the threshold. For WB, only the noise ceiling applies.
        # Condition 2: Vertices with a noise ceiling greater than the threshold
        combined_condition = noise_ceiling >= threshold
        if roi != 'WB':
            mask = np.load(rois_masks_file, allow_pickle=True).item()[roi]
            # Condition 1: Non-zero values of the mask (selecting vertices belonging to the ROI)
            combined_condition = combined_condition & (mask != 0)

        # Selecting the vertices that satisfy both conditions and Averaging the fMRI responses
        # across repetitions
        fmri_train = np.mean(fmri_T, axis=1, dtype=np.float32)[:, combined_condition]
        fmri_test = np.mean(fmri_t, axis=1, dtype=np.float32)[:, combined_condition]
        del fmri_T, fmri_t

        if roi != 'WB':
            if hemisphere == 'both':
                print(f"Number of selected vertices - {hemi} hemisphere: (train, test)",
                      fmri_train.shape[1], fmri_test.shape[1])
            else:
                print("Number of selected vertices : (train, test)",
                      fmri_train.shape[1], fmri_test.shape[1])

        loaded[hemi] = (fmri_train, fmri_test)

    if hemisphere == 'both':
        return (np.concatenate(loaded['left'], axis=0),
                np.concatenate(loaded['right'], axis=0))

    return loaded[hemisphere]


def load_ncsnr(subject, hemisphere):
    """
    Loads the vertex-wise noise ceiling SNR for a full hemisphere, one value per fsaverage vertex
    Parameters
    ----------
    subject : str
        Subject id, as used in the 'sub-{subject}' path components.
    hemisphere : {'left', 'right'}

    Returns
    -------
    np.ndarray of shape (n_vertices,)
    """
    base_dir = ('/scratch/giffordale95/projects/eeg_moments_dataset/bold_moments_dataset/derivatives/versionB/fsaverage/GLM')
    noise_ceiling_file = os.path.join(
        base_dir, f'sub-{subject:02d}', 'prepared_betas',
        f'sub-{subject:02d}_noiseceiling_space-fsaverage_task-test_hemi-{hemisphere}_n-10.pkl')

    with open(noise_ceiling_file, 'rb') as f:
        # Element [1] of the pickle is the vertex-wise NCSNR -- same index load_fmri_data uses
        return pickle.load(f)[1]
    


def load_roi_indices(subject, hemisphere, roi):
    """
    Loads the vertex indices belonging to one ROI, for a single hemisphere.

    Parameters
    ----------
    subject : str
        Subject id, as used in the 'sub-{subject}' path components.
    hemisphere : {'left', 'right'}
    roi : str
        ROI name, as keyed in the subject's roi_masks file (e.g. 'V1v', 'hV4').

    Returns
    -------
    np.ndarray of int
        Indices of the vertices with a non-zero mask value for this ROI.
    """
    base_dir = ('/scratch/giffordale95/projects/eeg_moments_dataset/bold_moments_dataset/derivatives/versionB/fsaverage/GLM')
    rois_masks_file = os.path.join(
        base_dir, f'sub-{subject:02d}', 'prepared_betas',
        f'sub-{subject:02d}_roi_masks_hemi-{hemisphere}.npy')

    mask = np.load(rois_masks_file, allow_pickle=True).item()[roi]

    # Non-zero mask values flag ROI membership -- same criterion as load_fmri_data's condition1
    return np.where(mask != 0)[0]



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


    return {
        "observed_clusters": results,
        "significant_clusters": significant_clusters,
        "mean_corr": mean_corr,
        "upper_thr": upper_thr,
        "lower_thr": lower_thr
    }



PREPARED_DATA_DIR = '/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/prepared_data'


def load_eeg_data(multiple_subjects=True, subject=None, trial_average='all'):
    """
    Load the prepared EMD EEG responses

    Parameters
    ----------
    multiple_subjects : bool
        True  -> the version appended across subjects along the channel dimension.
        False -> a single subject's data (requires `subject`).
    subject : int, optional
        EEG participant number. Required when multiple_subjects is False.
    trial_average : {'all', 'even', 'odd'}
        Which repeat-averaging of the TRAINING set to return. The test set is
        always averaged across all repeats.
    data_dir : str, optional
        Directory holding the prepared files. Defaults to PREPARED_DATA_DIR.

    Returns
    -------
    eeg_train : (n_train_videos, n_channels, n_time)
    eeg_test : (n_test_videos, n_channels, n_time)
    """
    if trial_average not in ('all', 'even', 'odd'):
        raise ValueError(
            f"trial_average must be 'all', 'even' or 'odd', got {trial_average!r}")
    if not multiple_subjects and subject is None:
        raise ValueError("`subject` is required when multiple_subjects is False.")

    data_dir = PREPARED_DATA_DIR

    if multiple_subjects:
        train_file = f'eeg_train_multi_subject_trial_avg-{trial_average}.npy'
        test_file = 'eeg_test_multi_subject.npy'
    else:
        train_file = f'eeg_train_sub-{subject:02d}_trial_avg-{trial_average}.npy'
        test_file = f'eeg_test_sub-{subject:02d}.npy'

    eeg_train = np.load(os.path.join(data_dir, train_file), allow_pickle=True).item()['eeg_train']
    eeg_test = np.load(os.path.join(data_dir, test_file), allow_pickle=True).item()['eeg_test']

    return eeg_train, eeg_test