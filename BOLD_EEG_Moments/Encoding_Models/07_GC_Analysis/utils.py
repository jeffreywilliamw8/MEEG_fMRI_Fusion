import numpy as np
import pickle
from scipy.ndimage import label


def load_fmri_data(subject, hemisphere, roi, threshold=0.0):

    if hemisphere == 'both':

        fmri_file_train_left = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-'+subject+'/prepared_betas/sub-'+subject+'_organized_betas_task-train_hemi-left_normalized.pkl'
        fmri_file_test_left = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-'+subject+'/prepared_betas/sub-'+subject+'_organized_betas_task-test_hemi-left_normalized.pkl'
        fmri_file_train_right = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-'+subject+'/prepared_betas/sub-'+subject+'_organized_betas_task-train_hemi-right_normalized.pkl'
        fmri_file_test_right = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-'+subject+'/prepared_betas/sub-'+subject+'_organized_betas_task-test_hemi-right_normalized.pkl'

        rois_masks_dir_l = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_roi_masks_hemi-left.npy'
        noise_ceiling_file_l = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_noiseceiling_space-fsaverage_task-test_hemi-left_n-10.pkl'
        rois_masks_dir_r = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_roi_masks_hemi-right.npy'
        noise_ceiling_file_r = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_noiseceiling_space-fsaverage_task-test_hemi-right_n-10.pkl'



        f_train_l = open(fmri_file_train_left, 'rb')
        f_test_l = open(fmri_file_test_left, 'rb')
        f_train_r = open(fmri_file_train_right, 'rb')
        f_test_r = open(fmri_file_test_right, 'rb')



        fmri_T_l = np.float32(pickle.load(f_train_l)[0])
        fmri_t_l = np.float32(pickle.load(f_test_l)[0])
        fmri_T_r = np.float32(pickle.load(f_train_r)[0])
        fmri_t_r = np.float32(pickle.load(f_test_r)[0])



        noise_ceiling_l = pickle.load(open(noise_ceiling_file_l.format(subject, subject, 'left'), 'rb'))[1]
        noise_ceiling_r = pickle.load(open(noise_ceiling_file_r.format(subject, subject, 'right'), 'rb'))[1]

        if roi=='WB':
            fmri_train_l = np.mean(fmri_T_l, axis=1, dtype=np.float32)[:, noise_ceiling_l >= threshold]
            fmri_test_l = np.mean(fmri_t_l, axis=1, dtype=np.float32)[:, noise_ceiling_l >= threshold]
            fmri_train_r = np.mean(fmri_T_r, axis=1, dtype=np.float32)[:, noise_ceiling_r >= threshold]
            fmri_test_r = np.mean(fmri_t_r, axis=1, dtype=np.float32)[:, noise_ceiling_r >= threshold]
            del fmri_T_l, fmri_t_l, fmri_T_r, fmri_t_r
            return np.concatenate([fmri_train_l, fmri_test_l], axis=0), np.concatenate([fmri_train_r, fmri_test_r], axis=0)
        
        else:
            mask_l = np.load(rois_masks_dir_l.format(subject, subject, 'left'), allow_pickle=True).item()[roi]
            mask_r = np.load(rois_masks_dir_r.format(subject, subject, 'right'), allow_pickle=True).item()[roi]
            # Selecting vertices from the whole-brain surface based on 2 conditions: belonging to the ROI and noise ceiling above the threshold
            # Condition 1: Non-zero values of the mask (selecting vertices belonging to the ROI)
            condition1_l = mask_l != 0
            condition1_r = mask_r != 0


            # Condition 2: Vertices with a noise ceiling greater than the threshold
            condition2_l = noise_ceiling_l >= threshold
            condition2_r = noise_ceiling_r >= threshold

            # Combined conditions
            combined_condition_l = condition1_l & condition2_l
            combined_condition_r = condition1_r & condition2_r

            # Selecting the vertices that satisfy both conditions and Averaging the fMRI responses across repetitions
            fmri_train_l = np.mean(fmri_T_l, axis=1, dtype=np.float32)[:, combined_condition_l]
            fmri_test_l = np.mean(fmri_t_l, axis=1, dtype=np.float32)[:, combined_condition_l]
            del fmri_T_l, fmri_t_l
            fmri_train_r = np.mean(fmri_T_r, axis=1, dtype=np.float32)[:, combined_condition_r]
            fmri_test_r = np.mean(fmri_t_r, axis=1, dtype=np.float32)[:, combined_condition_r]
            del fmri_T_r, fmri_t_r

            print("Number of selected vertices - left hemisphere: (train, test)", fmri_train_l.shape[1], fmri_test_l.shape[1])
            print("Number of selected vertices - right hemisphere: (train, test)", fmri_train_r.shape[1], fmri_test_r.shape[1])
            return np.concatenate([fmri_train_l, fmri_test_l], axis=0), np.concatenate([fmri_train_r, fmri_test_r], axis=0)
    
    else:

        fmri_file_train = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-'+subject+'/prepared_betas/sub-'+subject+'_organized_betas_task-train_hemi-'+hemisphere+'_normalized.pkl'
        fmri_file_test = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-'+subject+'/prepared_betas/sub-'+subject+'_organized_betas_task-test_hemi-'+hemisphere+'_normalized.pkl'

        rois_masks_dir = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_roi_masks_hemi-{}.npy'
        noise_ceiling_file = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/fsaverage/GLM/sub-{}/prepared_betas/sub-{}_noiseceiling_space-fsaverage_task-test_hemi-{}_n-10.pkl'

        f_train = open(fmri_file_train, 'rb')
        f_test = open(fmri_file_test, 'rb')

        fmri_T = np.float32(pickle.load(f_train)[0])
        fmri_t = np.float32(pickle.load(f_test)[0])

        noise_ceiling = pickle.load(open(noise_ceiling_file.format(subject, subject, hemisphere), 'rb'))[1]

        if roi == 'WB':
            fmri_train = np.mean(fmri_T, axis=1, dtype=np.float32)[:, noise_ceiling >= threshold]
            fmri_test = np.mean(fmri_t, axis=1, dtype=np.float32)[:, noise_ceiling >= threshold]
            del fmri_T, fmri_t
            return fmri_train, fmri_test

        else:

            mask = np.load(rois_masks_dir.format(subject, subject, hemisphere), allow_pickle=True).item()[roi]
            # Selecting vertices from the whole-brain surface based on 2 conditions: belonging to the ROI and noise ceiling above the threshold
            # Condition 1: Non-zero values of the mask (selecting vertices belonging to the ROI)
            condition1 = mask != 0


            # Condition 2: Vertices with a noise ceiling greater than the threshold
            condition2 = noise_ceiling >= threshold

            # Combined conditions
            combined_condition = condition1 & condition2

            # Selecting the vertices that satisfy both conditions and Averaging the fMRI responses across repetitions
            fmri_train = np.mean(fmri_T, axis=1, dtype=np.float32)[:, combined_condition]
            fmri_test = np.mean(fmri_t, axis=1, dtype=np.float32)[:, combined_condition]
            del fmri_T, fmri_t

            print("Number of selected vertices : (train, test)", fmri_train.shape[1], fmri_test.shape[1])

            return fmri_train, fmri_test
        

def load_eeg_times(sfreq):
    eeg_path = f'/scratch/giffordale95/projects/eeg_moments/dataset/preprocessed_data/eeg/sub-01/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-{sfreq:04d}/preprocessed_data.npy'
    times = np.load(eeg_path, allow_pickle=True).item()['times']  # Loading the time vector from the EEG data
    return times


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