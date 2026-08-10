import os
import h5py
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import RidgeCV
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

def load_fmri_roi_data2(subject, roi, nc_threshold=0.2):
    """
    Loads fMRI data for a given ROI across BOTH hemispheres and any sub-ROIs,
    concatenating them cleanly along the vertex dimension (axis 1).
    
    Parameters:
    - subject: int (e.g. 1, 6, 8)
    - roi: str (e.g. 'V1', 'FFA', 'V4')
    - roi_groups: dict, mappings of composite regions to atomic lists
    - nc_threshold: float, noise ceiling filtration cutoff
    """
    # 1. Determine the target sub-ROIs to load via your structural dictionary lookup
    roi_groups = {
    'V1': ['V1v', 'V1d'],
    'V2': ['V2v', 'V2d'],
    'V3': ['V3v', 'V3d'],
    'hV4': ['hV4'],
    'FFA': ['FFA-1', 'FFA-2'],
    'OFA': ['OFA'],
    'EBA': ['EBA'],
    'PPA': ['PPA'],
    'early': ['early'],
    'midventral': ['midventral'],
    'midparietal': ['midparietal'],
    'midlateral': ['midlateral'],
    'ventral': ['ventral'],
    'lateral': ['lateral'],
    'parietal': ['parietal']
    }
    sub_rois = roi_groups[roi]
    train_vertex_pool = []
    test_vertex_pool = []
    
    # 2. Cycle systematically through both hemispheres and all sub-components
    for hemisphere in ['lh', 'rh']:
        for sub_roi in sub_rois:
            try:
                # Leverage previous code to load each sub-ROI's data with noise ceiling filtration
                f_train, f_test = load_fmri_roi_data(
                    subject=subject, 
                    hemisphere=hemisphere, 
                    roi=sub_roi, 
                    nc_threshold=nc_threshold
                )
                
                # Check to prevent adding empty arrays if an entire ROI fails the noise ceiling check
                if f_train.shape[1] > 0:
                    train_vertex_pool.append(f_train)
                    test_vertex_pool.append(f_test)
                    
            except KeyError:
                # Catches instances where a sub-ROI might only exist in one hemisphere profile
                print(f"Warning: {sub_roi} not found in {hemisphere} metadata index grid. Skipping.")
                continue

    # 3. Handle combining the pools across the spatial vertex dimension (Axis 1)
    if len(train_vertex_pool) > 0:
        combined_train = np.concatenate(train_vertex_pool, axis=1)
        combined_test = np.concatenate(test_vertex_pool, axis=1)
    else:
        # Fallback security if absolutely zero vertices in the entire ROI survive the noise ceiling filter
        print(f"CRITICAL: Zero vertices passed NCSNR >= {nc_threshold} for macro-ROI: {roi}")
        combined_train = np.empty((0, 0), dtype=np.float32)
        combined_test = np.empty((0, 0), dtype=np.float32)
        
    return combined_train, combined_test


def get_roi_noise_ceiling_corr(roi, subject):
    """
    Computes the average correlation noise ceiling for a given macro-ROI 
    by pooling ALL vertices across both hemispheres and all sub-ROIs.
    
    Parameters:
    - roi: str (e.g., 'V1', 'FFA', 'PPA')
    - subject: int (e.g., 1, 4, 5)
    
    Returns:
    - float: The average correlation noise ceiling value across all vertices in the ROI.
    """
    # 1. Structural mapping of macro-ROIs to underlying sub-ROIs
    roi_groups = {
        'V1': ['V1v', 'V1d'],
        'V2': ['V2v', 'V2d'],
        'V3': ['V3v', 'V3d'],
        'hV4': ['hV4'],
        'FFA': ['FFA-1', 'FFA-2'],
        'OFA': ['OFA'],
        'EBA': ['EBA'],
        'PPA': ['PPA']
    }
    
    if roi not in roi_groups:
        raise ValueError(f"Requested ROI '{roi}' is not defined in structural roi_groups mapping.")
        
    sub_rois = roi_groups[roi]
    
    # Initialize your BERG instance (matching your local path structure)
    berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
    
    # Pool to accumulate individual vertex-level correlation ceilings across parts
    pooled_vertex_ceilings = []
    
    # 2. Cycle systematically through both hemispheres and all sub-components
    for hemisphere in ['lh', 'rh']:
        # Fetch metadata once per hemisphere to extract ROI indicators and ncsnr metrics
        metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=subject)
        wb_noise_ceilings = metadata['fmri'][f'{hemisphere}_ncsnr']
        
        for sub_roi in sub_rois:
            try:
                # Isolate target sub-ROI vertex indexes
                roi_idx = metadata['fmri'][f'{hemisphere}_fsaverage_rois'][sub_roi]
                
                # Convert the index list/array into valid vertex positions
                valid_vertices = np.array(roi_idx, dtype=int)
                
                if len(valid_vertices) > 0:
                    # Extract the raw SNR (ncsnr) values for all vertices in the ROI
                    vertex_snrs = wb_noise_ceilings[valid_vertices]
                    
                    # Ensure negative SNR estimations from background noise are clipped to 0
                    vertex_snrs = np.clip(vertex_snrs, 0.0, None)
                    
                    # Taking the square root gives us the correlation noise ceiling (r) directly
                    vertex_corr_ceilings = np.sqrt(vertex_snrs)
                    
                    # Append these calculations to the overarching spatial pool
                    pooled_vertex_ceilings.extend(vertex_corr_ceilings)
                    
            except KeyError:
                # Gracefully skip if a sub-ROI mapping doesn't exist for the current hemisphere
                continue
                
    # 3. Aggregate spatial calculations into a single macro scalar output
    if len(pooled_vertex_ceilings) > 0:
        roi_noise_ceiling_corr = np.mean(pooled_vertex_ceilings)
    else:
        print(f"Warning: Zero vertices found matching criteria for {roi}. Returning 0.0")
        roi_noise_ceiling_corr = 0.0
        
    return float(roi_noise_ceiling_corr)

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
    - correlation_data: array (n_subjects, n_timepoints or n_vertices)
    - alpha: significance level
    - method: 'fdr_bh' for Benjamini-Hochberg, 'bonferroni' for the strict way.
    """
    # 1. Fisher z-transform (handles the r-distribution skew)
    # Note: We use np.clip to avoid infinity if r is exactly 1.0
    z_data = np.arctanh(np.clip(correlation_data, -0.9999, 0.9999))
    
    # 2. Perform 1-sample t-test across subjects at each timepoint
    t_stats, p_values = stats.ttest_1samp(z_data, 0, axis=0)
    
    # 3. Correct for Multiple Comparisons
    # reject is a boolean mask, pvals_corrected are the adjusted p-values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
    
    return pvals_corrected < alpha

def get_significance_mask2(correlation_data, alpha=0.05, method='fdr_bh'):
    """
    Performs Fisher z-transform, 1-sample t-test, and Multiple Comparison Correction.
    
    Parameters:
    - correlation_data: array (n_subjects, n_timepoints or n_vertices)
    - alpha: significance level
    - method: 'fdr_bh' for Benjamini-Hochberg, 'bonferroni' for the strict way.
    """
    # 1. Fisher z-transform (handles the r-distribution skew)
    # Note: We use np.clip to avoid infinity if r is exactly 1.0
    z_data = np.arctanh(np.clip(correlation_data, -0.9999, 0.9999))
    
    # 2. Perform 1-sample t-test across subjects at each timepoint
    t_stats, p_values = stats.ttest_1samp(correlation_data, 0, axis=0)
    
    # 3. Correct for Multiple Comparisons
    # reject is a boolean mask, pvals_corrected are the adjusted p-values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
    
    return pvals_corrected < alpha


def sign_permutation_cluster_test(corr_timecourses, n_permutations=10000, p_thresh=0.1, alpha=0.1):
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


def flatten_rdm(rdm):
    return rdm[np.triu_indices_from(rdm, k=1)]  # k=1 excludes diagonal

def single_feature_rdm(array):
    n_samples = array.shape
    # Fast vectorized calculation of squared Euclidean distances for a single feature vector
    # (array - array.T)^2
    diff = array - array.T
    return np.square(diff, dtype=np.float32)

def get_single_feature_rdms(data):
    n_samples, n_features = data.shape
    n_cells = int(n_samples * (n_samples - 1) / 2)
    feature_specific_rdms = np.empty((n_cells, n_features), dtype=np.float32)

    for j in range(n_features):
        x = data[:, j].reshape(-1, 1)
        d = flatten_rdm(single_feature_rdm(x))
        feature_specific_rdms[:, j] = d

    return feature_specific_rdms

def feature_reweighting_model(feature_specific_rdms, target_rdm, alphas=np.logspace(-6, 5, 30)):
    model = RidgeCV(alphas=alphas)
    model.fit(feature_specific_rdms, target_rdm)
    regression_weights = {}
    regression_weights['coef_'] = []
    regression_weights['intercept_'] = []
    return model

def clip_rdm_values(rdm, metric):
    range_dict = {
        'correlation': (0.0, 2.0),
        'cosine': (0.0, 1.0),
        'euclidean': (0.0, np.inf)
    }
    if metric in range_dict:
        return np.clip(rdm, *range_dict[metric], dtype=np.float32)
    else:
        raise ValueError(f"Unknown target RDM metric: {metric}")

