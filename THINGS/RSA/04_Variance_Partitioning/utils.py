"""
Helpers for the THINGS RSA scripts: MEG RDM loading/averaging across
subjects, ROI voxel-index lookup, stimulus-feature alignment, and the
closed-form correlation/residual math used by the commonality and variance
partitioning analyses.
"""

import os

import numpy as np

MEG_RDM_DIRS = {
    'pearsonr': '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data',
    'crossnobis': '/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/crossnobis_rdms',
    'decoding_accuracy': '/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/meg_rdms',
}

MEG_RDM_FILES = {
    'pearsonr': 'meg_rdms_sub-P{}.npy',
    'crossnobis': 'crossnobis_rdm_meg_sub-P{}.npy',
    'decoding_accuracy': 'decoding_accuracy_rdm_meg_sub-P{}.npy',
}


def load_averaged_meg_rdms(metric, meg_subjects=(1, 2, 3, 4), rdm_dir=None, time_point=None):
    """
    Load each MEG subject's RDM time series for `metric` and average across subjects.

    Returns (rdms, times). rdms is (n_time, n_pairs), or (n_pairs,) if time_point is given.
    'pearsonr' RDMs are stored as full (n_time, n_stim, n_stim) matrices and are converted
    to the same flattened upper-triangle ordering the other two metrics already use.
    """
    rdm_dir = rdm_dir or MEG_RDM_DIRS[metric]
    rdm_sum, times = None, None

    for msub in meg_subjects:
        d = np.load(os.path.join(rdm_dir, MEG_RDM_FILES[metric].format(msub)),
                    allow_pickle=True).item()
        rdms = d['rdms']
        times = d['times'] if times is None else times

        if metric == 'pearsonr':
            triu = np.triu_indices(rdms.shape[1], k=1)
            rdms = rdms[:, triu[0], triu[1]]  # (n_time, n_pairs)

        if time_point is not None:
            rdms = rdms[time_point]

        rdm_sum = rdms.astype(np.float64) if rdm_sum is None else rdm_sum + rdms

    return (rdm_sum / len(meg_subjects)).astype(np.float32), times


def get_roi_voxel_idx(data_dir, fmri_subject, roi, ncsnr_threshold=0.0):
    """Whole-brain voxel indices of an ROI's voxels passing the noise-ceiling threshold."""
    d = np.load(os.path.join(data_dir, f'fmri_sub-{fmri_subject:02d}.npy'),
                allow_pickle=True).item()
    roi_idx = np.asarray(d['roi'][roi])
    ncsnr = d['noise_ceiling_testset'][roi_idx]
    del d
    return roi_idx[ncsnr > ncsnr_threshold]


def get_noise_ceilings(data_dir, fmri_subject):
    """Whole-brain test-set noise ceiling array."""
    d = np.load(os.path.join(data_dir, f'fmri_sub-{fmri_subject:02d}.npy'),
                allow_pickle=True).item()
    nc = np.asarray(d['noise_ceiling_testset'])
    del d
    return nc


def get_test_stimuli(data_dir, fmri_subject=1):
    """Canonical test-stimulus order the fMRI/MEG RDMs are built in."""
    d = np.load(os.path.join(data_dir, f'fmri_sub-{fmri_subject:02d}.npy'),
                allow_pickle=True).item()
    stimuli = list(d['test_stimuli'])
    del d
    return stimuli


def reindex_features(features, features_stimuli, target_order):
    """Reorder feature rows to match target_order via a dict lookup."""
    stim_to_idx = {stim: i for i, stim in enumerate(features_stimuli)}
    idx = np.array([stim_to_idx[stim] for stim in target_order])
    return features[idx].astype(np.float32)


# =============================================================================
# RDM math
# =============================================================================
def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)


def corr_1d_vs_1d(x, y):
    xc = x - x.mean()
    yc = y - y.mean()
    return (xc @ yc) / (np.linalg.norm(xc) * np.linalg.norm(yc))


def corr_1d_vs_2d(x, Y):
    """Correlation between vector x (n_pairs,) and every row of Y (n_rows, n_pairs)."""
    x_c = x - x.mean()
    x_norm = np.linalg.norm(x_c)
    Y_c = Y - Y.mean(axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y_c, axis=1)
    denom = Y_norm * x_norm
    denom[denom == 0] = np.nan
    return (Y_c @ x_c) / denom


def corr_2d_vs_2d(A, B):
    """corr(A[i], B[j]) for all i, j. A: (n_a, n_pairs), B: (n_b, n_pairs) -> (n_a, n_b)."""
    A_c = A - A.mean(axis=1, keepdims=True)
    B_c = B - B.mean(axis=1, keepdims=True)
    denom = np.outer(np.linalg.norm(A_c, axis=1), np.linalg.norm(B_c, axis=1))
    denom[denom == 0] = np.nan
    return (A_c @ B_c.T) / denom


def resid_1d(y, x):
    """OLS residual of y after regressing out x (both 1D)."""
    yc = y - y.mean()
    xc = x - x.mean()
    return yc - ((xc @ yc) / (xc @ xc)) * xc


def resid_2d(Y, x):
    """OLS residual of every row of Y (n_rows, n_pairs) after regressing out x (n_pairs,)."""
    xc = x - x.mean()
    Y_c = Y - Y.mean(axis=1, keepdims=True)
    beta = (Y_c @ xc) / (xc @ xc)
    return Y_c - beta[:, None] * xc[None, :]


def two_predictor_r2(r1, r2, r12):
    """Closed-form R^2 for a 2-predictor OLS regression, given pairwise correlations."""
    return (r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * r12) / (1 - r12 ** 2)