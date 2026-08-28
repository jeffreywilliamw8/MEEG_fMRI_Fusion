import os
import numpy as np
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits

DEFAULT_ALPHAS = np.logspace(-6, 10, 17)
EPS = 1e-8

def load_fmri_wb_data(fmri_subject):
    """
    Load pre-computed whole-brain THINGS-fMRI responses for one subject.

    Returns
    -------
    fmri_train : ndarray, shape (n_train, n_voxels)
    fmri_test : ndarray, shape (n_test, n_voxels)
    """
    data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'
    fmri_dict = np.load(
        os.path.join(data_dir, f'fmri_sub-{fmri_subject:02d}.npy'),
        allow_pickle=True
    ).item()
    return fmri_dict['fmri_train'], fmri_dict['fmri_test']


def load_fmri_roi_data(fmri_subject, roi, ncsnr_threshold):
    """
    Load pre-computed THINGS-fMRI responses for one subject, restricted to
    an ROI and further filtered by a noise-ceiling (ncsnr) threshold.


    Returns
    -------
    fmri_train : ndarray, shape (n_train, n_roi_voxels)
        ROI- and ncsnr-filtered fMRI training responses
    fmri_test : ndarray, shape (n_test, n_roi_voxels)
        ROI- and ncsnr-filtered fMRI test responses
    voxel_idx : ndarray, shape (n_roi_voxels,)
        Whole-brain voxel indices of the selected voxels
    """
    data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'
    fmri_dict = np.load(
        os.path.join(data_dir, f'fmri_sub-{fmri_subject:02d}.npy'),
        allow_pickle=True
    ).item()

    roi_idx = np.asarray(fmri_dict['roi'][roi])
    noise_ceiling_testset = fmri_dict['noise_ceiling_testset']
    roi_noise_ceilings = noise_ceiling_testset[roi_idx]

    # Dual condition: voxel is in the ROI AND clears the ncsnr threshold.

    ncsnr_mask = roi_noise_ceilings >= ncsnr_threshold
    voxel_idx = roi_idx[ncsnr_mask]

    fmri_train = fmri_dict['fmri_train'][:, voxel_idx]
    fmri_test = fmri_dict['fmri_test'][:, voxel_idx]

    return fmri_train, fmri_test, voxel_idx


def load_meg_data(tmax=0.6):
    """
    Load pre-computed whole-brain THINGS-MEG responses (sensors concatenated
    across all MEG subjects)

    Returns
    -------
    meg_train : ndarray, shape (n_train, n_channels, n_time)
    meg_test : ndarray, shape (n_test, n_channels, n_time)
    """
    data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'
    meg_train_dict = np.load(os.path.join(data_dir, 'meg_train.npy'), allow_pickle=True).item()
    meg_test_dict = np.load(os.path.join(data_dir, 'meg_test.npy'), allow_pickle=True).item()

    times = meg_train_dict['times']
    if tmax > times.max():
        raise ValueError(
            f"Requested tmax={tmax} exceeds the tmax the MEG data was "
            f"pre-computed with ({times.max():.3f}s). Re-run "
            f"THINGS_Precompute_MEG_fMRI_Data.py with a larger --tmax first."
        )
    time_idx = np.where(times <= tmax)[0]

    meg_train = meg_train_dict['meg_train'][:, :, time_idx]
    meg_test = meg_test_dict['meg_test'][:, :, time_idx]

    return meg_train, meg_test


def get_meg_times(data_dir, tmax=0.6):
    """
    Return the MEG time vector (seconds post stimulus onset), truncated to
    tmax -- matches the time axis of whatever load_meg_data(..., tmax=tmax)
    returns. Mirrors the NSD project's get_eeg_times() helper, for use by
    plotting scripts that need an x-axis without loading the full MEG data.
    """
    meg_train_dict = np.load(os.path.join(data_dir, 'meg_train.npy'), allow_pickle=True).item()
    times = meg_train_dict['times']
    if tmax > times.max():
        raise ValueError(
            f"Requested tmax={tmax} exceeds the tmax the MEG data was "
            f"pre-computed with ({times.max():.3f}s). Re-run "
            f"THINGS_Precompute_MEG_fMRI_Data.py with a larger --tmax first."
        )
    return times[times <= tmax]


def fit_predict_correlate_timepoint(t, meg_train, meg_test, fmri_train, fmri_test_z,
                                     alphas=DEFAULT_ALPHAS, eps=EPS, return_weights=False):
    """
    Fit a RidgeCV MEG-to-fMRI encoding model at one timepoint, predict the
    test fMRI responses, and correlate against the actual test fMRI
    responses. Used as the joblib-parallelized worker in both
    THINGS_Whole_Brain_Encoding_Fusion.py and THINGS_ROI_Encoding_Fusion.py -- identical
    either way, since a RidgeCV model doesn't care how many voxels its
    targets span.

    threadpool_limits(1) caps BLAS/OpenMP threads inside this worker so
    joblib's process-level parallelism across timepoints doesn't get
    oversubscribed by numpy/scikit-learn's own internal multithreading.

    Parameters
    ----------
    t : int
        Timepoint index into the time (3rd) axis of meg_train/meg_test.
    meg_train : ndarray, shape (n_train, n_channels, n_time)
    meg_test : ndarray, shape (n_test, n_channels, n_time)
    fmri_train : ndarray, shape (n_train, n_voxels)
        Raw-units training targets (whole-brain or ROI-restricted).
    fmri_test_z : ndarray, shape (n_test, n_voxels)
        Centered/normalized test targets, precomputed once outside the
        per-timepoint loop (same fmri_test_z reused at every timepoint).
    alphas : array-like
        Ridge regularization strengths RidgeCV chooses from.
    eps : float
        Numerical-stability constant for standard deviations.
    return_weights : bool, default False
        If True, also return the fitted RidgeCV weights (coef_, intercept_)
        for this timepoint. Off by default (whole-brain runs don't need
        this and it's a lot of extra data to carry back through joblib);
        THINGS_ROI_Encoding_Fusion.py turns it on since ROI voxel counts are small
        enough that saving weights per timepoint is cheap.

    Returns
    -------
    corr_t : ndarray, shape (n_voxels,)
        Per-voxel correlation between predicted and actual test fMRI
        responses at this timepoint.
    coef_t : ndarray, shape (n_voxels, n_channels), only if return_weights
        RidgeCV regression weights for this timepoint.
    intercept_t : ndarray, shape (n_voxels,), only if return_weights
        RidgeCV intercepts for this timepoint.
    """
    with threadpool_limits(limits=1):
        meg2fmri = RidgeCV(alphas=alphas, alpha_per_target=True)
        meg2fmri.fit(meg_train[:, :, t], fmri_train)

        # Predict via explicit matrix multiplication instead of .predict()
        t_fmri = meg_test[:, :, t] @ meg2fmri.coef_.T + meg2fmri.intercept_

        # Center/normalize, then correlate via the elementwise z-scored
        # product-and-mean trick (== diag(A.T @ B) / n_test, but without
        # forming the unused off-diagonal terms).
        t_fmri_z = (t_fmri - t_fmri.mean(0)) / (t_fmri.std(0) + eps)
        corr_t = (t_fmri_z * fmri_test_z).mean(axis=0)

    if return_weights:
        return corr_t, meg2fmri.coef_, meg2fmri.intercept_
    return corr_t


def fit_timepoint_weights(t, meg_train, fmri_train, alphas=DEFAULT_ALPHAS):
    """
    Fit a RidgeCV MEG-to-fMRI encoding model at one timepoint and return
    only its weights -- no prediction, no correlation, no test data
    involved at all.
 
    Parameters
    ----------
    t : int
        Timepoint index into the time (3rd) axis of meg_train.
    meg_train : ndarray, shape (n_train, n_channels, n_time)
    fmri_train : ndarray, shape (n_train, n_voxels)
        Raw-units training targets (whole-brain, voxel-split chunk).
    alphas : array-like
        Ridge regularization strengths RidgeCV chooses from.
 
    Returns
    -------
    coef_t : ndarray, shape (n_voxels, n_channels)
        RidgeCV regression weights for this timepoint.
    intercept_t : ndarray, shape (n_voxels,)
        RidgeCV intercepts for this timepoint.
    """
    with threadpool_limits(limits=1):
        meg2fmri = RidgeCV(alphas=alphas, alpha_per_target=True)
        meg2fmri.fit(meg_train[:, :, t], fmri_train)
    return meg2fmri.coef_, meg2fmri.intercept_




"""
Closed-form partial correlation, for fast computation

With a single control variable, the partial correlation has an exact closed
form:

    r(y, x1 | x2) = (r_y_x1 - r_y_x2 * r_x1_x2)
                    / sqrt((1 - r_y_x2^2) * (1 - r_x1_x2^2))

This is algebraically identical to explicitly residualising y and x1 on x2
via linear regression and correlating the residuals, but reduces to three
column-wise correlations that vectorise across all columns (voxels) at
once. Verified to ~1e-9 against the residualisation baseline in
VL_v2_partial_correlation_utils.py's __main__ block; the math here is
unchanged, just re-hosted under the THINGS_ prefix.
"""

import numpy as np

EPS = 1e-8


def _safe_std(X):
    """
    Column standard deviations, with exact-zero columns replaced by 1.

    Deliberately does NOT use the `std + 1e-8` idiom -- adding a constant to
    every denominator biases every correlation by ~1e-8 relative. Substituting
    1.0 only where std is exactly zero is exact everywhere the correlation is
    defined, and yields 0 (rather than inf/nan) for a constant column.
    """
    std = X.std(axis=0)
    return np.where(std > 0, std, 1.0)


def columnwise_corr(A, B):
    """
    Pearson correlation between matching columns of A and B.

    A, B : (n_samples, n_columns)
    returns : (n_columns,)
    """
    A_z = (A - A.mean(axis=0)) / _safe_std(A)
    B_z = (B - B.mean(axis=0)) / _safe_std(B)
    return (A_z * B_z).mean(axis=0)


def partial_correlations(y, x1, x2, eps=EPS):
    """
    Column-wise partial correlations between y and each of two predictors,
    each controlling for the other.

    y, x1, x2 : (n_samples, n_columns)
        Column j of each is one voxel's values across test stimuli.

    Returns
    -------
    r_x1 : (n_columns,)  partial correlation of y with x1, controlling for x2
    r_x2 : (n_columns,)  partial correlation of y with x2, controlling for x1
    """
    r_y_x1 = columnwise_corr(y, x1)
    r_y_x2 = columnwise_corr(y, x2)
    r_x1_x2 = columnwise_corr(x1, x2)

    denom_x1 = np.sqrt(np.clip((1 - r_y_x2 ** 2) * (1 - r_x1_x2 ** 2), eps, None))
    denom_x2 = np.sqrt(np.clip((1 - r_y_x1 ** 2) * (1 - r_x1_x2 ** 2), eps, None))

    r_x1 = (r_y_x1 - r_y_x2 * r_x1_x2) / denom_x1
    r_x2 = (r_y_x2 - r_y_x1 * r_x1_x2) / denom_x2

    return r_x1, r_x2