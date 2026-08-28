import os
import numpy as np
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits

DEFAULT_ALPHAS = np.logspace(-6, 10, 17)
EPS = 1e-8


def load_fmri_data(data_dir, fmri_subject):
    """
    Load pre-computed whole-brain THINGS-fMRI responses for one subject.

    Returns
    -------
    fmri_train : ndarray, shape (n_train, n_voxels)
        Whole-brain fMRI training responses, raw units (not z-scored).
    fmri_test : ndarray, shape (n_test, n_voxels)
        Whole-brain fMRI test responses, averaged across repetitions, raw
        units (not z-scored).
    """
    fmri_dict = np.load(
        os.path.join(data_dir, f'fmri_sub-{fmri_subject:02d}.npy'),
        allow_pickle=True
    ).item()
    return fmri_dict['fmri_train'], fmri_dict['fmri_test']


def load_fmri_roi_data(data_dir, fmri_subject, roi, ncsnr_threshold):
    """
    Load pre-computed THINGS-fMRI responses for one subject, restricted to
    an ROI and further filtered by a noise-ceiling (ncsnr) threshold.

    Starts from the same whole-brain fmri_train/fmri_test arrays as
    load_fmri_data, then applies a dual voxel mask: voxel must be (1) in
    the requested ROI, AND (2) have a noise ceiling strictly greater than
    ncsnr_threshold. Both fields come from the precomputed fmri_sub-XX.npy
    (metadata_fmri['roi'] and metadata_fmri['encoding_model']
    ['noise_ceiling_testset'], saved as-is by THINGS_Precompute_MEG_fMRI_Data.py).

    Returns
    -------
    fmri_train : ndarray, shape (n_train, n_roi_voxels)
        ROI- and ncsnr-filtered fMRI training responses, raw units.
    fmri_test : ndarray, shape (n_test, n_roi_voxels)
        ROI- and ncsnr-filtered fMRI test responses, raw units.
    voxel_idx : ndarray, shape (n_roi_voxels,)
        Whole-brain voxel indices of the selected voxels (i.e. which
        columns of the full fmri_train/fmri_test this corresponds to) --
        keep this around if you need to map results back to whole-brain
        space.
    """
    fmri_dict = np.load(
        os.path.join(data_dir, f'fmri_sub-{fmri_subject:02d}.npy'),
        allow_pickle=True
    ).item()

    roi_idx = np.asarray(fmri_dict['roi'][roi])
    noise_ceiling_testset = fmri_dict['noise_ceiling_testset']
    roi_noise_ceilings = noise_ceiling_testset[roi_idx]

    # Dual condition: voxel is in the ROI AND clears the ncsnr threshold.
    # roi_idx is already restricted to the ROI, so this just sub-selects
    # within it -- adjust '>' to '>=' here if you want an inclusive cutoff.
    ncsnr_mask = roi_noise_ceilings > ncsnr_threshold
    voxel_idx = roi_idx[ncsnr_mask]

    fmri_train = fmri_dict['fmri_train'][:, voxel_idx]
    fmri_test = fmri_dict['fmri_test'][:, voxel_idx]

    return fmri_train, fmri_test, voxel_idx


def load_meg_data(data_dir, tmax=0.6):
    """
    Load pre-computed whole-brain THINGS-MEG responses (sensors concatenated
    across all MEG subjects), realigned to one fMRI subject's training-
    stimulus order, truncated to tmax seconds post stimulus onset.

    Returns
    -------
    meg_train : ndarray, shape (n_train, n_channels, n_time)
        MEG training responses, sensors concatenated across subjects,
        realigned to fmri_subject's training-stimulus order.
    meg_test : ndarray, shape (n_test, n_channels, n_time)
        MEG test responses, sensors concatenated across subjects.
    """
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

    meg_train_native = meg_train_dict['meg_train'][:, :, time_idx]
    meg_test = meg_test_dict['meg_test'][:, :, time_idx]

    # Realign MEG training data to this fMRI subject's training-stimulus order
    train_stimuli_meg_native = list(meg_train_dict['train_stimuli'])
    meg_train_stim_to_idx = {stim: i for i, stim in enumerate(train_stimuli_meg_native)}

    fmri_dict = np.load(
        os.path.join(data_dir, f'fmri_sub-01.npy'),
        allow_pickle=True
    ).item()
    train_stimuli_fmri = fmri_dict['train_stimuli']

    idx_meg = np.array([meg_train_stim_to_idx[stim] for stim in train_stimuli_fmri])
    meg_train = meg_train_native[idx_meg]

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