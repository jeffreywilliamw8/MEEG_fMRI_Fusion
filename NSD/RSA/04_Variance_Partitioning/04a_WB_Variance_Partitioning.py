"""
This script performs whole-brain variance partitioning analysis of VDNN and LLM features on EEG and fMRI data.
For computational efficiency, we use closed-form R2 calculations, matrix multiplications, and perform the
analysis only for vertices that pass the NCSNR threshold. The analysis is performed for a specific subject, hemisphere, and fMRI split.

"""





import numpy as np
import os
import argparse
import h5py
from berg import BERG
from sklearn.metrics import pairwise_distances
import time

start_time = time.time()
seed = 8
np.random.seed(seed)

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh')
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--n_neighbours', type=int, default=100)
parser.add_argument('--eeg_rdm_metric', type=str, default='pearsonr', choices=['pearsonr', 'crossnobis', 'decoding_accuracy'])
args = parser.parse_args()

print('>>> RSA Variance Partitioning: Vision vs. Language (Vectorized, closed-form R^2) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)


def corr_1d_vs_1d(x, y):
    xc = x - x.mean()
    yc = y - y.mean()
    return (xc @ yc) / (np.linalg.norm(xc) * np.linalg.norm(yc))


def corr_1d_vs_2d(x, Y):
    """Correlation between a single vector x (n_pairs,) and every row of Y (n_rows, n_pairs)."""
    x_c = x - x.mean()
    x_norm = np.linalg.norm(x_c)
    Y_c = Y - Y.mean(axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y_c, axis=1)
    denom = Y_norm * x_norm
    denom[denom == 0] = np.nan
    return (Y_c @ x_c) / denom


def resid_1d(y, x):
    """OLS residual of y after regressing out x (both 1D, n_pairs,)."""
    yc = y - y.mean()
    xc = x - x.mean()
    beta = (xc @ yc) / (xc @ xc)
    return yc - beta * xc


def resid_2d(Y, x):
    """OLS residual of every row of Y (n_rows, n_pairs) after regressing out x (n_pairs,)."""
    xc = x - x.mean()
    xc_ss = xc @ xc
    Y_c = Y - Y.mean(axis=1, keepdims=True)
    beta = (Y_c @ xc) / xc_ss  # (n_rows,)
    return Y_c - beta[:, None] * xc[None, :]


def two_predictor_r2(r1, r2, r12):
    """Closed-form R^2 for a 2-predictor OLS regression, given pairwise correlations."""
    denom = 1 - r12 ** 2
    return (r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * r12) / denom


# =============================================================================
# 2. Load EEG Predictor Data and Compute Time-Resolved RDMs
# =============================================================================
data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/eeg_rdms'
eeg_rdms = np.load(os.path.join(data_dir, f"{args.eeg_rdm_metric}_rdm_eeg_sub-{args.subject}.npy"))
print("Shape of the EEG RDMs: ", eeg_rdms.shape)
n_time_points = eeg_rdms.shape[0]

# =============================================================================
# 3. Load Static Model Predictor Data (Vision DNN and LLM) and Compute RDMs
# =============================================================================
vision_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/vit_b_32'
vision_test = np.load(os.path.join(vision_dir, f'fmri_sub-{args.subject:02d}_fmaps.npy'), allow_pickle=True).item()['fmaps_test']
vision_rdm = flatten_rdm(pairwise_distances(vision_test, metric='cosine'))
print("Shape of the Vision DNN RDM: ", vision_rdm.shape)
del vision_test

lang_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/llms'
lang_test = np.load(os.path.join(lang_dir, f"llm_embeddings_sub-{args.subject:02d}.npy"), allow_pickle=True).item()['llm_embeddings_test']
lang_rdm = flatten_rdm(pairwise_distances(lang_test, metric='cosine'))
print("Shape of the LLM RDM: ", lang_rdm.shape)
del lang_test

# =============================================================================
# Precompute quantities that are CONSTANT across the entire script (previously
# recomputed inside the vertex x timepoint double loop, ~2.8 million times)
# =============================================================================
r_vl = corr_1d_vs_1d(vision_rdm, lang_rdm)
vision_minus_lang = resid_1d(vision_rdm, lang_rdm)
lang_minus_vis = resid_1d(lang_rdm, vision_rdm)

# =============================================================================
# Precompute quantities that depend on TIMEPOINT ONLY (previously recomputed
# once per vertex per timepoint -- now computed once per timepoint, vectorized
# across all 359 timepoints simultaneously)
# =============================================================================
eeg_minus_lang = resid_2d(eeg_rdms, lang_rdm)   # (n_time, n_pairs)
eeg_minus_vis = resid_2d(eeg_rdms, vision_rdm)  # (n_time, n_pairs)

r_ve = corr_1d_vs_2d(vision_rdm, eeg_rdms)  # (n_time,) corr(vision, eeg(t))
r_le = corr_1d_vs_2d(lang_rdm, eeg_rdms)    # (n_time,) corr(lang, eeg(t))

# ==================================
# fMRI Noise ceilings
# ==================================
berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=args.subject)
wb_noise_ceilings = metadata['fmri'][f'{args.hemisphere}_ncsnr']
wb_noise_ceilings = wb_noise_ceilings[7802 * (args.fmri_split - 1):7802 * args.fmri_split]

# =============================================================================
# Loading the Precomputed fMRI RDMs
# =============================================================================
fmri_h5_file = os.path.join(
    f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/fmri_searchlight_rdms/n_neighbours-{args.n_neighbours}',
    f'fmri_sub-{args.subject}_hemi-{args.hemisphere}_rdms.h5'
)
with h5py.File(fmri_h5_file, 'r') as f:
    dset = f['rdms']
    fmri_rdms = dset[7802 * (args.fmri_split - 1):7802 * args.fmri_split, :]

n_vertices = fmri_rdms.shape[0]
print(f"Loaded fMRI RDMs for {n_vertices} vertices.")

results = {
    'unique_vision': np.zeros((n_time_points, n_vertices), dtype=np.float32),
    'unique_language': np.zeros((n_time_points, n_vertices), dtype=np.float32),
    'shared_vision_language': np.zeros((n_time_points, n_vertices), dtype=np.float32),
}

valid_mask = wb_noise_ceilings >= 0.2
valid_idx = np.where(valid_mask)[0]
n_valid = len(valid_idx)
print(f"{n_valid} / {n_vertices} vertices pass the noise-ceiling criterion.")

if n_valid > 0:
    fmri_valid = fmri_rdms[valid_idx, :]  # (n_valid, n_pairs)

    # =============================================================================
    # Precompute quantities that depend on VERTEX ONLY
    # =============================================================================
    fmri_minus_lang = resid_2d(fmri_valid, lang_rdm)   # (n_valid, n_pairs)
    fmri_minus_vis = resid_2d(fmri_valid, vision_rdm)  # (n_valid, n_pairs)

    r_vision = corr_1d_vs_2d(vision_rdm, fmri_valid)  # (n_valid,)
    r_lang = corr_1d_vs_2d(lang_rdm, fmri_valid)      # (n_valid,)

    vision_r2 = corr_1d_vs_2d(vision_minus_lang, fmri_minus_lang) ** 2   # (n_valid,)
    language_r2 = corr_1d_vs_2d(lang_minus_vis, fmri_minus_vis) ** 2     # (n_valid,)
    features_r2 = two_predictor_r2(r_vision, r_lang, r_vl)               # (n_valid,)

    # =============================================================================
    # Quantities that depend on BOTH vertex and timepoint: each is now exactly
    # ONE big matrix multiply across the full (n_time x n_valid) grid.
    # =============================================================================
    def corr_2d_vs_2d(A, B):
        """corr(A[i], B[j]) for all i,j. A: (n_a, n_pairs), B: (n_b, n_pairs) -> (n_a, n_b)."""
        A_c = A - A.mean(axis=1, keepdims=True)
        B_c = B - B.mean(axis=1, keepdims=True)
        A_norm = np.linalg.norm(A_c, axis=1)
        B_norm = np.linalg.norm(B_c, axis=1)
        denom = np.outer(A_norm, B_norm)
        denom[denom == 0] = np.nan
        return (A_c @ B_c.T) / denom

    eeg_r2_lang = corr_2d_vs_2d(eeg_minus_lang, fmri_minus_lang) ** 2  # (n_time, n_valid)
    eeg_r2_vis = corr_2d_vs_2d(eeg_minus_vis, fmri_minus_vis) ** 2     # (n_time, n_valid)
    eeg_r2_raw = corr_2d_vs_2d(eeg_rdms, fmri_valid) ** 2              # (n_time, n_valid)

    # Recover signed correlations (needed for the cross term in the 2-predictor formula)
    r_vision_minus_lang = corr_1d_vs_2d(vision_minus_lang, fmri_minus_lang)  # (n_valid,) signed
    r_lang_minus_vis = corr_1d_vs_2d(lang_minus_vis, fmri_minus_vis)        # (n_valid,) signed
    r_eeg_minus_lang = corr_2d_vs_2d(eeg_minus_lang, fmri_minus_lang)        # (n_time, n_valid) signed
    r_eeg_minus_vis = corr_2d_vs_2d(eeg_minus_vis, fmri_minus_vis)           # (n_time, n_valid) signed

    # r12 for unique_vision block: corr(vision_minus_lang, eeg_minus_lang(t)) -- depends on t only
    r12_vision_eeg = corr_1d_vs_2d(vision_minus_lang, eeg_minus_lang)  # (n_time,)
    # r12 for unique_language block: corr(lang_minus_vis, eeg_minus_vis(t)) -- depends on t only
    r12_lang_eeg = corr_1d_vs_2d(lang_minus_vis, eeg_minus_vis)  # (n_time,)

    vision_eeg_r2 = two_predictor_r2(
        r_vision_minus_lang[None, :], r_eeg_minus_lang, r12_vision_eeg[:, None]
    )  # (n_time, n_valid)
    language_eeg_r2 = two_predictor_r2(
        r_lang_minus_vis[None, :], r_eeg_minus_vis, r12_lang_eeg[:, None]
    )  # (n_time, n_valid)

    unique_vision = vision_r2[None, :] + eeg_r2_lang - vision_eeg_r2
    unique_language = language_r2[None, :] + eeg_r2_vis - language_eeg_r2

    # --- shared_vision_language: needs the general 3-predictor (vision, lang, eeg(t)) formula ---
    shared_vision_language = np.zeros((n_time_points, n_valid), dtype=np.float32)
    for t in range(n_time_points):
        Rxx = np.array([
            [1.0, r_vl, r_ve[t]],
            [r_vl, 1.0, r_le[t]],
            [r_ve[t], r_le[t], 1.0],
        ])
        Rxx_inv = np.linalg.inv(Rxx)
        r_e_signed = corr_1d_vs_2d(eeg_rdms[t], fmri_valid)  # (n_valid,) signed, cheap (one matvec)
        r_vec = np.stack([r_vision, r_lang, r_e_signed], axis=1)  # (n_valid, 3)
        features_eeg_r2_t = np.einsum('vi,ij,vj->v', r_vec, Rxx_inv, r_vec)
        shared_vision_language[t] = features_r2 + eeg_r2_raw[t] - features_eeg_r2_t

    results['unique_vision'][:, valid_idx] = unique_vision.astype(np.float32)
    results['unique_language'][:, valid_idx] = unique_language.astype(np.float32)
    results['shared_vision_language'][:, valid_idx] = shared_vision_language.astype(np.float32)

print("Variance partitioning analysis complete!")

# =============================================================================
# 5. Saving Results
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/variance_partitioning/eeg_rdm_metric-{args.eeg_rdm_metric}/wb/subject-{args.subject}/hemisphere-{args.hemisphere}'
os.makedirs(save_dir, exist_ok=True)

file_name = f'fmri_split-{args.fmri_split}.npy'
np.save(os.path.join(save_dir, file_name), results)
print(f"Results successfully saved to: {os.path.join(save_dir, file_name)}")

execution_time = time.time() - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")