"""
This script performs the layer-wise commonality analysis between EEG RDMs and fMRI RDMs using features extracted from the AlexNet model
at the whole-brain level.
For computational efficiency, we use closed-form R2 calculations, matrix multiplications, and perform the
analysis only for vertices that pass the NCSNR threshold. The analysis is performed for a specific subject, hemisphere, and fMRI split.

Parameters
----------
subject : int
    The NSD participant pair ID for which the commonality analysis is performed.
hemisphere : str
    The hemisphere of the brain to analyze ('lh' for left hemisphere, 'rh' for right hemisphere).
fmri_split : int
    The split index for fMRI vertices. The fMRI data is split into 21 splits, each containing 7802 vertices, to cover all 163842 fsaverage vertices per hemisphere.

n_neighbours : int
    The number of nearest neighbors to include in the searchlight for each vertex.
layer : str
    The layer of the AlexNet model from which the features are extracted for the commonality analysis.


"""



import numpy as np
import os
import argparse
import h5py
from berg import BERG
from sklearn.metrics import pairwise_distances
import time

# Start time
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
alexnet_layers = [
    'features.2', 'features.5', 'features.7', 'features.9', 'features.12',
    'classifier.2', 'classifier.5', 'classifier.6'
]
parser.add_argument('--layer', type=str, default='features.2', choices=alexnet_layers,
                    help='Layer of the Alexnet model from which the features are extracted for the commonality analysis.')
args = parser.parse_args()

print('>>> AlexNet Layer-wise Commonality Analysis <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)


def corr_1d_vs_2d(x, Y):
    """
    Pearson correlation between a single vector x (n_pairs,) and every row of a
    2D array Y (n_rows, n_pairs). Returns (n_rows,).
    """
    x_c = x - x.mean()
    x_norm = np.linalg.norm(x_c)
    Y_c = Y - Y.mean(axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y_c, axis=1)
    num = Y_c @ x_c
    denom = Y_norm * x_norm
    denom[denom == 0] = np.nan
    return num / denom


# =============================================================================
# 2. Load EEG Predictor Data and Compute Time-Resolved RDMs
# =============================================================================
data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlation_rdms'
eeg_rdms = np.load(os.path.join(data_dir, f"correlation_rdm_eeg_sub-{args.subject}.npy"))   # (n_time_points, n_pairs)
print("Shape of the EEG RDMs: ", eeg_rdms.shape)
n_time_points = eeg_rdms.shape[0]

# =============================================================================
# 3. Loading the layer features data and computing the features RDM
# =============================================================================
features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/alexnet'
features_test = np.load(os.path.join(features_dir, f"sub-{args.subject:02d}_layerwise_fmaps.npy"), allow_pickle=True).item()[args.layer]['test']
features_rdm = flatten_rdm(pairwise_distances(features_test, metric='cosine'))
print("Shape of the features RDM: ", features_rdm.shape)

# =============================================================================
# Precompute quantities that are shared across all vertices
# =============================================================================
eeg_centered = eeg_rdms - eeg_rdms.mean(axis=1, keepdims=True)       # (n_time, n_pairs)
eeg_norms = np.linalg.norm(eeg_centered, axis=1)                     # (n_time,)

features_centered = features_rdm - features_rdm.mean()
features_norm = np.linalg.norm(features_centered)

# r12(t): correlation between EEG RDM and features RDM at each timepoint.
# Depends only on the two predictors -- not on vertex/fMRI data -- so it is
# computed exactly once here
r12 = (eeg_centered @ features_centered) / (eeg_norms * features_norm)  # (n_time,)

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

# =============================================================================
# Filter to vertices passing the noise-ceiling criterion
# =============================================================================
valid_mask = wb_noise_ceilings >= 0.2
valid_idx = np.where(valid_mask)[0]
n_valid = len(valid_idx)
print(f"{n_valid} / {n_vertices} vertices pass the noise-ceiling threshold.")

r2_scores = np.zeros((n_time_points, n_vertices), dtype=np.float32)

if n_valid > 0:
    fmri_valid = fmri_rdms[valid_idx, :]  # (n_valid, n_pairs)

    fmri_centered = fmri_valid - fmri_valid.mean(axis=1, keepdims=True)   # (n_valid, n_pairs)
    fmri_norms = np.linalg.norm(fmri_centered, axis=1)                   # (n_valid,)

    # --- r2_feat: correlation between features RDM and each vertex's fMRI RDM ---
    # One matrix-vector product replaces n_valid separate LinearRegression fits.
    r2_feat = corr_1d_vs_2d(features_rdm, fmri_valid)  # (n_valid,)
    r2_feat_sq = r2_feat ** 2

    # --- r1(t, vertex): correlation between EEG RDM (each timepoint) and each vertex's fMRI RDM ---
    # one big matrix multiplication to cover the entire vertex loop and the timepoint loop.
    numerator = eeg_centered @ fmri_centered.T                 # (n_time, n_valid)
    denom = np.outer(eeg_norms, fmri_norms)                    # (n_time, n_valid)
    denom[denom == 0] = np.nan
    r1 = numerator / denom                                     # (n_time, n_valid)
    r1_sq = r1 ** 2

    # --- Closed-form combined (2-predictor) R^2 and commonality, vectorized ---
    r12_sq = (r12 ** 2)[:, None]              # (n_time, 1)
    r2_feat_row = r2_feat_sq[None, :]         # (1, n_valid)
    r12_col = r12[:, None]                    # (n_time, 1)

    combined_r2 = (r2_feat_row + r1_sq - 2 * r2_feat[None, :] * r1 * r12_col) / (1 - r12_sq)
    commonality = r2_feat_row + r1_sq - combined_r2  # (n_time, n_valid)

    r2_scores[:, valid_idx] = commonality.astype(np.float32)

print("Commonality analysis complete!")

# =============================================================================
# 5. Saving Results
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/commonality_analysis/layerwise_alexnet/eeg_rdm_metric-correlation/wb/subject-{args.subject}/layer-{args.layer}/hemisphere-{args.hemisphere}'
os.makedirs(save_dir, exist_ok=True)

file_name = f'fmri_split-{args.fmri_split}.npy'
np.save(os.path.join(save_dir, file_name), r2_scores)
print(f"Results successfully saved to: {os.path.join(save_dir, file_name)}")

# End time
execution_time = time.time() - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")