"""
Searchlight commonality analysis for one AlexNet layer: variance in each
voxel's searchlight fMRI RDM shared between the MEG RDM (per timepoint) and
the layer's feature RDM. Fully vectorized closed-form R^2 -- one matrix
multiply covers the whole (timepoint x voxel) grid, no joblib.

Parameters
----------
subject : fMRI participant (1-3).
layer : AlexNet layer whose feature RDM enters the commonality.
eeg_rdm_metric : {'pearsonr', 'crossnobis', 'decoding_accuracy'} MEG RDM to use.
radius : searchlight radius (mm), must match the fMRI searchlight RDM file.
n_pcs : leading PCs kept from the layer's features.
ncsnr_threshold : voxels below this noise ceiling are left at zero.
"""

import os
import argparse
import time

import numpy as np
from sklearn.metrics import pairwise_distances

from utils import flatten_rdm, get_noise_ceilings, get_test_stimuli, reindex_features

start_time = time.time()
seed = 8
np.random.seed(seed)

alexnet_layers = [
    'features.2', 'features.5', 'features.7', 'features.9', 'features.12',
    'classifier.2', 'classifier.5', 'classifier.6'
]

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--layer', type=str, default='features.2', choices=alexnet_layers)
parser.add_argument('--eeg_rdm_metric', type=str, default='pearsonr',
                     choices=['pearsonr', 'crossnobis', 'decoding_accuracy'])
parser.add_argument('--radius', type=float, default=10.0)
parser.add_argument('--n_pcs', type=int, default=250)
parser.add_argument('--ncsnr_threshold', type=float, default=0.2)
parser.add_argument('--meg_subjects', type=int, nargs='+', default=[1, 2, 3, 4])
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--alexnet_features_dir', type=str,
                     default='/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/visual/alexnet_layerwise')
parser.add_argument('--alexnet_features_file', type=str,
                     default='alexnet_layerwise_features_250_pcs.npy')
args = parser.parse_args()

print('>>> THINGS AlexNet Layer-wise Searchlight Commonality Analysis <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =============================================================================
# MEG RDMs (averaged across MEG subjects)
# =============================================================================
data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'
n_time_points = 141 # number of time points corresponding to tmax = 800 ms

meg_rdms_sum = np.zeros((n_time_points, 4950), dtype=np.float32) # 4950 is the number of unique pairwise distances correspoding to 100 stimuli
for msub in args.meg_subjects:
    meg_rdms_sum += np.load(os.path.join(data_dir, f"{args.eeg_rdm_metric}_rdm_meg_sub-P{msub}.npy"))

meg_rdms = (meg_rdms_sum / len(args.meg_subjects)).astype(np.float32)
print(f" MEG RDMs ({args.eeg_rdm_metric}), averaged over {len(args.meg_subjects)} subjects: {meg_rdms.shape}")

# =============================================================================
# AlexNet layer feature RDM
# =============================================================================
test_stimuli = get_test_stimuli(args.data_dir, fmri_subject=1)

feat_dict = np.load(os.path.join(args.alexnet_features_dir, args.alexnet_features_file),
                    allow_pickle=True).item()
features_test = feat_dict['features']['test'][args.layer][:, :args.n_pcs]
features_stimuli = list(feat_dict['test_stimuli'])
del feat_dict

features_test = reindex_features(features_test, features_stimuli, test_stimuli)
features_rdm = flatten_rdm(pairwise_distances(features_test, metric='cosine'))
print(f"Features RDM ('{args.layer}'): {features_rdm.shape}")

# =============================================================================
# Quantities shared across all voxels
# =============================================================================
meg_centered = meg_rdms - meg_rdms.mean(axis=1, keepdims=True)   # (n_time, n_pairs)
meg_norms = np.linalg.norm(meg_centered, axis=1)                 # (n_time,)

features_centered = features_rdm - features_rdm.mean()
features_norm = np.linalg.norm(features_centered)

# corr(MEG RDM, features RDM) per timepoint -- independent of the fMRI data
r12 = (meg_centered @ features_centered) / (meg_norms * features_norm)  # (n_time,)

# =============================================================================
# fMRI searchlight RDMs
# =============================================================================
fmri_npy_file = (
    f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/searchlight_rdms/'
    f'sub-{args.subject:02d}/searchlight_rdms_r-{args.radius}.npy'
)
fmri_rdms = np.load(fmri_npy_file, mmap_mode='r')
n_voxels, n_pairs_fmri = fmri_rdms.shape
assert n_pairs_fmri == meg_rdms.shape[1], \
    f"Pair count mismatch: fMRI has {n_pairs_fmri}, MEG has {meg_rdms.shape[1]}"
print(f"fMRI searchlight RDMs: {n_voxels} voxels x {n_pairs_fmri} pairs")

noise_ceilings = get_noise_ceilings(args.data_dir, args.subject)
valid_idx = np.where(noise_ceilings >= args.ncsnr_threshold)[0]
n_valid = len(valid_idx)
print(f"{n_valid} / {n_voxels} voxels pass the noise-ceiling criterion.")

r2_scores = np.zeros((n_time_points, n_voxels), dtype=np.float32)

if n_valid > 0:
    fmri_valid = np.array(fmri_rdms[valid_idx, :])  # (n_valid, n_pairs)
    fmri_centered = fmri_valid - fmri_valid.mean(axis=1, keepdims=True)
    fmri_norms = np.linalg.norm(fmri_centered, axis=1)

    # corr(features RDM, each voxel's fMRI RDM): one matrix-vector product
    denom_feat = fmri_norms * features_norm
    denom_feat[denom_feat == 0] = np.nan
    r2_feat = (fmri_centered @ features_centered) / denom_feat  # (n_valid,)
    r2_feat_sq = r2_feat ** 2

    # corr(MEG RDM at each timepoint, each voxel's fMRI RDM): one matrix multiply
    denom = np.outer(meg_norms, fmri_norms)
    denom[denom == 0] = np.nan
    r1 = (meg_centered @ fmri_centered.T) / denom  # (n_time, n_valid)
    r1_sq = r1 ** 2

    r12_col = r12[:, None]
    combined_r2 = (r2_feat_sq[None, :] + r1_sq
                   - 2 * r2_feat[None, :] * r1 * r12_col) / (1 - r12_col ** 2)
    commonality = r2_feat_sq[None, :] + r1_sq - combined_r2

    r2_scores[:, valid_idx] = commonality.astype(np.float32)

print("Commonality analysis complete!")

# =============================================================================
# Saving
# =============================================================================
save_dir = (
    f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/commonality_analysis/'
    f'layerwise_alexnet/eeg_rdm_metric-{args.eeg_rdm_metric}/radius-{args.radius}/'
    f'subject-{args.subject:02d}'
)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'layer-{args.layer}.npy')
np.save(save_path, r2_scores)
print(f"Results saved to: {save_path} (shape {r2_scores.shape})")

print(f"Execution time: {time.time() - start_time:.2f} seconds.")