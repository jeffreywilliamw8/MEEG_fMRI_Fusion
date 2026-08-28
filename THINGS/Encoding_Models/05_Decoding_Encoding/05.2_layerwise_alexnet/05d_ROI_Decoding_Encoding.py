"""
Decoding-Encoding (DE) fusion -- ROI, AlexNet layerwise


-----------------------------
Parameters
----------
fmri_subject : int
half : int
    Which 4320-stimulus half trains BOTH the decoder and the encoder (no
    crossing).
roi, ncsnr_threshold : ROI selection.
layer : str
    AlexNet layer key into the precomputed layerwise features dict.
data_dir, alexnet_features_dir, alexnet_features_file : as in the SFEF
    layerwise script.
corrs_save_dir : root under which correlations are saved.
n_jobs : parallel workers across timepoints (-1 = all cores).
tmax : MEG truncation (s). Default 0.6.
"""

import argparse
import os
import time
import gc

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits

from utils import load_fmri_roi_data, load_meg_data

start_time = time.time()

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)  # 1, 2, or 3
parser.add_argument('--half', type=int, default=1,
                     help='Which 4320-stimulus half trains BOTH the decoder and '
                          'the encoder (no crossing).')
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--ncsnr_threshold', type=float, default=0.0)
parser.add_argument('--layer', type=str, default='features.2',
                     choices=['features.2', 'features.5', 'features.7', 'features.9',
                              'features.12', 'classifier.2', 'classifier.5', 'classifier.6'],
                     help='AlexNet layer key into the precomputed layerwise features dict.')
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--alexnet_features_dir', type=str,
                     default='/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/visual/alexnet_layerwise')
parser.add_argument('--alexnet_features_file', type=str,
                     default='alexnet_layerwise_features_250_pcs.npy')
parser.add_argument('--corrs_save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/'
                             'results/correlations/decoding_encoding_fusion/roi/layerwise_alexnet')
parser.add_argument('--n_jobs', type=int, default=-1)
parser.add_argument('--tmax', type=float, default=0.6)
args = parser.parse_args()

assert args.half in (1, 2), f"--half must be 1 or 2, got {args.half}"

print('>>> Decoding-Encoding Fusion -- AlexNet Layerwise (ROI) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

seed = 20200220
np.random.seed(seed)
eps = 1e-8
alphas = np.logspace(-6, 10, 17)  # matches the DE partial-correlation script's grid

# =============================================================================
# fMRI master stimulus order (train + test IDs) and the real, ROI+ncsnr-
# filtered fMRI (both train and test -- DE trains its encoder directly on
# fmri_train, unlike SFEF).
# =============================================================================
fmri_dict = np.load(
    os.path.join(args.data_dir, f'fmri_sub-{args.fmri_subject:02d}.npy'), allow_pickle=True
).item()
train_stimuli_fmri = list(fmri_dict['train_stimuli'])     # length 8640, MEG/fMRI master train order
unique_test_stimuli = list(fmri_dict['test_stimuli'])     # length 100
del fmri_dict

n_train = len(train_stimuli_fmri)
assert n_train == 8640, \
    f"Expected 8640 THINGS training stimuli for sub-{args.fmri_subject:02d}, got {n_train}."
half_size = n_train // 2  # 4320
half_idx = slice(0, half_size) if args.half == 1 else slice(half_size, n_train)

fmri_train_roi, fmri_test_roi, voxel_idx = load_fmri_roi_data(
    args.data_dir, args.fmri_subject, args.roi, args.ncsnr_threshold
)
n_voxels = fmri_train_roi.shape[1]
print(f"\n[ROI {args.roi}, ncsnr > {args.ncsnr_threshold}] fMRI train {fmri_train_roi.shape}, "
      f"test {fmri_test_roi.shape} ({n_voxels} voxels)")

if n_voxels == 0:
    print(f"No voxels above the noise ceiling threshold ({args.ncsnr_threshold}) in ROI "
          f"'{args.roi}' (subject {args.fmri_subject}). Nothing to fit -- exiting.")
    raise SystemExit(0)

fmri_train_half = fmri_train_roi[half_idx]
fmri_test_z = (fmri_test_roi - fmri_test_roi.mean(0)) / (fmri_test_roi.std(0) + eps)
print(f"Half {args.half}: fMRI train {fmri_train_half.shape}")

# =============================================================================
# MEG train (sliced to --half) and MEG test (real, separate test set)
# =============================================================================
meg_train_full, meg_test = load_meg_data(args.data_dir, args.fmri_subject, tmax=args.tmax)
n_time = meg_train_full.shape[2]
meg_train_half = meg_train_full[half_idx]
print(f"MEG, half {args.half}: {meg_train_half.shape}, test: {meg_test.shape}, {n_time} timepoints")

assert meg_train_half.shape[0] == fmri_train_half.shape[0], \
    f"Half {args.half}: MEG train ({meg_train_half.shape[0]}) / fMRI train " \
    f"({fmri_train_half.shape[0]}) stimulus count mismatch."
assert meg_test.shape[0] == fmri_test_roi.shape[0], \
    f"MEG test ({meg_test.shape[0]}) / fMRI test ({fmri_test_roi.shape[0]}) stimulus count mismatch."

# =============================================================================
# Load this layer's AlexNet features (one file load) and align to the fMRI
# master stimulus order via a dict-based lookup.
# =============================================================================
def reindex_features(features, features_stimuli, target_order):
    stim_to_idx = {stim: i for i, stim in enumerate(features_stimuli)}
    idx = np.array([stim_to_idx[stim] for stim in target_order])
    return features[idx].astype(np.float32)


feat_path = os.path.join(args.alexnet_features_dir, args.alexnet_features_file)
feat_dict = np.load(feat_path, allow_pickle=True).item()
train_features_raw = feat_dict['features']['train'][args.layer]
test_features_raw = feat_dict['features']['test'][args.layer]
train_features_stimuli = list(feat_dict['train_stimuli'])
test_features_stimuli = list(feat_dict['test_stimuli'])
del feat_dict

print(f"\nLoaded AlexNet layer '{args.layer}' features from {feat_path} "
      f"(train {train_features_raw.shape}, test {test_features_raw.shape})")

train_features_aligned = reindex_features(train_features_raw, train_features_stimuli, train_stimuli_fmri)
test_features = reindex_features(test_features_raw, test_features_stimuli, unique_test_stimuli)

train_features_half = train_features_aligned[half_idx]
print(f"Half {args.half}: features train {train_features_half.shape}, test {test_features.shape}")

assert train_features_half.shape[0] == meg_train_half.shape[0], \
    "Feature train half and MEG train half are not the same length."

# =============================================================================
# Per-timepoint: decode this layer's AlexNet features from MEG, re-encode
# into fMRI using the decoder's OWN train-set predictions, then correlate
# ft-fMRI against the real test fMRI. Parallelized across timepoints with
# joblib.
# =============================================================================
save_dir = os.path.join(args.corrs_save_dir, f'fmri_sub-{args.fmri_subject:02d}',
                         f'roi-{args.roi}', f'half-{args.half}', f'layer-{args.layer}')
os.makedirs(save_dir, exist_ok=True)


def fit_timepoint(t):
    with threadpool_limits(limits=1):
        meg_t_train = meg_train_half[:, :, t]
        meg_t_test = meg_test[:, :, t]

        # Decode MEG -> this layer's features.
        decoder = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
        decoder.fit(meg_t_train, train_features_half)
        decoded_train = meg_t_train @ decoder.coef_.T + decoder.intercept_
        decoded_test = meg_t_test @ decoder.coef_.T + decoder.intercept_

        # Encode decoded (train-set) features -> fMRI.
        encoder = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
        encoder.fit(decoded_train, fmri_train_half)
        ft_fmri = decoded_test @ encoder.coef_.T + encoder.intercept_

        # Plain correlation against the real test fMRI (elementwise
        # z-scored product-and-mean trick, same as the rest of this project).
        ft_fmri_z = (ft_fmri - ft_fmri.mean(0)) / (ft_fmri.std(0) + eps)
        corrs = (ft_fmri_z * fmri_test_z).mean(axis=0)

    return corrs


print(f"\nStarting Decoding-Encoding Fusion (parallelized across {n_time} timepoints, "
      f"n_jobs={args.n_jobs})...")
results = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(fit_timepoint)(t) for t in range(n_time)
)
correlations = np.array(results, dtype=np.float32)  # (n_time, n_voxels)

corrs_path = os.path.join(save_dir, 'correlations.npy')
np.save(corrs_path, correlations)
print(f"\nSaved correlations: {corrs_path} (shape {correlations.shape})")

voxel_idx_path = os.path.join(save_dir, 'voxel_idx.npy')
np.save(voxel_idx_path, voxel_idx)
print(f"Saved: {voxel_idx_path} (whole-brain indices of the {n_voxels} selected voxels)")

del meg_train_full, meg_train_half, meg_test, fmri_train_roi, fmri_train_half, fmri_test_roi
del fmri_test_z, train_features_raw, test_features_raw, train_features_aligned
del train_features_half, test_features
gc.collect()

print(f"\nDecoding-Encoding Fusion (AlexNet layerwise) complete! "
      f"Total time: {time.time() - start_time:.2f} seconds.")