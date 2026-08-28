"""
Stimulus Feature Encoding Fusion (SFEF), Phase 2 -- whole brain.

"""

import argparse
import os
import time
import gc

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits

from utils import load_meg_data

start_time = time.time()

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)  # 1, 2, or 3
parser.add_argument('--half', type=int, default=1,
                     help='Which half phase 1 was fit on (1 or 2). Phase 2 uses '
                          'the opposite half automatically.')
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--n_splits', type=int, default=500)
parser.add_argument('--dnn_type', type=str, default='vdnn', choices=['vdnn', 'llm'])
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--phase1_weights_dir', type=str, default='/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/regression_weights/stimulus_feature_encoding_fusion/phase_1/wb')
parser.add_argument('--vdnn_features_dir', type=str,
                     default='/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/visual/ViT_B_32')
parser.add_argument('--llm_features_dir', type=str,
                     default='/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/language/'
                             'image_description_embeddings')
parser.add_argument('--corrs_save_dir', type=str, default='/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/correlations/stimulus_feature_encoding_fusion/phase_2/wb')
parser.add_argument('--weights_save_dir', type=str, default='/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/regression_weights/stimulus_feature_encoding_fusion/phase_2/wb')
parser.add_argument('--n_jobs', type=int, default=-1)
parser.add_argument('--tmax', type=float, default=0.6)
args = parser.parse_args()

assert args.half in (1, 2), f"--half must be 1 or 2, got {args.half}"
opposite_half = 2 if args.half == 1 else 1

print('>>> Stimulus Feature Encoding Fusion -- Phase 2 (Whole-Brain) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))
print(f"\nPhase 1 was fit on half {args.half} -> Phase 2 fits on half {opposite_half}'s "
      f"4320 stimuli, and evaluates on the real THINGS test set.")

seed = 8
np.random.seed(seed)
eps = 1e-8
alphas = np.logspace(-6, 10, 20)

# =============================================================================
# fMRI master stimulus order + real test fMRI (from the precomputed data
# loader every other THINGS script reads).
# =============================================================================
fmri_dict = np.load(
    os.path.join(args.data_dir, f'fmri_sub-{args.fmri_subject:02d}.npy'), allow_pickle=True
).item()
train_stimuli_fmri = list(fmri_dict['train_stimuli'])     # length 8640, MEG/fMRI master train order
unique_test_stimuli = list(fmri_dict['test_stimuli'])     # length 100
fmri_test_full = fmri_dict['fmri_test']
del fmri_dict

n_train = len(train_stimuli_fmri)
assert n_train == 8640, \
    f"Expected 8640 THINGS training stimuli for sub-{args.fmri_subject:02d}, got {n_train}."
half_size = n_train // 2  # 4320
opp_idx = slice(0, half_size) if opposite_half == 1 else slice(half_size, n_train)

fmri_test_splits = np.array_split(fmri_test_full, args.n_splits, axis=1)
fmri_test = fmri_test_splits[args.fmri_split - 1]
n_voxels = fmri_test.shape[1]
print(f"\nfMRI test, split {args.fmri_split}/{args.n_splits}: {fmri_test.shape} ({n_voxels} voxels)")

fmri_test_z = (fmri_test - fmri_test.mean(0)) / (fmri_test.std(0) + eps)

# =============================================================================
# MEG (train, all 8640 stimuli in fMRI-aligned order -- sliced to the
# opposite half below)
# =============================================================================
meg_train_full, _ = load_meg_data(args.data_dir, args.fmri_subject, tmax=args.tmax)
n_time = meg_train_full.shape[2]
meg_opposite = meg_train_full[opp_idx]
print(f"MEG, opposite half ({opposite_half}): {meg_opposite.shape}")

# =============================================================================
# Load the phase-1 MEG-to-fMRI encoder weights (fit on --half, this voxel split)
# =============================================================================
phase_1_weights_path = os.path.join(
    args.phase1_weights_dir, f'fmri_sub-{args.fmri_subject:02d}', f'half-{args.half}',
    f'fmri_split-{args.fmri_split:03d}.npy'
)
phase_1_weights = np.load(phase_1_weights_path, allow_pickle=True).item()
phase_1_coef = phase_1_weights['coef']            # (n_time, n_voxels, n_channels)
phase_1_intercept = phase_1_weights['intercept']  # (n_time, n_voxels)
print(f"\nLoaded phase-1 weights: {phase_1_weights_path} "
      f"(coef {phase_1_coef.shape}, intercept {phase_1_intercept.shape})")

# =============================================================================
# Load this run's stimulus features and align to the fMRI master stimulus
# =============================================================================
def reindex_features(features, features_stimuli, target_order):
    stim_to_idx = {stim: i for i, stim in enumerate(features_stimuli)}
    idx = np.array([stim_to_idx[stim] for stim in target_order])
    return features[idx].astype(np.float32)

if args.dnn_type == 'vdnn':
    feat_path = os.path.join(args.vdnn_features_dir, 'ViT_B_32_features_768_PCs.npy')
    feat_dict = np.load(feat_path, allow_pickle=True).item()
    train_features_raw = feat_dict['train'][:, :250]
    test_features_raw = feat_dict['test'][:, :250]
    train_features_stimuli = list(feat_dict['train_order_map'].keys())
    test_features_stimuli = list(feat_dict['test_order_map'].keys())
else:  # 'llm'
    feat_path = os.path.join(args.llm_features_dir, 'language_features_all-mpnet-base-v2.npy')
    feat_dict = np.load(feat_path, allow_pickle=True).item()
    train_features_raw = feat_dict['pca_train_features'][:, :250]
    test_features_raw = feat_dict['pca_test_features'][:, :250]
    train_features_stimuli = list(feat_dict['train_stimuli_names'])
    test_features_stimuli = list(feat_dict['test_stimuli_names'])
del feat_dict

print(f"\nLoaded '{args.dnn_type}' features from {feat_path} "
      f"(train {train_features_raw.shape}, test {test_features_raw.shape}")

train_features_aligned = reindex_features(train_features_raw, train_features_stimuli, train_stimuli_fmri)
test_features = reindex_features(test_features_raw, test_features_stimuli, unique_test_stimuli)

features_train_opp = train_features_aligned[opp_idx]
print(f"train shape = {features_train_opp.shape}, test shape = {test_features.shape}")

# =============================================================================
# Per-timepoint: reconstruct the phase-1 MEG->fMRI encoder, apply it to the
# opposite half's MEG to get t-fMRI, fit features->t-fMRI, evaluate on the
# real test set. Parallelized across timepoints with joblib.
# =============================================================================
save_dir_suffix = os.path.join(f'fmri_sub-{args.fmri_subject:02d}', f'half-{args.half}',
                                f'dnn_type-{args.dnn_type}')
corrs_save_dir = os.path.join(args.corrs_save_dir, save_dir_suffix)
weights_save_dir = os.path.join(args.weights_save_dir, save_dir_suffix)
os.makedirs(corrs_save_dir, exist_ok=True)
os.makedirs(weights_save_dir, exist_ok=True)
file_name = f'fmri_split-{args.fmri_split:03d}.npy'


def fit_timepoint(t):
    with threadpool_limits(limits=1):
        # Reconstruct the phase-1 MEG-to-fMRI encoder at this timepoint and
        # apply it to the opposite half's MEG -> this half's predicted fMRI.
        t_fmri = (meg_opposite[:, :, t] @ phase_1_coef[t].T) + phase_1_intercept[t]

        # Fit a new encoder: this model's opposite-half features -> t-fMRI.
        encoding_model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
        encoding_model.fit(features_train_opp, t_fmri)

        coef_ = encoding_model.coef_.astype(np.float32)
        intercept_ = encoding_model.intercept_.astype(np.float32)

        # Evaluate on the real THINGS test set.
        ft_fmri = test_features @ encoding_model.coef_.T + encoding_model.intercept_
        ft_fmri_z = (ft_fmri - ft_fmri.mean(0)) / (ft_fmri.std(0) + eps)
        corrs = (ft_fmri_z * fmri_test_z).mean(axis=0)

    return coef_, intercept_, corrs


print(f"\nStarting SFEF Phase 2 (parallelized across {n_time} timepoints, "
      f"n_jobs={args.n_jobs})...")
results = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(fit_timepoint)(t) for t in range(n_time)
)

encoding_weights = {
    'coef': np.array([r[0] for r in results], dtype=np.float32),       # (n_time, n_voxels, n_pcs)
    'intercept': np.array([r[1] for r in results], dtype=np.float32),  # (n_time, n_voxels)
}
correlations = np.array([r[2] for r in results], dtype=np.float32)  # (n_time, n_voxels)

np.save(os.path.join(weights_save_dir, file_name), encoding_weights)
np.save(os.path.join(corrs_save_dir, file_name), correlations)
print(f"\nSaved weights: {os.path.join(weights_save_dir, file_name)}")
print(f"Saved correlations: {os.path.join(corrs_save_dir, file_name)} (shape {correlations.shape})")

del meg_train_full, meg_opposite, fmri_test_full, fmri_test, fmri_test_z
del train_features_raw, test_features_raw, train_features_aligned, features_train_opp, test_features
gc.collect()

print(f"\nSFEF Phase 2 complete! Total time: {time.time() - start_time:.2f} seconds.")