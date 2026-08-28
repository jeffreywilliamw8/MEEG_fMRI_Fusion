"""
Decoding-Encoding (DE) fusion -- ROI, partial correlation.
Streamlined and joblib-parallelized.

Unlike SFEF (which reconstructs a phase-1 MEG->fMRI encoder and fits a
second-stage stimulus-feature->predicted-fMRI encoder), DE never predicts
fMRI from MEG directly. Instead, per timepoint, per feature category:

    1. DECODER: MEG(t)  -> stimulus features   (RidgeCV)
    2. ENCODER: decoded features -> fMRI        (RidgeCV, trained on the
       decoder's OWN predictions on the training set, not the ground-truth
       features -- this is what makes it "decoding-encoding" rather than a
       plain feature encoder)
    3. Evaluate by decoding the REAL MEG test set, encoding that into
       ft-fMRI, and partial-correlating against the REAL test fMRI.

This is done for VDNN and LLM simultaneously (one decoder+encoder pair
each), because the quantity actually saved here is the partial correlation
of vdnn-ft-fMRI and llm-ft-fMRI against the true test fMRI -- controlling
each for the other -- not the two categories' plain correlations. This
script is ROI-only and has no whole-brain counterpart: it's a supplementary
analysis, and full-brain per-timepoint decoder+encoder fitting (twice, for
two feature categories) would be prohibitively expensive across ~350k
voxels.

No two-phase split like SFEF: there's no "opposite half" here, since the
decoder and encoder are trained on the SAME stimuli (no crossing) -- --half
selects which one of the two non-random 4320-stimulus halves is used to fit
BOTH the decoder and the encoder, matching SFEF's per-run training-set size
(4320 stimuli) for a fair comparison between methods. Evaluation always uses
the real, separate 100-stimulus THINGS test set (MEG test -> decode -> encode
-> ft-fMRI -> partial-correlate against the real fMRI test set).

Partial correlation: utils.partial_correlations reduces to three column-wise correlations instead of
per-vertex regression residualisation.

-----------------------------
Parameters
----------
fmri_subject : int
half : int
    Which non-random 4320-stimulus half trains BOTH the decoder and the
    encoder (no crossing).
roi, ncsnr_threshold : ROI selection, as in the other ROI THINGS scripts.
n_pcs : int
    Leading PCs kept per feature file (default 250, matching the SFEF scripts).
data_dir, vdnn_features_dir, llm_features_dir : as in the SFEF scripts.
corrs_save_dir : root under which the partial-correlation results are saved.
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

from utils import load_fmri_roi_data, load_meg_data, partial_correlations

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
parser.add_argument('--n_pcs', type=int, default=250)
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--vdnn_features_dir', type=str,
                     default='/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/visual/ViT_B_32')
parser.add_argument('--llm_features_dir', type=str,
                     default='/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/language/'
                             'image_description_embeddings')
parser.add_argument('--corrs_save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/'
                             'results/correlations/decoding_encoding_fusion/roi')
parser.add_argument('--n_jobs', type=int, default=-1)
parser.add_argument('--tmax', type=float, default=0.6)
args = parser.parse_args()

assert args.half in (1, 2), f"--half must be 1 or 2, got {args.half}"

print('>>> Decoding-Encoding Fusion -- Partial Correlation (ROI) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

seed = 8
np.random.seed(seed)
eps = 1e-8
alphas = np.logspace(-6, 10, 17) 

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
# Load VDNN + LLM stimulus features (both, simultaneously -- DE's partial
# correlation needs both categories' ft-fMRI every timepoint), aligned to the
# fMRI master stimulus order via a dict-based lookup.
# =============================================================================
def reindex_features(features, features_stimuli, target_order):
    stim_to_idx = {stim: i for i, stim in enumerate(features_stimuli)}
    idx = np.array([stim_to_idx[stim] for stim in target_order])
    return features[idx].astype(np.float32)


vdnn_feat_path = os.path.join(args.vdnn_features_dir, 'ViT_B_32_features_768_PCs.npy')
vdnn_feat_dict = np.load(vdnn_feat_path, allow_pickle=True).item()
vdnn_train_raw = vdnn_feat_dict['train'][:, :args.n_pcs]
vdnn_test_raw = vdnn_feat_dict['test'][:, :args.n_pcs]
vdnn_train_stimuli = list(vdnn_feat_dict['train_order_map'].keys())
vdnn_test_stimuli = list(vdnn_feat_dict['test_order_map'].keys())
del vdnn_feat_dict

llm_feat_path = os.path.join(args.llm_features_dir, 'language_features_all-mpnet-base-v2.npy')
llm_feat_dict = np.load(llm_feat_path, allow_pickle=True).item()
llm_train_raw = llm_feat_dict['pca_train_features'][:, :args.n_pcs]
llm_test_raw = llm_feat_dict['pca_test_features'][:, :args.n_pcs]
llm_train_stimuli = list(llm_feat_dict['train_stimuli_names'])
llm_test_stimuli = list(llm_feat_dict['test_stimuli_names'])
del llm_feat_dict

vdnn_train_aligned = reindex_features(vdnn_train_raw, vdnn_train_stimuli, train_stimuli_fmri)
vdnn_test = reindex_features(vdnn_test_raw, vdnn_test_stimuli, unique_test_stimuli)
llm_train_aligned = reindex_features(llm_train_raw, llm_train_stimuli, train_stimuli_fmri)
llm_test = reindex_features(llm_test_raw, llm_test_stimuli, unique_test_stimuli)

vdnn_train_half = vdnn_train_aligned[half_idx]
llm_train_half = llm_train_aligned[half_idx]

print(f"\nVDNN features -- half {args.half} train {vdnn_train_half.shape}, test {vdnn_test.shape}")
print(f"LLM  features -- half {args.half} train {llm_train_half.shape}, test {llm_test.shape}")

assert vdnn_train_half.shape[0] == meg_train_half.shape[0], \
    "VDNN train features and MEG train are not the same length after half-slicing."
assert llm_train_half.shape[0] == meg_train_half.shape[0], \
    "LLM train features and MEG train are not the same length after half-slicing."

# =============================================================================
# Per-timepoint: decode VDNN and LLM features from MEG, re-encode into fMRI
# using each decoder's own predictions (not the actual DNN features), then
# partial-correlate the two categories' ft-fMRI against the real test fMRI.
# Parallelized across timepoints with joblib.
# =============================================================================
save_dir = os.path.join(args.corrs_save_dir, f'fmri_sub-{args.fmri_subject:02d}',
                         f'roi-{args.roi}', f'half-{args.half}')
os.makedirs(save_dir, exist_ok=True)


def fit_timepoint(t):
    with threadpool_limits(limits=1):
        meg_t_train = meg_train_half[:, :, t]
        meg_t_test = meg_test[:, :, t]

        # --- VDNN: decode MEG -> VDNN features, then encode decoded VDNN
        # features -> fMRI (trained on the decoder's own train-set output).
        vdnn_decoder = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
        vdnn_decoder.fit(meg_t_train, vdnn_train_half)
        decoded_vdnn_train = meg_t_train @ vdnn_decoder.coef_.T + vdnn_decoder.intercept_
        decoded_vdnn_test = meg_t_test @ vdnn_decoder.coef_.T + vdnn_decoder.intercept_

        vdnn_encoder = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
        vdnn_encoder.fit(decoded_vdnn_train, fmri_train_half)
        vdnn_ft_fmri = decoded_vdnn_test @ vdnn_encoder.coef_.T + vdnn_encoder.intercept_

        # --- LLM: same, decoding/encoding LLM features instead.
        llm_decoder = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
        llm_decoder.fit(meg_t_train, llm_train_half)
        decoded_llm_train = meg_t_train @ llm_decoder.coef_.T + llm_decoder.intercept_
        decoded_llm_test = meg_t_test @ llm_decoder.coef_.T + llm_decoder.intercept_

        llm_encoder = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
        llm_encoder.fit(decoded_llm_train, fmri_train_half)
        llm_ft_fmri = decoded_llm_test @ llm_encoder.coef_.T + llm_encoder.intercept_

        # --- Partial correlation of each category's ft-fMRI against the
        # real test fMRI, controlling for the other category.
        vision_partial, language_partial = partial_correlations(
            fmri_test_roi, vdnn_ft_fmri, llm_ft_fmri
        )

    return vision_partial, language_partial


print(f"\nStarting Decoding-Encoding Fusion (parallelized across {n_time} timepoints, "
      f"n_jobs={args.n_jobs})...")
results = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(fit_timepoint)(t) for t in range(n_time)
)

partial_corr_results = {
    'vdnn_partial_correlation': np.array([r[0] for r in results], dtype=np.float32),   # (n_time, n_voxels)
    'llm_partial_correlation': np.array([r[1] for r in results], dtype=np.float32),  # (n_time, n_voxels)
    'half': args.half,
}

save_path = os.path.join(save_dir, 'partial_correlations.npy')
np.save(save_path, partial_corr_results)
print(f"\nSaved: {save_path}")

voxel_idx_path = os.path.join(save_dir, 'voxel_idx.npy')
np.save(voxel_idx_path, voxel_idx)
print(f"Saved: {voxel_idx_path} (whole-brain indices of the {n_voxels} selected voxels)")

del meg_train_full, meg_train_half, meg_test, fmri_train_roi, fmri_train_half, fmri_test_roi
del vdnn_train_raw, vdnn_test_raw, vdnn_train_aligned, vdnn_train_half, vdnn_test
del llm_train_raw, llm_test_raw, llm_train_aligned, llm_train_half, llm_test
gc.collect()

print(f"\nDecoding-Encoding Fusion Partial Correlation complete! "
      f"Total time: {time.time() - start_time:.2f} seconds.")