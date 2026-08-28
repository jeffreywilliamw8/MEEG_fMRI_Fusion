"""
Stimulus Feature Encoding Fusion (SFEF), ROI VDNN vs LLM partial correlation.


-----------------------------
Parameters
----------
fmri_subject : int
half : int
    Which half PHASE 1 was fit on for the vdnn/llm SFEF phase 2 weights
    being loaded here (must match the --half those phase 2 runs used).
roi, ncsnr_threshold : ROI selection.
n_pcs : int
    Leading PCs kept per feature file -- must match what SFEF phase 2 was
    fit with (default 250, same as the SFEF/DE scripts).
data_dir : precomputed fMRI/MEG data root.
sfef_weights_dir : root SFEF phase 2 weights were saved to (the
    --weights_save_dir passed to THINGS_ROI_SFEF_Phase2.py).
vdnn_features_dir / llm_features_dir : as in the SFEF scripts.
corrs_save_dir : root under which the partial-correlation results are saved.
n_jobs : parallel workers across timepoints (-1 = all cores).
"""

import argparse
import os
import time
import gc

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from utils import load_fmri_roi_data, partial_correlations

start_time = time.time()

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)  # 1, 2, or 3
parser.add_argument('--half', type=int, default=1,
                     help='Which half PHASE 1 was fit on for the vdnn/llm SFEF '
                          'phase 2 weights being loaded.')
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--ncsnr_threshold', type=float, default=0.0)
parser.add_argument('--n_pcs', type=int, default=250)
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--sfef_weights_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/'
                             'results/regression_weights/stimulus_feature_encoding_fusion/phase_2/roi')
parser.add_argument('--vdnn_features_dir', type=str,
                     default='/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/visual/ViT_B_32')
parser.add_argument('--llm_features_dir', type=str,
                     default='/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/language/'
                             'image_description_embeddings')
parser.add_argument('--corrs_save_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/'
                             'results/correlations/stimulus_feature_encoding_fusion/'
                             'partial_correlation/roi')
parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()

assert args.half in (1, 2), f"--half must be 1 or 2, got {args.half}"

print('>>> Stimulus Feature Encoding Fusion -- VDNN vs LLM Partial Correlation (ROI) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

seed = 8
np.random.seed(seed)
eps = 1e-8

# =============================================================================
# fMRI master test-stimulus order and the real, ROI+ncsnr-filtered test fMRI.
# =============================================================================
fmri_dict = np.load(
    os.path.join(args.data_dir, f'fmri_sub-{args.fmri_subject:02d}.npy'), allow_pickle=True
).item()
unique_test_stimuli = list(fmri_dict['test_stimuli'])  # length 100
del fmri_dict

_, fmri_test_roi, voxel_idx = load_fmri_roi_data(
    args.data_dir, args.fmri_subject, args.roi, args.ncsnr_threshold
)
n_voxels = fmri_test_roi.shape[1]
print(f"\n[ROI {args.roi}, ncsnr > {args.ncsnr_threshold}] fMRI test {fmri_test_roi.shape} "
      f"({n_voxels} voxels)")

if n_voxels == 0:
    print(f"No voxels above the noise ceiling threshold ({args.ncsnr_threshold}) in ROI "
          f"'{args.roi}' (subject {args.fmri_subject}). Nothing to correlate -- exiting.")
    raise SystemExit(0)

# =============================================================================
# Load this ROI/half's vdnn and llm SFEF phase 2 weights.
# =============================================================================
def load_phase2_weights(dnn_type):
    path = os.path.join(
        args.sfef_weights_dir, f'fmri_sub-{args.fmri_subject:02d}', f'roi-{args.roi}',
        f'half-{args.half}', f'dnn_type-{dnn_type}', 'regression_weights.npy'
    )
    d = np.load(path, allow_pickle=True).item()
    return d['coef'], d['intercept'], path  # (n_time, n_voxels, n_pcs), (n_time, n_voxels)


vdnn_coef, vdnn_intercept, vdnn_weights_path = load_phase2_weights('vdnn')
llm_coef, llm_intercept, llm_weights_path = load_phase2_weights('llm')
print(f"\nLoaded VDNN phase-2 weights: {vdnn_weights_path} (coef {vdnn_coef.shape})")
print(f"Loaded LLM  phase-2 weights: {llm_weights_path} (coef {llm_coef.shape})")

n_time = vdnn_coef.shape[0]
assert llm_coef.shape[0] == n_time, \
    f"VDNN weights have {n_time} timepoints, LLM weights have {llm_coef.shape[0]} -- " \
    f"were they fit on the same MEG data (same --tmax)?"
assert vdnn_coef.shape[1] == n_voxels and llm_coef.shape[1] == n_voxels, \
    f"Phase-2 weights cover a different number of voxels ({vdnn_coef.shape[1]} / " \
    f"{llm_coef.shape[1]}) than this ROI has ({n_voxels}). Did --roi/--ncsnr_threshold " \
    f"match what SFEF phase 2 was fit with?"

# =============================================================================
# Load and align each model's real test-set features to the fMRI master
# test-stimulus order via a dict-based lookup.
# =============================================================================
def reindex_features(features, features_stimuli, target_order):
    stim_to_idx = {stim: i for i, stim in enumerate(features_stimuli)}
    idx = np.array([stim_to_idx[stim] for stim in target_order])
    return features[idx].astype(np.float32)


vdnn_feat_path = os.path.join(args.vdnn_features_dir, 'ViT_B_32_features_768_PCs.npy')
vdnn_feat_dict = np.load(vdnn_feat_path, allow_pickle=True).item()
vdnn_test_raw = vdnn_feat_dict['test'][:, :args.n_pcs]
vdnn_test_stimuli = list(vdnn_feat_dict['test_order_map'].keys())
del vdnn_feat_dict

llm_feat_path = os.path.join(args.llm_features_dir, 'language_features_all-mpnet-base-v2.npy')
llm_feat_dict = np.load(llm_feat_path, allow_pickle=True).item()
llm_test_raw = llm_feat_dict['pca_test_features'][:, :args.n_pcs]
llm_test_stimuli = list(llm_feat_dict['test_stimuli_names'])
del llm_feat_dict

vdnn_test = reindex_features(vdnn_test_raw, vdnn_test_stimuli, unique_test_stimuli)
llm_test = reindex_features(llm_test_raw, llm_test_stimuli, unique_test_stimuli)
print(f"\nVDNN test features: {vdnn_test.shape}  |  LLM test features: {llm_test.shape}")

assert vdnn_coef.shape[2] == vdnn_test.shape[1], \
    f"VDNN weights expect {vdnn_coef.shape[2]} PCs, loaded test features have " \
    f"{vdnn_test.shape[1]} -- does --n_pcs match what SFEF phase 2 was fit with?"
assert llm_coef.shape[2] == llm_test.shape[1], \
    f"LLM weights expect {llm_coef.shape[2]} PCs, loaded test features have " \
    f"{llm_test.shape[1]} -- does --n_pcs match what SFEF phase 2 was fit with?"

# =============================================================================
# Per-timepoint: predict vdnn_ft_fmri / llm_ft_fmri from the already-fitted
# phase-2 weights, then partial-correlate each against the real test fMRI.
# =============================================================================
def fit_timepoint(t):
    with threadpool_limits(limits=1):
        vdnn_ft_fmri = vdnn_test @ vdnn_coef[t].T + vdnn_intercept[t]
        llm_ft_fmri = llm_test @ llm_coef[t].T + llm_intercept[t]
        vision_partial, language_partial = partial_correlations(
            fmri_test_roi, vdnn_ft_fmri, llm_ft_fmri
        )
    return vision_partial, language_partial


print(f"\nStarting VDNN vs LLM partial correlation (parallelized across {n_time} timepoints, "
      f"n_jobs={args.n_jobs})...")
results = Parallel(n_jobs=args.n_jobs, verbose=10)(
    delayed(fit_timepoint)(t) for t in range(n_time)
)

partial_corr_results = {
    'vdnn_partial_correlation': np.array([r[0] for r in results], dtype=np.float32),   # (n_time, n_voxels)
    'llm_partial_correlation': np.array([r[1] for r in results], dtype=np.float32),  # (n_time, n_voxels)
}

save_dir = os.path.join(args.corrs_save_dir, f'fmri_sub-{args.fmri_subject:02d}',
                         f'roi-{args.roi}', f'half-{args.half}')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'partial_correlations.npy')
np.save(save_path, partial_corr_results)
print(f"\nSaved: {save_path}")

voxel_idx_path = os.path.join(save_dir, 'voxel_idx.npy')
np.save(voxel_idx_path, voxel_idx)
print(f"Saved: {voxel_idx_path} (whole-brain indices of the {n_voxels} selected voxels)")

del fmri_test_roi, vdnn_coef, vdnn_intercept, llm_coef, llm_intercept
del vdnn_test_raw, llm_test_raw, vdnn_test, llm_test
gc.collect()

print(f"\nSFEF VDNN vs LLM Partial Correlation complete! "
      f"Total time: {time.time() - start_time:.2f} seconds.")