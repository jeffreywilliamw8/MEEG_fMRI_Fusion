"""
Searchlight RSA variance partitioning: splits the variance each voxel's
searchlight fMRI RDM shares with the MEG RDM into a part unique to the VDNN
RDM, a part unique to the LLM RDM, and a part shared between them. Fully
vectorized closed-form R^2 across the whole (timepoint x voxel) grid; 
only the 3-predictor shared term loops over timepoints.

Parameters
----------
subject : fMRI participant (1-3).
eeg_rdm_metric : {'pearsonr', 'crossnobis', 'decoding_accuracy'} MEG RDM to use.
radius : searchlight radius (mm), must match the fMRI searchlight RDM file.
n_pcs : leading PCs kept from each feature file.
ncsnr_threshold : voxels below this noise ceiling are left at zero.
"""

import os
import argparse
import time

import numpy as np
from sklearn.metrics import pairwise_distances

from utils import (corr_1d_vs_1d, corr_1d_vs_2d, corr_2d_vs_2d, flatten_rdm,
                               get_noise_ceilings, get_test_stimuli,
                               reindex_features, resid_1d, resid_2d, two_predictor_r2)

start_time = time.time()
seed = 8
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--eeg_rdm_metric', type=str, default='pearsonr',
                     choices=['pearsonr', 'crossnobis', 'decoding_accuracy'])
parser.add_argument('--radius', type=float, default=10.0)
parser.add_argument('--n_pcs', type=int, default=250)
parser.add_argument('--ncsnr_threshold', type=float, default=20.0)
parser.add_argument('--meg_subjects', type=int, nargs='+', default=[1, 2, 3, 4])
parser.add_argument('--data_dir', type=str,
                     default='/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data')
parser.add_argument('--vdnn_features_dir', type=str,
                     default='/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/visual/ViT_B_32')
parser.add_argument('--llm_features_dir', type=str,
                     default='/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/language/'
                             'image_description_embeddings')
args = parser.parse_args()

print('>>> THINGS RSA Variance Partitioning: VDNN vs LLM (Searchlight) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =============================================================================
# MEG RDMs (averaged across MEG subjects)
# =============================================================================
data_dir = '/scratch/jeffreykatab/Projects/fusion/THINGS/prepared_data'
n_time_points = 141 # number of time points corresponding to tmax = 600 ms

meg_rdms_sum = np.zeros((n_time_points, 4950), dtype=np.float32) # 4950 is the number of unique pairwise distances correspoding to 100 stimuli
for msub in args.meg_subjects:
    meg_rdms_sum += np.load(os.path.join(data_dir, f"{args.eeg_rdm_metric}_rdm_meg_sub-P{msub}.npy"))

meg_rdms = (meg_rdms_sum / len(args.meg_subjects)).astype(np.float32)
print(f" MEG RDMs ({args.eeg_rdm_metric}), averaged over {len(args.meg_subjects)} subjects: {meg_rdms.shape}")

# =============================================================================
# VDNN and LLM feature RDMs
# =============================================================================
test_stimuli = get_test_stimuli(args.data_dir, fmri_subject=1)

vdnn_dict = np.load(os.path.join(args.vdnn_features_dir, 'ViT_B_32_features_768_PCs.npy'),
                    allow_pickle=True).item()
vdnn_test = vdnn_dict['test'][:, :args.n_pcs]
vdnn_stimuli = list(vdnn_dict['test_order_map'].keys())
del vdnn_dict
vdnn_test = reindex_features(vdnn_test, vdnn_stimuli, test_stimuli)
vision_rdm = flatten_rdm(pairwise_distances(vdnn_test, metric='cosine'))
del vdnn_test
print(f"VDNN RDM: {vision_rdm.shape}")

llm_dict = np.load(os.path.join(args.llm_features_dir, 'language_features_all-mpnet-base-v2.npy'),
                   allow_pickle=True).item()
llm_test = llm_dict['pca_test_features'][:, :args.n_pcs]
llm_stimuli = list(llm_dict['test_stimuli_names'])
del llm_dict
llm_test = reindex_features(llm_test, llm_stimuli, test_stimuli)
lang_rdm = flatten_rdm(pairwise_distances(llm_test, metric='cosine'))
del llm_test
print(f"LLM RDM: {lang_rdm.shape}")

# =============================================================================
# Quantities constant across the whole script (not dependent on time or voxels)
# =============================================================================
r_vl = corr_1d_vs_1d(vision_rdm, lang_rdm) # Correlation between VDNN RDM and LLM RDM
vision_minus_lang = resid_1d(vision_rdm, lang_rdm) # VDNN RDM with LLM RDM regressed out from it(, n_pairs)
lang_minus_vis = resid_1d(lang_rdm, vision_rdm) # LLM RDM with VDNN RDM regressed out from it(, n_pairs)

# Timepoint-only quantities, vectorized across all timepoints at once
meg_minus_lang = resid_2d(meg_rdms, lang_rdm)   # MEG RDMs with LLM RDMs regressed out from it(n_time, n_pairs)
meg_minus_vis = resid_2d(meg_rdms, vision_rdm)  # MEG RDMs with VDNN RDMs regressed out from it (n_time, n_pairs)
r_ve = corr_1d_vs_2d(vision_rdm, meg_rdms)      # R2 score for the VDNN RDMs (n_time,)
r_le = corr_1d_vs_2d(lang_rdm, meg_rdms)        # R2 score for the LLM RDMs(n_time,)

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

results = {
    'unique_vision': np.zeros((n_time_points, n_voxels), dtype=np.float32),
    'unique_language': np.zeros((n_time_points, n_voxels), dtype=np.float32),
}

if n_valid > 0:
    fmri_valid = np.array(fmri_rdms[valid_idx, :])  # (n_valid, n_pairs)

    # Voxel-only quantities
    fmri_minus_lang = resid_2d(fmri_valid, lang_rdm)
    fmri_minus_vis = resid_2d(fmri_valid, vision_rdm)

    r_vision = corr_1d_vs_2d(vision_rdm, fmri_valid)  # (n_valid,)
    r_lang = corr_1d_vs_2d(lang_rdm, fmri_valid)      # (n_valid,)

    r_vision_minus_lang = corr_1d_vs_2d(vision_minus_lang, fmri_minus_lang)  
    r_lang_minus_vis = corr_1d_vs_2d(lang_minus_vis, fmri_minus_vis)         

    vision_r2 = r_vision_minus_lang ** 2
    language_r2 = r_lang_minus_vis ** 2
    features_r2 = two_predictor_r2(r_vision, r_lang, r_vl)

    # Timepoint x voxel quantities: one matrix multiply each
    r_meg_minus_lang = corr_2d_vs_2d(meg_minus_lang, fmri_minus_lang)  # (n_time, n_valid)
    r_meg_minus_vis = corr_2d_vs_2d(meg_minus_vis, fmri_minus_vis)     # (n_time, n_valid)
    meg_r2_lang = r_meg_minus_lang ** 2
    meg_r2_vis = r_meg_minus_vis ** 2
    r_meg_raw = corr_2d_vs_2d(meg_rdms, fmri_valid)                    # (n_time, n_valid)
    meg_r2_raw = r_meg_raw ** 2

    r12_vision_meg = corr_1d_vs_2d(vision_minus_lang, meg_minus_lang)  # (n_time,)
    r12_lang_meg = corr_1d_vs_2d(lang_minus_vis, meg_minus_vis)        # (n_time,)

    vision_meg_r2 = two_predictor_r2(
        r_vision_minus_lang[None, :], r_meg_minus_lang, r12_vision_meg[:, None])
    language_meg_r2 = two_predictor_r2(
        r_lang_minus_vis[None, :], r_meg_minus_vis, r12_lang_meg[:, None])

    unique_vision = vision_r2[None, :] + meg_r2_lang - vision_meg_r2
    unique_language = language_r2[None, :] + meg_r2_vis - language_meg_r2

    results['unique_vision'][:, valid_idx] = unique_vision.astype(np.float32)
    results['unique_language'][:, valid_idx] = unique_language.astype(np.float32)

print("Variance partitioning complete!")

# =============================================================================
# Saving
# =============================================================================
save_dir = (
    f'/scratch/jeffreykatab/Projects/fusion/THINGS/RSA/results/variance_partitioning/'
    f'eeg_rdm_metric-{args.eeg_rdm_metric}/radius-{args.radius}'
)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'subject-{args.subject:02d}.npy')
np.save(save_path, results)
print(f"Results saved to: {save_path}")

print(f"Execution time: {time.time() - start_time:.2f} seconds.")