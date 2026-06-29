"""
This script implements Feature-Reweighted RSA using joblib for parallelization.
Follows implementation of: Kaniuth, Philipp, and Martin N. Hebart. "Feature-reweighted representational similarity analysis: A method for 
improving the fit between computational models, brains, and behavior." NeuroImage 257 (2022): 119294.
"""

import os
# --- Force single-threaded linear algebra BEFORE numpy/sklearn initialization ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from sklearn.metrics import pairwise_distances
import argparse
import random
from scipy.stats import spearmanr
from utils import load_fmri_roi_data2
from sklearn.linear_model import RidgeCV
from joblib import Parallel, delayed
import time 

# Start time
start_time = time.time()

seed = 8
np.random.seed(seed)
random.seed(seed)

#=============================================================================
# Input arguments
#=============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--target', type=str, default='fmri', choices=['fmri', 'eeg'], help='Choose either fMRI or EEG RDM as the target for feature reweighting')
parser.add_argument('--target_rdm_metric', type=str, default='correlation', choices=['correlation', 'cosine', 'euclidean'], help='Distance metric for target RDM')
args = parser.parse_args()

print('>>> Feature-Reweighted RSA Fusion (Parallelized) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

#=============================================================================
# Loading the fMRI data
#=============================================================================
fmri_train, fmri_test = load_fmri_roi_data2(args.subject, args.roi, nc_threshold=0.20)

#=============================================================================
# Helper Functions
#=============================================================================
def flatten_rdm(rdm):
    return rdm[np.triu_indices_from(rdm, k=1)]  # k=1 excludes diagonal

def single_feature_rdm(array):
    n_samples = array.shape
    # Fast vectorized calculation of squared Euclidean distances for a single feature vector
    # (array - array.T)^2
    diff = array - array.T
    return np.square(diff, dtype=np.float32)

def get_single_feature_rdms(data):
    n_samples, n_features = data.shape
    n_cells = int(n_samples * (n_samples - 1) / 2)
    feature_specific_rdms = np.empty((n_cells, n_features), dtype=np.float32)

    for j in range(n_features):
        x = data[:, j].reshape(-1, 1)
        d = flatten_rdm(single_feature_rdm(x))
        feature_specific_rdms[:, j] = d

    return feature_specific_rdms

def feature_reweighting_model(feature_specific_rdms, target_rdm, alphas=np.logspace(-6, 5, 30)):
    model = RidgeCV(alphas=alphas)
    model.fit(feature_specific_rdms, target_rdm)
    regression_weights = {}
    regression_weights['coef_'] = []
    regression_weights['intercept_'] = []
    return model

def clip_rdm_values(rdm, metric):
    range_dict = {
        'correlation': (0.0, 2.0),
        'cosine': (0.0, 2.0),
        'euclidean': (0.0, np.inf)
    }
    if metric in range_dict:
        return np.clip(rdm, *range_dict[metric], dtype=np.float32)
    else:
        raise ValueError(f"Unknown target RDM metric: {metric}")

#=============================================================================
# Parallel Job Workers
#=============================================================================
def process_time_point_fmri_target(eeg_train_t, eeg_test_t, target_rdm_train, target_rdm_test, metric):
    """Worker handling a single timepoint calculation when fMRI is the target."""
    feature_specific_rdm_train = get_single_feature_rdms(eeg_train_t)
    feature_specific_rdm_test = get_single_feature_rdms(eeg_test_t)
    
    model = feature_reweighting_model(feature_specific_rdm_train, target_rdm_train)
    feature_reweighted_rdm = model.predict(feature_specific_rdm_test)
    
    clipped_rdm = clip_rdm_values(feature_reweighted_rdm, metric)
    corr = spearmanr(clipped_rdm, target_rdm_test).correlation
    
    # Return both the metric and the model parameters as a structured dictionary
    return {
        'corr': corr,
        'coef': model.coef_,
        'intercept': model.intercept_
    }

def process_time_point_eeg_target(target_rdm_train_t, target_rdm_test_t, feature_specific_rdm_train, feature_specific_rdm_test, metric):
    """Worker handling a single timepoint calculation when EEG is the target."""
    model = feature_reweighting_model(feature_specific_rdm_train, target_rdm_train_t)
    feature_reweighted_rdm = model.predict(feature_specific_rdm_test)
    
    clipped_rdm = clip_rdm_values(feature_reweighted_rdm, metric)
    corr = spearmanr(clipped_rdm, target_rdm_test_t).correlation
    return corr

#=============================================================================
# Loading the EEG Responses and preparing the EEG RDMs
#=============================================================================
data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
eeg_train = np.load(os.path.join(data_path, f'eeg_train_sub-{args.subject:02d}_trial_avg-all.npy'), allow_pickle=True).item()['eeg_train']
eeg_test = np.load(os.path.join(data_path, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True).item()['eeg_test']
eeg_test = np.mean(eeg_test, axis=1)

# Downsampling the training dataset matrix
n_train_stim = eeg_train.shape[0]
train_idx = np.random.choice(n_train_stim, size=1500, replace=False)
eeg_train = eeg_train[train_idx, :, :]
fmri_train = fmri_train[train_idx, :]

print('Shape of the EEG data (train, test):', eeg_train.shape, eeg_test.shape)
print('Shape of the fMRI data (train, test):', fmri_train.shape, fmri_test.shape)

#=============================================================================
# Parallel Feature Reweighting Engine
#=============================================================================
n_timepoints = eeg_train.shape[2]

if args.target == 'fmri':
    print("\n>>> Computing Fixed Global Target fMRI RDMs <<<")
    target_rdm_train = flatten_rdm(pairwise_distances(fmri_train, metric=args.target_rdm_metric))
    target_rdm_test = flatten_rdm(pairwise_distances(fmri_test, metric=args.target_rdm_metric))
    
    # Flush fMRI variables to reclaim memory before parallel dispatch
    del fmri_train, fmri_test 

    print(f"\n>>> Dispatching {n_timepoints} Timepoints to Joblib Parallel Pool (Target: fMRI) <<<")
    # n_jobs=-1 automatically scales up to the CPU allocation specified in your Slurm configuration
    parallel_outputs = Parallel(n_jobs=4, verbose=10)(
    delayed(process_time_point_fmri_target)(
        eeg_train[:, :, t], 
        eeg_test[:, :, t], 
        target_rdm_train, 
        target_rdm_test, 
        args.target_rdm_metric
    ) for t in range(n_timepoints)
)
    
elif args.target == 'eeg': 
    print("\n>>> Pre-computing EEG Target RDMs across all Timepoints <<<")
    target_rdms_train = [flatten_rdm(pairwise_distances(eeg_train[:, :, t], metric=args.target_rdm_metric)) for t in range(n_timepoints)]
    target_rdms_test = [flatten_rdm(pairwise_distances(eeg_test[:, :, t], metric=args.target_rdm_metric)) for t in range(n_timepoints)]
    
    # Flush EEG matrices out of main RAM before parallel split
    del eeg_train, eeg_test
    
    print("\n>>> Extracting Predictor fMRI RDMs <<<")
    feature_specific_rdm_train = get_single_feature_rdms(fmri_train)
    feature_specific_rdm_test = get_single_feature_rdms(fmri_test)

    print(f"\n>>> Dispatching {n_timepoints} Timepoints to Joblib Parallel Pool (Target: EEG) <<<")
    correlations = Parallel(n_jobs=4, verbose=10)(
        delayed(process_time_point_eeg_target)(
            target_rdms_train[t], 
            target_rdms_test[t], 
            feature_specific_rdm_train, 
            feature_specific_rdm_test, 
            args.target_rdm_metric
        ) for t in range(n_timepoints)
    )

# --- 1. Extract the Correlation Time Course Array ---
correlations = np.array([res['corr'] for res in parallel_outputs], dtype=np.float32)

# --- 2. Aggregate all Model Weights into One Consolidated Dictionary ---
regression_weights = {
    'coef_': np.array([res['coef'] for res in parallel_outputs], dtype=np.float32),
    'intercept_': np.array([res['intercept'] for res in parallel_outputs], dtype=np.float32)
}

print(f"Aggregated coefficients matrix shape: {regression_weights['coef_'].shape}")
print(f"Aggregated intercepts array shape: {regression_weights['intercept_'].shape}")

#=============================================================================
# Saving the separate outputs
#=============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/feature_reweighted_rsa/target-{args.target}/target_rdm_metric-{args.target_rdm_metric}/subject-{args.subject}'
os.makedirs(save_dir, exist_ok=True)

# Save the primary correlations time course array
np.save(os.path.join(save_dir, f'{args.roi}.npy'), correlations)

# Save the weights dictionary object
np.save(os.path.join(save_dir, f'{args.roi}_weights.npy'), regression_weights)
print("Correlation time courses and weights dictionary saved.")

# Execution Timing
execution_time = time.time() - start_time
print(f"Execution complete! Total Wall Time: {execution_time:.2f} seconds.")