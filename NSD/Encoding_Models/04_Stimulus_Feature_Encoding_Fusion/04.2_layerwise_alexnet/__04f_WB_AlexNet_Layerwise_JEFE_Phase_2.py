import numpy as np
import os
import random
import argparse
from sklearn.linear_model import LinearRegression, RidgeCV
import time
from tqdm import tqdm
import h5py

# --- Set Thread Caps BEFORE importing joblib/sklearn to prevent core thrashing ---
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from joblib import Parallel, delayed
from utils import load_fmri_hemi_data

# Start time
start_time = time.time()

# Random seed for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

#================================================
# Input arguments
#================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--hemisphere', type=str, default='lh')
parser.add_argument('--fmri_split', type=int, default=1)
alexnet_layers = [
    'features.2',    # Conv1 + Pool
    'features.5',    # Conv2 + Pool
    'features.7',    # Conv3
    'features.9',    # Conv4
    'features.12',   # Conv5 + Pool
    'classifier.2',  # FC6
    'classifier.5',  # FC7
    'classifier.6'   # FC8 (Output)
]
parser.add_argument('--layer', type=str, default='features.2', choices=alexnet_layers,
                    help='Layer of the Alexnet model from which the features are extracted for the joint encoding fusion.')
args = parser.parse_args()

print(f'>>> Parallel Joint EEG-Features Encoding Fusion Phase 2 (ROI) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

#==============================================================
# Loading the training EEG responses (odd repeats for phase 2)
#==============================================================
data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
eeg_train = np.load(os.path.join(data_path, f'eeg_train_sub-{args.subject:02d}_trial_avg-odd.npy'), allow_pickle=True).item()['eeg_train'] # Shape: (9000, 160, 359)
print('Shape of the EEG data (train):', eeg_train.shape)

# =============================================================================
# Load the fMRI responses
# =============================================================================
_, fmri_test = load_fmri_hemi_data(args.subject, args.hemisphere)
fmri_test = fmri_test[:, 7802*(args.fmri_split - 1):7802*args.fmri_split]
print('Shape of the fMRI data (test):', fmri_test.shape)

#=======================================================================
# Loading the pre-trained EEG-to-fMRI encoder's weights (from phase 1)
#=======================================================================
phase_1_weights_path = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/joint_eeg_feature_encoding/wb/phase_1/subject-{args.subject}/hemi-{args.hemisphere}'
phase_1_weights = np.load(os.path.join(phase_1_weights_path, f'fmri_split-{args.fmri_split}_cv_split-even.npy'), allow_pickle=True).item()
print("Loaded pre-trained EEG-to-fMRI encoder's weights")

#====================================================================
# Loading the layer features data
#====================================================================
features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/alexnet'
features_data = np.load(os.path.join(features_dir, f"sub-{args.subject:02d}_layerwise_fmaps.npy"), allow_pickle=True).item()[args.layer]
features_train = features_data['train']
features_test = features_data['test']
print("Shape of the features data (train, test):", features_train.shape, features_test.shape)

#=========================================================================
# Settings for saving the correlation coefficients and regression weights
#========================================================================= 
correlations_save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/jefe_phase_2/roi/layerwise_alexnet/layer-{args.layer}/subject-{args.subject}/hemi-{args.hemisphere}'
os.makedirs(correlations_save_dir, exist_ok=True)
file_name = f'fmri_split-{args.fmri_split}.npy'

#=========================================================================
# Worker function for single time point processing
#=========================================================================
def process_single_timepoint(t, eeg_t_data, coef_t, intercept_t, features_train, features_test, fmri_test, alphas):
    """
    Isolates computation for timepoint `t` inside a thread-safe worker execution space.
    """
    # 1. Reconstruct predicted t_fmri using fast matrix mult
    t_fmri = eeg_t_data @ coef_t.T + intercept_t

    # 2. Fit RidgeCV across visual features mapping to projected fMRI space
    encoding_model = RidgeCV(alphas=alphas, alpha_per_target=True)
    encoding_model.fit(features_train, t_fmri)

    # 3. Evaluate model accuracy using Pearson correlation on independent test sets
    pred_fmri = encoding_model.predict(features_test)
    
    n_vertices = fmri_test.shape[1]
    t_correlations = np.zeros(n_vertices, dtype=np.float32)
    for i in range(n_vertices):
        t_correlations[i] = np.corrcoef(pred_fmri[:, i], fmri_test[:, i])[0, 1]
        
    return t, t_correlations

#=========================================================================
# Launching joblib pool
#=========================================================================
alphas = np.logspace(-6, 3, 20)
n_timepoints = eeg_train.shape[2]

print(f"Starting Parallel Joint EEG-Feature Encoding Fusion over {n_timepoints} timepoints...")

# Use backend="loky" for clean memory space serialization across workers
results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
    delayed(process_single_timepoint)(
        t, 
        eeg_train[:, :, t], 
        phase_1_weights['coef_'][t], 
        phase_1_weights['intercept_'][t], 
        features_train, 
        features_test, 
        fmri_test, 
        alphas
    )
    for t in range(n_timepoints)
)

print("\nProcessing complete! Organizing temporal order...")

# Unpack and sort results by timepoint index to ensure chronological alignment
correlations_array = np.zeros((n_timepoints, fmri_test.shape[1]), dtype=np.float32)
for t_idx, t_corrs in results:
    correlations_array[t_idx, :] = t_corrs

# Save complete matrix to file
save_path = os.path.join(correlations_save_dir, file_name)
np.save(save_path, correlations_array)
print(f"✅ Success! Joint EEG-Feature Encoding completed and saved to {save_path}")

# End time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution complete! Time: {execution_time:.2f} seconds.")