import argparse
import os
import numpy as np
from tqdm import tqdm
import random
from sklearn.linear_model import LinearRegression, RidgeCV
import time
from sklearn.metrics import pairwise_distances
from berg import BERG
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Start time
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)
parser.add_argument('--rois', default='V1-V4', type=str, help="Select the pair of ROIs for GC analysis")
parser.add_argument('--cross_validate', default="True", type=str, help="Determines whether the RDM regression will use 2 independent data splits for cross-validation or not")
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--berg_dir', default='/scratch/jeffreykatab/Code/Encoding_Models/brain-encoding-response-generator', type=str)
args = parser.parse_args()
args, unknown = parser.parse_known_args()

print('>>> Granger Causality Analysis <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 8
random.seed(seed)
np.random.seed(seed)

# =============================================================================
# Loading the fMRI metadata
# =============================================================================
# Load the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata('fmri-things_fmri_1-vit_b_32', subject=args.fmri_subject)

# Get the image files names
test_stimuli_fmri = metadata_fmri['encoding_model']['test_stimuli']
unique_test_stimuli = np.unique(test_stimuli_fmri)


# =============================================================================
# Loading the MEG test data
# =============================================================================
# Loop across MEG subjects
for ms, msub in enumerate(tqdm(args.meg_subjects)):

    # Load the MEG metadata
    metadata_meg = berg.get_model_metadata(
        'meg-things_meg_1-vit_b_32',
        subject=msub
    )

    # Load the MEG responses
    meg_dir = os.path.join('/scratch/jeffreykatab/berg', 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_split-test.h5')
    meg_test_all = h5py.File(meg_dir, 'r')['neural_data']

    # Time point selection
    tmax = 0.8
    times = metadata_meg['meg']['times']
    time_idx = np.zeros(len(times), dtype=int)
    time_idx[times <= tmax] = 1
    time_idx = np.where(time_idx == 1)[0]
    times = times[times <= tmax]
    meg_test_all = meg_test_all[:,:,time_idx].astype(np.float32)

    # Average the MEG responses across repetitions for the images shared with
    # the fMRI
    test_stimuli_meg = metadata_meg['encoding_model']['test_stimuli']
    meg_test_sub = []
    for stim in unique_test_stimuli:
        idx = [i for i, x in enumerate(test_stimuli_meg) if x == stim]
        meg_test_sub.append(meg_test_all[idx].mean(0))
    meg_test_sub = np.array(meg_test_sub)

    # Append the MEG sensor responses across subjects
    if ms == 0:
        meg_data = meg_test_sub
    else:
        meg_data = np.append(meg_data, meg_test_sub, 1)
    del meg_test_all, meg_test_sub

print("Shape of the MEG data: ", meg_data.shape)
# ================================================================================
# Loading the t-fMRI data (computing t-fMRI data using weights from JMFE phase 1)
# ================================================================================
def get_t_fmri(roi):
    if args.cross_validate=="True": # If cross_validate = True, we use weights from Halves 1 and 2 of the MEG data for training and testing, respectively
        train_weights_dir = os.path.join('/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models', 'jmfe_phase_1', 'roi', 'half_1', f'fmri_sub-{args.fmri_subject:02d}')
        test_weights_dir = os.path.join('/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models', 'jmfe_phase_1', 'roi', 'half_2', f'fmri_sub-{args.fmri_subject:02d}')

    else: # If cross_validate = "False", the training and testing data are identical, i.e, t_fmri obtained from MEG Half 1 (or 2)
        train_weights_dir = os.path.join('/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models', 'jmfe_phase_1', 'roi', 'half_1', f'fmri_sub-{args.fmri_subject:02d}')
        test_weights_dir = os.path.join('/scratch/jeffreykatab/Projects/fusion/THINGS/Encoding_Models', 'jmfe_phase_1', 'roi', 'half_1', f'fmri_sub-{args.fmri_subject:02d}')

    train_weights = np.load(os.path.join(train_weights_dir, f'{roi}.npy'), allow_pickle=True).item() # Loading the trained models' weights
    test_weights = np.load(os.path.join(test_weights_dir, f'{roi}.npy'), allow_pickle=True).item() # Loading the trained models' weights
    train_coefs, train_intercepts = train_weights['coef_'], train_weights['intercept_'] # coefs: a list containing n_time_points matrices of shape (n_voxels, n_channels)
    test_coefs, test_intercepts = test_weights['coef_'], test_weights['intercept_'] # intercepts: a list containing n_time_points matrices of shape (n_voxels,)
    del train_weights, test_weights

    # Selecting only voxels above 20% noice ceiling threshold
    roi_idx = metadata_fmri['roi'][roi]
    whole_brain_nc = metadata_fmri['encoding_model']['noise_ceiling_testset']
    roi_noise_ceilings = whole_brain_nc[roi_idx]
    valid_voxels = np.where(roi_noise_ceilings > 20.0)[0] # Selecting voxels above noise ceiling threshold of 20%
    print(valid_voxels)
    t_fmri_train = np.zeros((meg_data.shape[2], meg_data.shape[0], len(valid_voxels)), dtype=np.float32) # shape: (time, stimuli, voxels)
    t_fmri_test = np.zeros((meg_data.shape[2], meg_data.shape[0], len(valid_voxels)), dtype=np.float32) # shape: (time, stimuli, voxels)
    for t in range(t_fmri_train.shape[0]): # using the model weights at each time point to predict t-fMRI from MEG
        t_fmri_train[t,:,:] = meg_data[:,:,t] @ train_coefs[t][valid_voxels, :].T + train_intercepts[t][valid_voxels] # coefs[t] and intercepts[t] have the respective shapes (n_voxels, n_channels) and (n_voxels,)
        t_fmri_test[t,:,:] = meg_data[:,:,t] @ test_coefs[t][valid_voxels, :].T + test_intercepts[t][valid_voxels]

    return t_fmri_train, t_fmri_test
    

# ===================================================================================================================================
# Load and Concatenate t-fMRI responses for the selected ROIs (Voxel-level Concatenation)
# ===================================================================================================================================
def get_roi_data(roi_name, z_score=True):
    """
    Loads training and testing t-fMRI data for a given ROI. If the ROI whose name is provided
    is composite, all its part are loaded and concatenated along the voxel dimension.
    The z_score argument determines whether the returned t-fMRI data will be z-scored
    
    input: roi_name (str), z_score (bool)
    output: training and testing numpy arrays of shape (Time, Stimuli, Total_Voxels)
    """
    if roi_name =='V4': # V4: ROI name given as argument to the function
        roi_name = 'hV4' # hV4: actual name of the V4 ROI as stored on the disk
    
    # Determine which sub-ROI names to load
    if roi_name == 'CSR': # C.S.R: Category-Selective Regions
        target_rois = ['lFFA', 'rFFA', 'lOFA', 'rOFA', 'lEBA', 'rEBA', 'lPPA', 'rPPA', 'lLOC', 'rLOC']
    elif roi_name == 'EVC':
        target_rois = ['V1', 'V2', 'V3']
    else:
        target_rois = [roi_name]
        
    all_parts_train = []
    all_parts_test = []
    for roi in target_rois:
        t_fmri_train, t_fmri_test = get_t_fmri(roi)
        all_parts_train.append(t_fmri_train)
        all_parts_test.append(t_fmri_test)

    # Concatenate along the last axis (voxels). 
    # Shapes are (Time, Stimuli, Voxels) -> Result: (Time, Stimuli, Sum_of_Voxels)
    if len(target_rois) == 1: # If the ROI is not composite (i.e has only 1 part), its t-fMRI data is returned as-is, without concatenation
        train_data = np.array(all_parts_train[0])
        test_data = np.array(all_parts_test[0])
    else: # If the ROI is composite (i.e made up of more than 1 sub-ROI)
        train_data = np.concatenate(all_parts_train, axis=-1)
        test_data = np.concatenate(all_parts_test, axis=-1)

    # Z-scoring the t-fMRI data:
    if z_score:
        for t in range(train_data.shape[0]):
            scaler = StandardScaler()
            train_data[t] = scaler.fit_transform(train_data[t])
            test_data[t] = scaler.transform(test_data[t])
    
    return train_data, test_data

# Mini-function to return only the flattened upper triangular part of an RDM
def flatten_rdm(rdm):
    return rdm[np.triu_indices_from(rdm, k=1)]

# Loading t-fMRI data
roi_1_name = args.rois.split('-')[0]
roi_2_name = args.rois.split('-')[1]

print("Loading t-fMRI data...")
t_fmri_roi_1_train, t_fmri_roi_1_test= get_roi_data(roi_1_name)
t_fmri_roi_2_train, t_fmri_roi_2_test= get_roi_data(roi_2_name)


print(f"Concatenated voxel dimension for {roi_1_name} (train): {t_fmri_roi_1_train.shape}")
print(f"Concatenated voxel dimension for {roi_1_name} (test): {t_fmri_roi_1_test.shape}")
print(f"Concatenated voxel dimension for {roi_2_name} (train): {t_fmri_roi_2_train.shape}")
print(f"Concatenated voxel dimension for {roi_2_name} (test): {t_fmri_roi_2_test.shape}")


# 2. Compute RDMs
# We iterate over time (t) and compute the RDM for the stimulus matrix (Stimuli x Voxels)
print("Computing RDMs...")
rdms_roi_1_train = np.array([
    flatten_rdm(pairwise_distances(t_fmri_roi_1_train[t, :, :], metric='correlation')) 
    for t in tqdm(range(t_fmri_roi_1_train.shape[0]))
], dtype=np.float32)

rdms_roi_1_test = np.array([
    flatten_rdm(pairwise_distances(t_fmri_roi_1_test[t, :, :], metric='correlation')) 
    for t in tqdm(range(t_fmri_roi_1_test.shape[0]))
], dtype=np.float32)

rdms_roi_2_train = np.array([
    flatten_rdm(pairwise_distances(t_fmri_roi_2_train[t, :, :], metric='correlation')) 
    for t in tqdm(range(t_fmri_roi_2_train.shape[0]))
], dtype=np.float32)

rdms_roi_2_test = np.array([
    flatten_rdm(pairwise_distances(t_fmri_roi_2_test[t, :, :], metric='correlation')) 
    for t in tqdm(range(t_fmri_roi_2_test.shape[0]))
], dtype=np.float32)



print(f"Shape of the RDMs for ROI 1 (train): {rdms_roi_1_train.shape} (Time points x Stimulus pairs)")
print(f"Shape of the RDMs for ROI 1 (test): {rdms_roi_1_test.shape} (Time points x Stimulus pairs)")
print(f"Shape of the RDMs for ROI 2 (train): {rdms_roi_2_train.shape} (Time points x Stimulus pairs)")
print(f"Shape of the RDMs for ROI 2 (test): {rdms_roi_2_test.shape} (Time points x Stimulus pairs)")

# ===================================================================================================================================
# Performing Granger Causality Analysis: Observations = Image Pairs
# ===================================================================================================================================

# --- Configuration ---
# 5ms resolution: 50ms window = 10 indices | 20ms offset = 4 indices
n_timepoints = rdms_roi_1_train.shape[0]
window_width_n_timepoints = 10 # Number of time points in a 50ms window
offset_n_timepoints = 4 # Number of time points in a 20ms window 
alphas = np.logspace(-6, 10, 17) # Penalization parameters for Ridge regression



def calculate_gc_step(target_rdms_train, target_rdms_test, source_rdms_train, source_rdms_test, t, time_aggregation=None):
    """
    Computes GC at time t by comparing the predictive power of the source ROI's past on the target ROI's present, beyond what the target ROI's own past can predict.
    """
    # Define the 'Past' window indices: e.g., if t=20, past_idx is [6, 7, ..., 15] (10 indices)
    # This covers 50ms (10 indices) and ends exactly 20ms (4 indices) before t
    past_idx = np.arange(t - (window_width_n_timepoints + offset_n_timepoints), t - offset_n_timepoints)

    # Target: The RDM vector at current time t (Shape: 4950,)
    y_train = target_rdms_train[t]
    y_test = target_rdms_test[t]
    
    # Predictors: Each row is a pairwise distance, each column is a time-lag
    # .T transforms (10, 4950) -> (4950, 10)
    X_target_past_train = target_rdms_train[past_idx].T
    X_source_past_train = source_rdms_train[past_idx].T

    X_target_past_test = target_rdms_test[past_idx].T
    X_source_past_test = source_rdms_test[past_idx].T

    if time_aggregation == "average":
        # Averaging predictors across time points
        X_target_past_train = np.mean(X_target_past_train, axis=1).reshape(-1, 1) # Shape: (4950,1)
        X_source_past_train = np.mean(X_source_past_train, axis=1).reshape(-1, 1) # Shape: (4950,1)

        X_target_past_test = np.mean(X_target_past_test, axis=1).reshape(-1, 1) # Shape: (4950,1)
        X_source_past_test = np.mean(X_source_past_test, axis=1).reshape(-1, 1) # Shape: (4950,1)

    elif time_aggregation == "pca":
        # PCA-based dimensionality reduction of predictors across time points
        n_components = 1 # Number of principal components to retain
        pca_target = PCA(n_components=n_components)
        pca_source = PCA(n_components=n_components)

        X_target_past_train = pca_target.fit_transform(X_target_past_train) # Shape: (4950, n_components)
        X_source_past_train = pca_source.fit_transform(X_source_past_train) # Shape: (4950, n_components)

        X_target_past_test = pca_target.transform(X_target_past_test) # Shape: (4950, n_components)
        X_source_past_test = pca_source.transform(X_source_past_test) # Shape: (4950, n_components)
    
    # --- Reduced Model (Self-Prediction) ---
    reduced_model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True).fit(X_target_past_train, y_train)
    u_reduced = np.mean((y_test - reduced_model.predict(X_target_past_test))**2)
    
    # --- Full Model (Self + Source Prediction) ---
    X_full_train = np.hstack([X_target_past_train, X_source_past_train]) # Shape: (4950, 20) or (4950, 2)
    X_full_test = np.hstack([X_target_past_test, X_source_past_test])
    full_model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True).fit(X_full_train, y_train)
    u_full = np.mean((y_test - full_model.predict(X_full_test))**2)

    
    # Adjusting the MSE scores for the number of predictors in the models
    n = len(target_rdms_test[t])
    p_reduced = X_target_past_train.shape[1]
    p_full = p_reduced + X_source_past_train.shape[1]
    u_reduced = u_reduced * (n - 1) / (n - p_reduced - 1)
    u_full = u_full * (n - 1) / (n - p_full - 1)

    # Granger Influence: ln(U_reduced / U_full)
    # If Source improves prediction, u_reduced > u_full => GC > 0
    return np.log(u_reduced/u_full)

# --- Execution ---
folder_name = "cv" if args.cross_validate=="True" else "ncv" # We will use 2 separate folders for cross-validated (cv) and not cross-validated (ncv) results
for time_aggregation, suffix in zip(["average", "pca", None], ['avg', 'pca', 'none']): # we will store time-averaged (ta) and non-time-averaged (nta) results in separate files
    print("GC Analysis with time_aggregation :", str(time_aggregation))
    # Baseline: -30ms to 0ms (Indices 14 to 19)
    print("Starting Pre-stimulus Granger Causality Analysis...")
    baseline_1to2 = [calculate_gc_step(rdms_roi_2_train, rdms_roi_2_test, rdms_roi_1_train, rdms_roi_1_test, t, time_aggregation) for t in range(14, 20)]
    baseline_2to1 = [calculate_gc_step(rdms_roi_1_train, rdms_roi_1_test, rdms_roi_2_train, rdms_roi_2_test, t, time_aggregation) for t in range(14, 20)]
    print("Pre-stimulus Granger Causality Analysis Complete!")

    # Main Analysis: 0ms to 600ms (Indices 20 to 140)
    gc_1to2 = []
    gc_2to1 = []

    print("Starting Post-stimulus Granger Causality Analysis...")
    for t in tqdm(range(20, 141), desc="GC Analysis"):
        gc_1to2.append(calculate_gc_step(rdms_roi_2_train, rdms_roi_2_test, rdms_roi_1_train, rdms_roi_1_test, t, time_aggregation))
        gc_2to1.append(calculate_gc_step(rdms_roi_1_train, rdms_roi_1_test, rdms_roi_2_train, rdms_roi_2_test, t, time_aggregation))
    print("Post-stimulus Granger Causality Analysis Complete!")

    # Result Aggregation
    results = {
        'gc_ff': np.array(gc_1to2),
        'gc_fb': np.array(gc_2to1),
        'baseline_ff': np.array(baseline_1to2),
        'baseline_fb': np.array(baseline_2to1),
    }

    # =============================================================================
    # Saving the results
    # =============================================================================
    save_dir = f'/home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/granger_causality_analysis/phase_1/{folder_name}/fmri_sub-{args.fmri_subject:02d}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, f'{args.rois}_{suffix}.npy')
    np.save(save_file, results)
    print(f"Granger Causality results saved to: {save_file}")


# End time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds.")