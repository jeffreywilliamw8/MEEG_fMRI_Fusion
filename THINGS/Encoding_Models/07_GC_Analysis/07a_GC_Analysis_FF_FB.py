import argparse
import os
import numpy as np
from tqdm import tqdm
import random
from sklearn.linear_model import LinearRegression
import time
from sklearn.metrics import pairwise_distances
from berg import BERG
import h5py
from sklearn.preprocessing import StandardScaler

# Start time
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)
parser.add_argument('--rois', default='EVC-V4', type=str, help="Select the pair of ROIs for GC analysis")
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
# Loading the t-fMRI data
# =============================================================================
t_fmri_dir = f'/home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/t_fmri/meg_channel_policy-append/fmri_sub-{args.fmri_subject:02d}'
def get_t_fmri(roi):
    if args.cross_validate=="True": # If cross_validate = True, we use the even data for training and the odd data for testing
        t_fmri_train = np.load(os.path.join(t_fmri_dir, f'{roi}_even.npy'))
        t_fmri_test = np.load(os.path.join(t_fmri_dir, f'{roi}_odd.npy'))
    else: # If cross_validate = False, the training and testing data are identical, i.e, t_fmri obtained from MEG averaged across all repeats
        t_fmri_train = np.load(os.path.join(t_fmri_dir, f'{roi}.npy'))
        t_fmri_test = np.load(os.path.join(t_fmri_dir, f'{roi}.npy'))
    
    return t_fmri_train, t_fmri_test
    

# ===================================================================================================================================
# Load and Concatenate t-fMRI responses for the selected ROIs (Voxel-level Concatenation)
# ===================================================================================================================================
def get_roi_data(roi_name, z_score=False):
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
# 5ms resolution: 50ms window = 10 indices | 20ms gap = 4 indices
n_timepoints = rdms_roi_1_train.shape[0]
window_width_n_timepoints = 10 # Number of time points in a 50ms window
gap_n_timepoints = 4 # Number of time points in a 20ms window 



def calculate_gc_step(target_rdms_train, target_rdms_test, source_rdms_train, source_rdms_test, t, average_time_points=True):
    """
    Computes GC at time t by comparing the predictive power of the source ROI's past on the target ROI's present, beyond what the target ROI's own past can predict.
    """
    # Define the 'Past' window indices: e.g., if t=20, past_idx is [6, 7, ..., 15]
    # This covers 50ms (10 indices) and ends exactly 20ms (4 indices) before t
    past_idx = np.arange(t - (window_width_n_timepoints + gap_n_timepoints), t - gap_n_timepoints)

    # Target: The RDM vector at current time t (Shape: 4950,)
    y_train = target_rdms_train[t]
    y_test = target_rdms_test[t]
    
    # Predictors: Each row is a pairwise distance, each column is a time-lag
    # .T transforms (10, 4950) -> (4950, 10)
    X_target_past_train = target_rdms_train[past_idx].T
    X_source_past_train = source_rdms_train[past_idx].T

    X_target_past_test = target_rdms_test[past_idx].T
    X_source_past_test = source_rdms_test[past_idx].T

    if average_time_points:
        # Averaging predictors across time points
        X_target_past_train = np.mean(X_target_past_train, axis=1).reshape(-1, 1) # Shape: (4950,1)
        X_source_past_train = np.mean(X_source_past_train, axis=1).reshape(-1, 1) # Shape: (4950,1)

        X_target_past_test = np.mean(X_target_past_test, axis=1).reshape(-1, 1) # Shape: (4950,1)
        X_source_past_test = np.mean(X_source_past_test, axis=1).reshape(-1, 1) # Shape: (4950,1)
    
    # --- Reduced Model (Self-Prediction) ---
    reduced_model = LinearRegression().fit(X_target_past_train, y_train)
    u_reduced = np.mean((y_test - reduced_model.predict(X_target_past_test))**2)
    
    # --- Full Model (Self + Source Prediction) ---
    X_full_train = np.hstack([X_target_past_train, X_source_past_train]) # Shape: (4950, 20) or (4950, 2)
    X_full_test = np.hstack([X_target_past_test, X_source_past_test])
    full_model = LinearRegression().fit(X_full_train, y_train)
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
for average_time_points, suffix in zip([True, False], ['ta', 'nta']): # we will store time-averaged (ta) and non-time-averaged (nta) results in separate files
    print("GC Analysis with predicting RDMs averaged across time: ", str(average_time_points))
    # Baseline: -30ms to 0ms (Indices 14 to 19)
    baseline_1to2 = [calculate_gc_step(rdms_roi_2_train, rdms_roi_2_test, rdms_roi_1_train, rdms_roi_1_test, t, average_time_points) for t in range(14, 20)]
    baseline_2to1 = [calculate_gc_step(rdms_roi_1_train, rdms_roi_1_test, rdms_roi_2_train, rdms_roi_2_test, t, average_time_points) for t in range(14, 20)]

    # Main Analysis: 0ms to 600ms (Indices 20 to 140)
    gc_1to2 = []
    gc_2to1 = []

    print("Starting Post-stimulus Granger Causality Analysis...")
    for t in tqdm(range(20, 141), desc="GC Analysis"):
        gc_1to2.append(calculate_gc_step(rdms_roi_2_train, rdms_roi_2_test, rdms_roi_1_train, rdms_roi_1_test, t, average_time_points))
        gc_2to1.append(calculate_gc_step(rdms_roi_1_train, rdms_roi_1_test, rdms_roi_2_train, rdms_roi_2_test, t, average_time_points))

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