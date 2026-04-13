import argparse
import os
import numpy as np
from tqdm import tqdm
import random
from sklearn.linear_model import RidgeCV, LinearRegression
import time
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler




# Start time
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=str, default='01')
parser.add_argument('--rois', default='V1-V4', type=str, help="Select the pair of ROIs for GC analysis")
parser.add_argument('--cross_validate', default="True", type=str, help="Determines whether the RDM regression will use 2 independent data splits for cross-validation or not")
parser.add_argument('--eeg_frequency', type=int, default=100)
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


#=================================================
# Loading the t-fMRI data
#=================================================
t_fmri_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/t_fmri/eeg2fmri_roi/{args.eeg_frequency}_Hz/eeg_channel_policy-append/fmri_sub-{args.fmri_subject}'
def get_concatenated_roi_data(roi_name):
    """
    Loads all sub-regions and hemispheres for a given ROI and concatenates them 
    along the vertex dimension (axis 2).
    
    input: roi_name (str)
    output: numpy array of shape (Time, Stimuli, Total_Vertices)
    """
    roi_map = {
        'V1': ['V1v', 'V1d'],
        'V2': ['V2v', 'V2d'],
        'V3': ['V3v', 'V3d', 'V3ab'],
        'V4': ['hV4'],
    }
    
    # Determine which sub-ROI names to load
    target_rois = roi_map[roi_name] if roi_name in list(roi_map.keys()) else [roi_name]
    all_parts_train = []
    all_parts_test = []
    if args.cross_validate=="True":
        for roi in target_rois:
            # Load Left and Right hemisphere files
            l_path_train = os.path.join(t_fmri_dir, f'{roi}_left_even.npy')
            l_path_test = os.path.join(t_fmri_dir, f'{roi}_left_odd.npy')

            r_path_train = os.path.join(t_fmri_dir, f'{roi}_right_even.npy')
            r_path_test = os.path.join(t_fmri_dir, f'{roi}_right_odd.npy')
            
            all_parts_train.append(np.load(l_path_train))
            all_parts_train.append(np.load(r_path_train))

            all_parts_test.append(np.load(l_path_test))
            all_parts_test.append(np.load(r_path_test))

    else: # If cross_validate = False, the training and testing data are identical
        for roi in target_rois:
            # Load Left and Right hemisphere files
            l_path_train = os.path.join(t_fmri_dir, f'{roi}_left.npy')
            l_path_test = os.path.join(t_fmri_dir, f'{roi}_left.npy')

            r_path_train = os.path.join(t_fmri_dir, f'{roi}_right.npy')
            r_path_test = os.path.join(t_fmri_dir, f'{roi}_right.npy')
            
            all_parts_train.append(np.load(l_path_train))
            all_parts_train.append(np.load(r_path_train))

            all_parts_test.append(np.load(l_path_test))
            all_parts_test.append(np.load(r_path_test))

    # Concatenate along the last axis (vertices). 
    # Shapes are (Time, Stimuli, Vertices) -> Result: (Time, Stimuli, Sum_of_Vertices)
    train_data = np.concatenate(all_parts_train, axis=-1)
    test_data = np.concatenate(all_parts_test, axis=-1)

    # Z-scoring the t-fMRI data:
    for t in range(train_data.shape[0]):
        scaler = StandardScaler()
        train_data[t] = scaler.fit_transform(train_data[t])
        test_data[t] = scaler.transform(test_data[t])


    return train_data, test_data

# Load and Concatenate t-fMRI data
roi_1_name = args.rois.split('-')[0]
roi_2_name = args.rois.split('-')[1]

print("Loading t-fMRI data...")
t_fmri_roi_1_train, t_fmri_roi_1_test= get_concatenated_roi_data(roi_1_name)
t_fmri_roi_2_train, t_fmri_roi_2_test= get_concatenated_roi_data(roi_2_name)

def flatten_rdm(rdm):
    return rdm[np.triu_indices_from(rdm, k=1)]  # k=1 excludes diagonal

print(f"Concatenated vertex dimension for {roi_1_name} (train): {t_fmri_roi_1_train.shape}")
print(f"Concatenated vertex dimension for {roi_1_name} (test): {t_fmri_roi_1_test.shape}")
print(f"Concatenated vertex dimension for {roi_2_name} (train): {t_fmri_roi_2_train.shape}")
print(f"Concatenated vertex dimension for {roi_2_name} (test): {t_fmri_roi_2_test.shape}")

#=================================================
# Computing the t-fMRI RDMs
#=================================================

# We iterate over time (t) and compute the RDM for the stimulus matrix (Stimuli x Vertices)
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
# 10ms resolution (eeg frequency = 100Hz): 100ms window = 10 indices | 20ms gap = 2 indices
n_timepoints = rdms_roi_1_train.shape[0]
eeg_frequency = int(n_timepoints/3.7) # 370 time points -> 100 Hz; 185 time points -> 50 Hz
window_width_ms = 100 # Width of the past window in milliseconds
window_width_n_timepoints = int((window_width_ms / 1000) * eeg_frequency) # Convert window width from ms to number of time points
                                                                          # At the default frequency of 100 Hz and for a window width of 100 ms,
                                                                          # this will be 10 time points (since 100 ms corresponds to 10 time points at 100 Hz).

gap_ms = 20 # Gap between the end of the past window and the current time point in milliseconds
gap_n_timepoints = int((gap_ms / 1000) * eeg_frequency) # Convert gap from ms to number of time points  



def calculate_gc_step(target_rdms_train, target_rdms_test, source_rdms_train, source_rdms_test, t, average_time_points=True):
    """
    Computes GC at time t by comparing the predictive power of the source ROI's past on the target ROI's present, beyond what the target ROI's own past can predict.
    """
    # Define the 'Past' window indices: e.g., if t=20, past_idx is [6, 7, ..., 15]
    # This covers 100ms (10 indices) and ends exactly 20ms (2 indices) before t
    past_idx = np.arange(t - (window_width_n_timepoints + gap_n_timepoints), t - gap_n_timepoints)

    # Target: The RDM vector at current time t (Shape: 5151,)
    y_train = target_rdms_train[t]
    y_test = target_rdms_test[t]
    
    # Predictors: Each row is a pairwise distance, each column is a time-lag
    # .T transforms (10, 5151) -> (5151, 10)
    X_target_past_train = target_rdms_train[past_idx].T
    X_source_past_train = source_rdms_train[past_idx].T

    X_target_past_test = target_rdms_test[past_idx].T
    X_source_past_test = source_rdms_test[past_idx].T

    if average_time_points:
        # Averaging predictors across time points
        X_target_past_train = np.mean(X_target_past_train, axis=1).reshape(-1, 1) # Shape: (5151,1)
        X_source_past_train = np.mean(X_source_past_train, axis=1).reshape(-1, 1) # Shape: (5151,1)

        X_target_past_test = np.mean(X_target_past_test, axis=1).reshape(-1, 1) # Shape: (5151,1)
        X_source_past_test = np.mean(X_source_past_test, axis=1).reshape(-1, 1) # Shape: (5151,1)
    
    # --- Model 1: Reduced (Self-Prediction) ---
    reduced_model = LinearRegression().fit(X_target_past_train, y_train)
    u_reduced = np.mean((y_test - reduced_model.predict(X_target_past_test))**2)
    
    # --- Model 2: Full (Self + Source Prediction) ---
    X_full_train = np.hstack([X_target_past_train, X_source_past_train]) # Shape: (5151, 20) or (5151, 2)
    X_full_test = np.hstack([X_target_past_test, X_source_past_test])
    full_model = LinearRegression().fit(X_full_train, y_train)
    u_full = np.mean((y_test - full_model.predict(X_full_test))**2)

    
    
    # Adjusting the MSE scores for the number of predictors in
    # the models
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
    # Baseline: -80ms to 0ms (Indices 12 to 19)
    baseline_1to2 = [calculate_gc_step(rdms_roi_2_train, rdms_roi_2_test, rdms_roi_1_train, rdms_roi_1_test, t, average_time_points) for t in range(12, 20)]
    baseline_2to1 = [calculate_gc_step(rdms_roi_1_train, rdms_roi_1_test, rdms_roi_2_train, rdms_roi_2_test, t, average_time_points) for t in range(12, 20)]

    # Main Analysis: 0ms to 3200ms (Indices 20 to 340)
    gc_1to2 = []
    gc_2to1 = []

    print("Starting Post-stimulus Granger Causality Analysis...")
    for t in tqdm(range(20, 341), desc="GC Analysis"):
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
    save_dir = f'/home/jeffreykatab/Projects/fusion/Bold_EEG_Moments/Encoding_Models/results/granger_causality_analysis/phase_1/{folder_name}/fmri_sub-{args.fmri_subject}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, f'{args.rois}_{suffix}.npy')
    np.save(save_file, results)
    print(f"Granger Causality results saved to: {save_file}")

# End time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds.")