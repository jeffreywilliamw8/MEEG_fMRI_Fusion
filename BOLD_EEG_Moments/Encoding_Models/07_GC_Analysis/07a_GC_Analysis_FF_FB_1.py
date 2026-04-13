import argparse
import os
import numpy as np
from tqdm import tqdm
import random
from sklearn.linear_model import RidgeCV, LinearRegression
import time
from sklearn.metrics import pairwise_distances




parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=str, default='01')
parser.add_argument('--rois', default='V1-V2', type=str, help="Select the pair of ROIs for GC analysis: 'V1-V4' or 'V4-PPA'")
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

# Start time
start_time = time.time()

# ===================================================================================================================================
# Load the pre-computed t-fMRI responses for the test set (for the selected ROIs) generated from the MEG->fMRI encoding fusion model
# ===================================================================================================================================
t_fmri_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/t_fmri/eeg2fmri_roi/{args.eeg_frequency}_Hz/fmri_sub-{args.fmri_subject}'

def get_roi_data(roi_name):
    """
    Utility function to load data for all the component ROIs
    for a given ROI umbrella name. 
    E.g, if roi_name is 'V1', it will load and return data for 'V1v' and 'V1d'

    input: roi_name (str): The umbrella name of the ROI (e.g., 'V1')
    output: dict with keys 'left' and 'right', each containing a list of numpy arrays for the corresponding component ROIs
    """
    roi_map  = {
        'V1': ['V1v', 'V1d'],
        'V2': ['V2v', 'V2d'],
        'V3': ['V3v', 'V3d', 'V3ab'],
        'V4': ['hV4'],
    }
    if roi_name in roi_map: # For ROIs with multiple components (ROIs whose names are in roi_map), load all of them and return as a list
        output_dict = {
            'left': [np.load(os.path.join(t_fmri_dir, f't_fmri_{roi}_left.npy')) for roi in roi_map.get(roi_name, [])],
            'right': [np.load(os.path.join(t_fmri_dir, f't_fmri_{roi}_right.npy')) for roi in roi_map.get(roi_name, [])]
        }
    else: # For ROIs without multiple components (ROIs whose names are not in roi_map), load the single corresponding file and return as a list with one element
        output_dict = {
            'left': [np.load(os.path.join(t_fmri_dir, f't_fmri_{roi_name}_left.npy'))],
            'right': [np.load(os.path.join(t_fmri_dir, f't_fmri_{roi_name}_right.npy'))]
        }
    return output_dict

def flatten_rdm(rdm):
    return rdm[np.triu_indices_from(rdm, k=1)]  # k=1 excludes diagonal

# Computing RDMs for the selected ROIs
rdms_roi_1_left = [[flatten_rdm(pairwise_distances(roi_data[t,:], metric='correlation')) for t in range(roi_data.shape[0])] for roi_data in get_roi_data(args.rois.split('-')[0])['left']]
rdms_roi_1_right = [[flatten_rdm(pairwise_distances(roi_data[t,:], metric='correlation')) for t in range(roi_data.shape[0])] for roi_data in get_roi_data(args.rois.split('-')[0])['right']]
rdms_roi_2_left = [[flatten_rdm(pairwise_distances(roi_data[t,:], metric='correlation')) for t in range(roi_data.shape[0])] for roi_data in get_roi_data(args.rois.split('-')[1])['left']]
rdms_roi_2_right = [[flatten_rdm(pairwise_distances(roi_data[t,:], metric='correlation')) for t in range(roi_data.shape[0])] for roi_data in get_roi_data(args.rois.split('-')[1])['right']]
# Averaging RDMs across component ROIs (if there are multiple) for each hemisphere
rdms_roi_1_left_avg = np.mean(rdms_roi_1_left, axis=0) if len(rdms_roi_1_left) > 1 else rdms_roi_1_left[0]
rdms_roi_1_right_avg = np.mean(rdms_roi_1_right, axis=0) if len(rdms_roi_1_right) > 1 else rdms_roi_1_right[0]
rdms_roi_2_left_avg = np.mean(rdms_roi_2_left, axis=0) if len(rdms_roi_2_left) > 1 else rdms_roi_2_left[0]
rdms_roi_2_right_avg = np.mean(rdms_roi_2_right, axis=0) if len(rdms_roi_2_right) > 1 else rdms_roi_2_right[0]
# Averaging RDMs across hemispheres to get a single RDM per ROI
rdms_roi_1 = np.mean([rdms_roi_1_left_avg, rdms_roi_1_right_avg], axis=0, dtype=np.float32)
rdms_roi_2 = np.mean([rdms_roi_2_left_avg, rdms_roi_2_right_avg], axis=0, dtype=np.float32)
print(f"Shape of the RDMs for ROI 1 ({args.rois.split('-')[0]}): {rdms_roi_1.shape} (Time points x Stimulus pairs)")
print(f"Shape of the RDMs for ROI 2 ({args.rois.split('-')[1]}): {rdms_roi_2.shape} (Time points x Stimulus pairs)")

# ===================================================================================================================================
# Performing Granger Causality Analysis: Observations = Image Pairs
# ===================================================================================================================================

# --- Configuration ---
# 10ms resolution (eeg frequency = 100Hz): 100ms window = 10 indices | 20ms gap = 2 indices
n_timepoints = rdms_roi_1.shape[0]
eeg_frequency = int(n_timepoints/3.7) # 370 time points -> 100 Hz; 185 time points -> 50 Hz
window_width_ms = 100 # Width of the past window in milliseconds
window_width_n_timepoints = int((window_width_ms / 1000) * eeg_frequency) # Convert window width from ms to number of time points
                                                                          # At the default frequency of 100 Hz and for a window width of 100 ms,
                                                                          # this will be 10 time points (since 100 ms corresponds to 10 time points at 100 Hz).

gap_ms = 20 # Gap between the end of the past window and the current time point in milliseconds
gap_n_timepoints = int((gap_ms / 1000) * eeg_frequency) # Convert gap from ms to number of time points  



def calculate_gc_step(target_rdms, source_rdms, t):
    """
    Computes GC at time t by comparing the predictive power of the source ROI's past on the target ROI's present, beyond what the target ROI's own past can predict.
    """
    # Define the 'Past' window indices: e.g., if t=20, past_idx is [6, 7, ..., 15]
    # This covers 100ms (10 indices) and ends exactly 20ms (2 indices) before t
    past_idx = np.arange(t - (window_width_n_timepoints + gap_n_timepoints), t - gap_n_timepoints)

    # Target: The RDM vector at current time t (Shape: 5151,)
    y = target_rdms[t]
    
    # Predictors: Each row is a pairwise distance, each column is a time-lag
    # .T transforms (10, 5151) -> (5151, 10)
    X_target_past = target_rdms[past_idx].T 
    X_source_past = source_rdms[past_idx].T
    
    # --- Model 1: Reduced (Self-Prediction) ---
    # Does B's own past predict B's present?
    reduced_model = LinearRegression().fit(X_target_past, y)
    rss_red = np.sum((y - reduced_model.predict(X_target_past))**2)
    r2_red = reduced_model.score(X_target_past, y)
    
    # --- Model 2: Full (Self + Source Prediction) ---
    # Does (B's past + A's past) predict B's present?
    X_full = np.hstack([X_target_past, X_source_past]) # Shape: (5151, 20)
    full_model = LinearRegression().fit(X_full, y)
    rss_full = np.sum((y - full_model.predict(X_full))**2)
    r2_full = full_model.score(X_full, y)

    # Granger Influence: ln(RSS_reduced / RSS_full)
    # If Source adds info, RSS_full < RSS_red -> GC > 0
    #return rss_red - rss_full # using the difference in RSS instead of the ratio to avoid issues with very small RSS values and to maintain interpretability in terms of variance explained. A positive value indicates that the source ROI adds predictive power for the target ROI's present RDM, suggesting a Granger causal influence.
    #return np.log(rss_red / (rss_full+1e-10)) # Adding small constant to avoid division by zero
    return r2_full - r2_red # Using the difference in R^2 values to quantify the increase in variance explained by adding the source ROI's past. A positive value indicates that the source ROI adds predictive power for the target ROI's present RDM, suggesting a Granger causal influence.

# --- Execution ---
# Baseline: -80ms to 0ms (Indices 12 to 19)
baseline_1to2 = [calculate_gc_step(rdms_roi_2, rdms_roi_1, t) for t in range(12, 20)]
baseline_2to1 = [calculate_gc_step(rdms_roi_1, rdms_roi_2, t) for t in range(12, 20)]

# Main Analysis: 0ms to 3200ms (Indices 20 to 340)
gc_1to2 = []
gc_2to1 = []

print("Starting Post-stimulus Granger Causality Analysis...")
for t in tqdm(range(20, 341), desc="GC Analysis"):
    gc_1to2.append(calculate_gc_step(rdms_roi_2, rdms_roi_1, t))
    gc_2to1.append(calculate_gc_step(rdms_roi_1, rdms_roi_2, t))

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
save_dir = f'/home/jeffreykatab/Projects/fusion/Bold_EEG_Moments/Encoding_Models/results/granger_causality_analysis/phase_1/fmri_sub-{args.fmri_subject}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_file = os.path.join(save_dir, f'{args.rois}.npy')
np.save(save_file, results)
print(f"\n[✅] Granger Causality results saved to: {save_file}")

# End time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds.")