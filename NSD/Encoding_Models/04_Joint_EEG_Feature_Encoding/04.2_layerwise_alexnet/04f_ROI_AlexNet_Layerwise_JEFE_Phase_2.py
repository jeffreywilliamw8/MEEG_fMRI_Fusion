"""
This script implements the second phase of the Stimulus feature encoding fusion/Joint EEG-Feature Encoding Fusion (JEFE) for
the a specific Region of Interest (ROI), using features extracted from a specific layer of the AlexNet model. 
The second phase consists of using the pre-trained EEG-to-fMRI encoding model weights (from phase 1) to predict fMRI responses
from EEG data at each time point, and then training a new linear regression model to predict the t-fMRI responses from 
the stimulus features (DNN features) at each time point.

Parameters
----------
subject : int
    The number of the NSD participant pair for which the encoding fusion is performed.
hemisphere : str
    The hemisphere of the brain to analyze ('lh' for left hemisphere, 'rh' for right hemisphere).
roi : str
    The region of interest (ROI) to analyze (e.g., 'V1v', 'V1d', etc.).
cv_split : str
    The half of repeats to use for training the EEG-to-fMRI encoding model ('even' or 'odd'). The other half will be used for testing in phase 2 of the joint EEG-feature encoding fusion analysis.
dnn_type : str
    The type of DNN features to use for the joint encoding fusion: "vdnn" for vision DNN features, "llm" for language model features, or "both" for using both types of features (via concatenation).
n_jobs : int
    Number of parallel workers used across time points (-1 = all available cores).

"""


import numpy as np
import os
import random
import argparse
from sklearn.linear_model import RidgeCV
import time
import tqdm
from utils import load_fmri_roi_data
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
parser.add_argument('--roi', type=str, default='V1v')
parser.add_argument('--cv_split', type=str, default='odd') # Even/odd cross-validation split

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

print(f'>>> Joint EEG-Features Encoding Fusion Phase 2 (ROI) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#==============================================================
# Loading the training EEG responses (odd repeats for phase 2)
#==============================================================
data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
cv_dict = {
        'even': 'odd', # If 'even' was used for phase 1, 'odd' will be used for phase 2, and vice-versa
        'odd': 'even'
    }
eeg_train = np.load(os.path.join(data_path, f'eeg_train_sub-{args.subject:02d}_trial_avg-{args.cv_split}.npy'), allow_pickle=True).item()['eeg_train'] # Shape: (9000, 160, 359)

print('Shape of the EEG data (train):', eeg_train.shape)


# =============================================================================
# Load the fMRI responses (only the test set are necessary for phase 2)
# =============================================================================
_, fmri_test = load_fmri_roi_data(args.subject, args.hemisphere, args.roi, nc_threshold=0.2) # Shape: (9000, n_vertices), (515, n_vertices)
print('Shape of the fMRI data (test):', fmri_test.shape)
if fmri_test.shape[1]>0:
    fmri_test_z = (fmri_test - fmri_test.mean(0)) / (fmri_test.std(0) + 1e-8) # z-score the fMRI test responses for correlation computation

    #=======================================================================
    # Loading the pre-trained EEG-to-fMRI encoder's weights (from phase 1)
    #=======================================================================
    phase_1_weights_path = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/regression_weights/joint_eeg_feature_encoding/roi/phase_1/subject-{args.subject}'
    phase_1_weights = np.load(os.path.join(phase_1_weights_path, f'{args.roi}_{args.hemisphere}_cv_split-{cv_dict[args.cv_split]}.npy'), allow_pickle=True).item()
    print("Loaded pre-trained EEG-to-fMRI encoder's weights")

    #====================================================================
    # Loading the layer features data
    #====================================================================
    features_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/alexnet'
    features_train = np.load(os.path.join(features_dir, f"sub-{args.subject:02d}_layerwise_fmaps.npy"), allow_pickle=True).item()[args.layer]['train']
    features_test = np.load(os.path.join(features_dir, f"sub-{args.subject:02d}_layerwise_fmaps.npy"), allow_pickle=True).item()[args.layer]['test']


    print("Shape of the features data (train, test):", features_train.shape, features_test.shape)

    #=========================================================================
    # Settings for saving the correlation coefficients and regression weights
    # The weights will be used for variance partitioning
    #========================================================================= 
    correlations_save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/jefe_phase_2/roi/layerwise_alexnet/layer-{args.layer}/subject-{args.subject}'
    if os.path.isdir(correlations_save_dir) == False:
        os.makedirs(correlations_save_dir)

    file_name = f'{args.roi}_{args.hemisphere}_cv_split-{args.cv_split}.npy'

    #=========================================================================
    # Fitting linear models that predict the responses of a group of vertices
    #  from visual features at each time point
    #========================================================================
    correlations = [] # Load existing correlations if the file exists, otherwise start with an empty list
    alphas = np.logspace(-6, 3, 20) # List of alphas for Ridge regression
    print("Starting Joint EEG-Feature Encoding Fusion...")
    for t in tqdm.tqdm(range(len(correlations), eeg_train.shape[2])): # Start from the next time point if correlations already exist for some time points
        # Predicting the fMRI responses from the EEG data using the pre-trained weights from phase 1
        # Matrix multiplication is faster than using the predict function of the linear regression model
        t_fmri = eeg_train[:, :, t] @ phase_1_weights['coef_'][t].T + phase_1_weights['intercept_'][t]

        # Fitting a new linear regression model using the trained predicted t-fMRI as target
        #encoding_model = RidgeCV(alphas=alphas, store_cv_results=True)
        encoding_model = RidgeCV(alphas=alphas, alpha_per_target=True)
        encoding_model.fit(features_train, t_fmri)

        # Evaluating the encoding model and saving the correlation coefficients
        pred_t_fmri = features_test @ encoding_model.coef_.T + encoding_model.intercept_
        pred_t_fmri_z = (pred_t_fmri - pred_t_fmri.mean(0)) / (pred_t_fmri.std(0) + 1e-8)
        corrs = np.diag(pred_t_fmri_z.T @ fmri_test_z) / len(pred_t_fmri_z)
        correlations.append(corrs)
    print(" Joint EEG-Feature Encoding Fusion complete!")
    correlations = np.array(correlations, dtype=np.float32)
    np.save(os.path.join(correlations_save_dir, file_name), correlations)
    print(f"Correlations saved to: {os.path.join(correlations_save_dir, file_name)}, shape={correlations.shape}")

else:
     print("No vertices above noise ceiling threshold found in this ROI. Terminating...")


# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")