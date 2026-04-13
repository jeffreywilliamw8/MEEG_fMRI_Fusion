import numpy as np
import os
import random
import argparse
from sklearn.linear_model import LinearRegression, RidgeCV
import time
import tqdm
from utils import load_fmri_data

# Start time
start_time = time.time()

# Random seed for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

#=======================================================
# Input arguments
#========================================================

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=str, default='01')
parser.add_argument('--hemisphere', type=str, default='left')
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--dnn_type', type=str, default='vdnn', choices=['vdnn', 'llm'],
                    help='Type of DNN features to use for the joint encoding fusion: "vdnn" for vision DNN features, "llm" for language model features.')
parser.add_argument('--eeg_frequency', type=int, default=100)
parser.add_argument('--eeg_channel_policy', type=str, default='append', 
                    help='Policy for handling EEG channels: "append" to use all channels, "average" to average across subjects',
                    choices=['append', 'average'])
args = parser.parse_args()

print(f'>>> Joint EEG-Features Encoding Fusion Phase 2 (Whole Brain) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#=======================================================
# Loading the EEG data (odd repeats for phase 2)
#=======================================================
eeg_path = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/prepared_eeg/{args.eeg_frequency}_Hz'
if args.eeg_channel_policy == 'append':
    eeg_train = np.load(os.path.join(eeg_path, 'eeg_train_zsc_odd_channel_policy-append.npy')) # Shape: (1000, 762, 185) or (1000, 762, 370)
    eeg_test = np.load(os.path.join(eeg_path, 'eeg_test_zsc_channel_policy-append.npy')) # Shape: (102, 762, 185) or (102, 762, 370)

elif args.eeg_channel_policy == 'average':
    eeg_train = np.load(os.path.join(eeg_path, 'eeg_train_zsc_odd_channel_policy-average.npy')) # Shape: (1000, 127, 185) or (1000, 127, 370)
    eeg_test = np.load(os.path.join(eeg_path, 'eeg_test_zsc_channel_policy-average.npy')) # Shape: (102, 127, 185) or (102, 127, 370)

print('Shape of the EEG data (train, test):', eeg_train.shape, eeg_test.shape)
#=========================
# Loading the fMRI data
#=========================
fmri_train, fmri_test = load_fmri_data(args.fmri_subject, args.hemisphere, roi='WB')

# For computational efficiency and parallelization, we fit the model on subsets of the whole-brain vertices.
# Each subset contains 7802 vertices, and we iterate through 21 subsets to cover all 163842 vertices of each hemisphere.
fmri_train = fmri_train[:, 7802*(args.fmri_split - 1):7802*args.fmri_split]
fmri_test = fmri_test[:, 7802*(args.fmri_split - 1):7802*args.fmri_split]
print("Shape of the fMRI data (train, test):", fmri_train.shape, fmri_test.shape)

#=======================================================================
# Loading the pre-trained EEG-to-fMRI encoder's weights (from phase 1)
#=======================================================================
phase_1_weights_path = f'/scratch/jeffreykatab/Code/Encoding_Models/regression_weights/jefe_phase_1_wb/{args.eeg_frequency}_Hz/fmri_sub-{args.fmri_subject}/{args.hemisphere}_hemisphere'
phase_1_weights = np.load(os.path.join(phase_1_weights_path, f'fmri_split-{args.fmri_split}.npy'), allow_pickle=True).item()
print("Loaded pre-trained EEG-to-fMRI encoder's weights")

#=======================================================================
# Loading the features data
#=======================================================================
if args.dnn_type == 'vdnn':
    features_dir = '/scratch/jeffreykatab/Code/Encoding_Models/stimulus_features/pca_model_features/modality-visual'
    features_train = np.load(os.path.join(features_dir, 'pca_stimulus_features_train_model-s3d.npy'), allow_pickle=True)
    features_test = np.load(os.path.join(features_dir, 'pca_stimulus_features_test_model-s3d.npy'), allow_pickle=True)
elif args.dnn_type == 'llm':
    features_dir = '/scratch/jeffreykatab/Code/Encoding_Models/semantic_models/embeddings/gte-Qwen1.5-7B-instruct'
    features_train = np.load(os.path.join(features_dir, 'gte-Qwen1.5-7B-instruct_1000PCs_train.npy'), allow_pickle=True)
    features_test = np.load(os.path.join(features_dir, 'gte-Qwen1.5-7B-instruct_1000PCs_test.npy'), allow_pickle=True)

print("Shape of the features data (train, test):", features_train.shape, features_test.shape)

#===========================================================================
# Joint EEG-Feature Encoding Fusion: Predict fMRI from EEG (t-fMRI), 
# and then train model to predict t-fMRI from the features
# Testing is done using the test fMRI responses
#===========================================================================
corrs = []
alphas = np.logspace(-1, 1, 20)
print("Starting Joint EEG-Feature Encoding Fusion...")

for t in tqdm.tqdm(range(eeg_train.shape[2])):
    # Loading the pre-trained EEG-to-fMRI encoder's weights for the current time point
    eeg2fmri = LinearRegression()
    # Loading the linear regression weights
    eeg2fmri.coef_ = phase_1_weights['coef_'][t]
    eeg2fmri.intercept_ = phase_1_weights['intercept_'][t]
    t_fmri = eeg2fmri.predict(eeg_train[:,:,t])

    # Fitting a new linear regression model using the trained predicted t-fMRI as target
    encoding_model = RidgeCV(alphas=alphas, store_cv_results=True)
    encoding_model.fit(features_train, t_fmri)

    # Evaluating the encoding model and saving the correlation coefficients
    pred_fmri = encoding_model.predict(features_test)
    corrs.append([np.corrcoef(pred_fmri[:,i], fmri_test[:,i], dtype=np.float32)[0,1] for i in range(fmri_test.shape[1])]) # Appending a list of 7802 correlation coefficients at each time point
print("Joint EEG-Feature Encoding Fusion complete!")

corrs = np.array(corrs, dtype=np.float32) # Shape: (n_timepoints, n_vertices)

#=======================================================================
# Saving the correlation coefficients
#=======================================================================  
corrs_save_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/jefe_wb/vision_language_models/dnn_type-{args.dnn_type}/{args.eeg_frequency}_Hz/fmri_sub-{args.fmri_subject}/{args.hemisphere}_hemisphere'
if os.path.isdir(corrs_save_dir) == False:
	os.makedirs(corrs_save_dir)

file_name = f'fmri_split-{args.fmri_split}.npy'
np.save(os.path.join(corrs_save_dir, file_name), corrs)

# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")