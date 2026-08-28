import numpy as np
import os
import random
import argparse
from sklearn.linear_model import LinearRegression, RidgeCV
import time
from tqdm import tqdm
from berg import BERG
from utils import load_fmri_hemi_data
import h5py
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

print(f'>>> Joint EEG-Features Encoding Fusion Phase 2 (ROI) <<<')
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

# ==================================
# fMRI Noise ceilings
# ==================================
berg = BERG(berg_dir='/scratch/giffordale95/projects/brain-encoding-response-generator')
metadata = berg.get_model_metadata('fmri-nsd_fsaverage-huze', subject=args.subject)
wb_noise_ceilings = metadata['fmri'][f'{args.hemisphere}_ncsnr']
wb_noise_ceilings = wb_noise_ceilings[7802*(args.fmri_split - 1):7802*args.fmri_split]

selected_vertices = np.where(wb_noise_ceilings >= 0.2)[0]
fmri_test = fmri_test[:, selected_vertices]
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
features_train = np.load(os.path.join(features_dir, f"sub-{args.subject:02d}_layerwise_fmaps.npy"), allow_pickle=True).item()[args.layer]['train']
features_test = np.load(os.path.join(features_dir, f"sub-{args.subject:02d}_layerwise_fmaps.npy"), allow_pickle=True).item()[args.layer]['test']


print("Shape of the features data (train, test):", features_train.shape, features_test.shape)

#=========================================================================
# Settings for saving the correlation coefficients
#========================================================================= 
correlations_save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/jefe_phase_2/roi/layerwise_alexnet/layer-{args.layer}/subject-{args.subject}/hemi-{args.hemisphere}'
if os.path.isdir(correlations_save_dir) == False:
    os.makedirs(correlations_save_dir)

file_name = f'fmri_split-{args.fmri_split}.npy'

#=========================================================================
# Fitting linear models that predict the responses of a group of vertices
#  from visual features at each time point
#========================================================================
correlations = np.zeros((308, 7802), dtype=np.float32) # 308 post-stimulus time points and  7802 vertices
corrs = [] # 
alphas = np.logspace(-6, 3, 20) # List of alphas for Ridge regression
print("Starting Joint EEG-Feature Encoding Fusion...")
for t in tqdm(range(51,eeg_train.shape[2])): # We focus the analysis on post-stimulus time points (51-359)
    # Predicting the fMRI responses from the EEG data using the pre-trained weights from phase 1
    # Matrix multiplication is faster than using the predict function of the linear regression model
    t_fmri = eeg_train[:, :, t] @ phase_1_weights['coef_'][t].T + phase_1_weights['intercept_'][t]
    t_fmri = t_fmri[:, selected_vertices].astype(np.float32) # Selecting vertices above noise ceiling threshold of 20% for computational efficiency

    # Fitting a new linear regression model using the trained predicted t-fMRI as target
    #encoding_model = RidgeCV(alphas=alphas, store_cv_results=True)
    encoding_model = RidgeCV(alphas=alphas, alpha_per_target=True)
    encoding_model.fit(features_train, t_fmri)

    # Evaluating the encoding model and saving the correlation coefficients
    pred_fmri = encoding_model.predict(features_test)
    corrs.append([np.corrcoef(pred_fmri[:,i], fmri_test[:,i], dtype=np.float32)[0,1] for i in range(fmri_test.shape[1])])
print(" Joint EEG-Feature Encoding Fusion complete!")
corrs = np.array(corrs, dtype=np.float32)
correlations[:, selected_vertices] = corrs
np.save(os.path.join(correlations_save_dir, file_name), correlations) # Saving the correlations after each time point to avoid data loss in case of interruption



# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")