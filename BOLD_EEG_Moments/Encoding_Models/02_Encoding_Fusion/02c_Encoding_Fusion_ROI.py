import numpy as np
import os
import random
import argparse
from sklearn.linear_model import RidgeCV
import time
import tqdm
from utils import load_fmri_data

# Start time
start_time = time.time()

# Random seed for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

######################################################
# Input arguments
######################################################

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=str, default='01')
parser.add_argument('--hemisphere', type=str, default='left')
parser.add_argument('--roi', type=str, default='V1v')
parser.add_argument('--eeg_frequency', type=int, default=100)
parser.add_argument('--eeg_channel_policy', type=str, default='append', 
                    help='Policy for handling EEG channels: "append" to use all channels, "average" to average across subjects',
                    choices=['append', 'average'])
args = parser.parse_args()

print(f'>>> EEG-fMRI Encoding Fusion (ROI) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#################################################
# Loading the EEG data
#################################################
eeg_path = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/prepared_eeg/{args.eeg_frequency}_Hz'
if args.eeg_channel_policy == 'append':
    eeg_train = np.load(os.path.join(eeg_path, 'concat_eeg_train_762.npy')) # Shape: (1000, 762, 185) or (1000, 762, 370)
    eeg_test = np.load(os.path.join(eeg_path, 'concat_eeg_test_762.npy')) # Shape: (102, 762, 185) or (102, 762, 370)
    eeg_test_even = np.load(os.path.join(eeg_path, 'eeg_test_even_append.npy'))
    eeg_test_odd = np.load(os.path.join(eeg_path, 'eeg_test_odd_append.npy'))

elif args.eeg_channel_policy == 'average':
    eeg_train = np.load(os.path.join(eeg_path, 'eeg_train_subject_avg.npy')) # Shape: (1000, 127, 185) or (1000, 127, 370)
    eeg_test = np.load(os.path.join(eeg_path, 'eeg_test_subject_avg.npy')) # Shape: (102, 127, 185) or (102, 127, 370)
    eeg_test_even = np.load(os.path.join(eeg_path, 'eeg_test_even_average.npy'))
    eeg_test_odd = np.load(os.path.join(eeg_path, 'eeg_test_odd_average.npy'))
print('Shape of the EEG data (train, test):', eeg_train.shape, eeg_test.shape)

#################################
# Loading the fMRI data
################################
fmri_train, fmri_test = load_fmri_data(args.fmri_subject, args.hemisphere, roi=args.roi, threshold=20.0)
print("Shape of the fMRI data (train, test):", fmri_train.shape, fmri_test.shape)

######################################################################################################
# Fitting a linear model that predicts the responses of a group of vertices
# using all channels at each time point
######################################################################################################
alphas = np.logspace(-1, 1, 20) # List of alphas for Ridge regression
t_fmri = np.zeros((eeg_test.shape[2], fmri_test.shape[0], fmri_test.shape[1]), dtype=np.float32) # Array to store the EEG-predicted fMRI responses; shape: (n_timepoints, n_stimuli, n_vertices)
t_fmri_even = np.zeros((eeg_test.shape[2], fmri_test.shape[0], fmri_test.shape[1]), dtype=np.float32) # Array to store the EEG-predicted fMRI responses; shape: (n_timepoints, n_stimuli, n_vertices)
t_fmri_odd = np.zeros((eeg_test.shape[2], fmri_test.shape[0], fmri_test.shape[1]), dtype=np.float32) # Array to store the EEG-predicted fMRI responses; shape: (n_timepoints, n_stimuli, n_vertices)
corrs = []
print("Starting training and prediction...")
for t in tqdm.tqdm(range(eeg_train.shape[2])):
    eeg2fmri = RidgeCV(alphas=alphas, store_cv_results=True)
    eeg2fmri.fit(eeg_train[:, :, t], fmri_train)
    pred_fmri = eeg2fmri.predict(eeg_test[:, :, t])

    # Correlating the model's prediction with the actual test fMRI responses
    corrs.append([np.corrcoef(pred_fmri[:, j], fmri_test[:, j])[0, 1] for j in range(fmri_test.shape[1])])

    # Storing the preedicted t-fMRI data
    t_fmri[t,:,:] = pred_fmri
    t_fmri_even[t,:,:] = eeg2fmri.predict(eeg_test_even[:, :, t])
    t_fmri_odd[t,:,:] = eeg2fmri.predict(eeg_test_odd[:, :, t])
    
print("Training and prediction complete!")
corrs = np.array(corrs, dtype=np.float32)
print("Shape of the correlation matrix:", corrs.shape)
print("Shape of the predicted fMRI responses:", t_fmri.shape)
#############################################################################
# Saving the correlations and t-fMRI data
#############################################################################

# Saving the correlations
file_name = f'{args.roi}_{args.hemisphere}.npy'
corrs_save_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/encoding_fusion_roi/{args.eeg_frequency}_Hz/eeg_channel_policy_{args.eeg_channel_policy}/fmri_sub-{args.fmri_subject}'
if os.path.isdir(corrs_save_dir) == False:
	os.makedirs(corrs_save_dir)
np.save(os.path.join(corrs_save_dir, file_name), corrs)

# Saving the t-fMRI data
t_fmri_save_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/t_fmri/eeg2fmri_roi/{args.eeg_frequency}_Hz/eeg_channel_policy-{args.eeg_channel_policy}/fmri_sub-{args.fmri_subject}'
if os.path.isdir(t_fmri_save_dir) == False:
	os.makedirs(t_fmri_save_dir)

file_name_even = f'{args.roi}_{args.hemisphere}_even.npy'
file_name_odd = f'{args.roi}_{args.hemisphere}_odd.npy'

np.save(os.path.join(t_fmri_save_dir, file_name), t_fmri)
np.save(os.path.join(t_fmri_save_dir, file_name_even), t_fmri_even)
np.save(os.path.join(t_fmri_save_dir, file_name_odd), t_fmri_odd)


# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")
