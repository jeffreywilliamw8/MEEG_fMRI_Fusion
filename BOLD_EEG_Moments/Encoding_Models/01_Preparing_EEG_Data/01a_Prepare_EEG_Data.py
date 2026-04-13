import numpy as np
import random
import argparse
from sklearn.preprocessing import StandardScaler
import time
import os

# Start time    
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--eeg_frequency', default=100, type=int, help='Sampling frequency of the EEG data', choices=[50, 100])
args, unknown = parser.parse_known_args()

print('>>> Preparing Single-Subject, Subject-Averaged and Subject-Appended EEG Data <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 8
random.seed(seed)
np.random.seed(seed)

N_TIMEPOINTS = int(args.eeg_frequency * 3.7) # 185 0r 370 time points per trial, depending on sampling frequency
N_CHANNELS = 127
N_STIMULI = 1102
N_TRAIN_STIMULI = 1000
N_TEST_STIMULI = 102
N_SUBJECTS = 6
SAVE_DIR = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/prepared_eeg/{args.eeg_frequency}_Hz'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def load_and_prepare_EEG_single_sub(sub_nr):
    ALL_TRIALS = np.arange(1, 1103)
    
    data_path = os.path.join('/scratch/giffordale95/projects/eeg_moments/dataset/preprocessed_data/eeg', 
                         'sub-0{}'.format(sub_nr), 
                         f'mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-{args.eeg_frequency:04d}', 
                         'preprocessed_data.npy')
    
    data_dict = np.load(data_path, allow_pickle=True).item() #extracts a dictionary

    # set EEG params
    times = data_dict['times']
    ch_names = data_dict['ch_names']
    info = data_dict['info']
    eeg_data_list = data_dict['eeg_data']
    stimuli_presentation_order_list = data_dict['stimuli_presentation_order']
    
    # average EEG responses
    stimuliOrders = data_dict['stimuli_presentation_order']
    concatenatedStimuliOrders = np.concatenate(stimuliOrders)
    concatenatedEEGData = np.concatenate(data_dict['eeg_data'], axis=0)

    del data_dict, eeg_data_list
    
    # for each video: 
    # 1. get subset of EEG data corresponding to that video's presentations
    # 2. average that subset 
    # 3. write to array for averaged EEG data
    averagedEEGData = np.empty((N_STIMULI, N_CHANNELS, N_TIMEPOINTS))
    for i, trialID in enumerate(ALL_TRIALS):
        trialIndices = np.argwhere(concatenatedStimuliOrders == trialID).squeeze()
        trialEEGData = concatenatedEEGData[trialIndices]
        meanTrialEEG = trialEEGData.mean(axis=0)
        averagedEEGData[i] = meanTrialEEG

    del concatenatedEEGData
    
    # divide data into training and test sets
    trainData = averagedEEGData[:N_TRAIN_STIMULI]
    testData = averagedEEGData[N_TRAIN_STIMULI:]

    assert testData.shape == (N_TEST_STIMULI, N_CHANNELS, N_TIMEPOINTS)
    assert trainData.shape == (N_TRAIN_STIMULI, N_CHANNELS, N_TIMEPOINTS)

    del averagedEEGData
    
    return trainData, testData



all_subs_train = []
all_subs_test = []

for sub_idx in range(1, N_SUBJECTS + 1):
    print(f"--- Processing Subject 0{sub_idx} ---")
    
    # Load your data: shape (Stimuli, Channels, Timepoints)
    train_raw, test_raw = load_and_prepare_EEG_single_sub(sub_idx)

    # Initialize arrays for z-scored data
    train_z = np.zeros_like(train_raw)
    test_z = np.zeros_like(test_raw)

    # Loop across time points independently
    for t in range(N_TIMEPOINTS):
        scaler = StandardScaler()
        
        # Extract slices: (n_stimuli, n_channels)
        train_slice = train_raw[:, :, t]
        test_slice = test_raw[:, :, t]

        # Fit on training stimuli at THIS time point only
        train_z[:, :, t] = scaler.fit_transform(train_slice)
        
        # Apply to test stimuli at THIS time point only
        test_z[:, :, t] = scaler.transform(test_slice)

    # Save single-subject z-scored responses
    np.save(f'{SAVE_DIR}/sub-0{sub_idx}_train_z.npy', train_z)
    np.save(f'{SAVE_DIR}/sub-0{sub_idx}_test_z.npy', test_z)
    print(f"Subject 0{sub_idx} saved.")

    all_subs_train.append(train_z)
    all_subs_test.append(test_z)

# 1. Compute and save Subject-Averaged responses
print("Computing subject-averaged responses...")
avg_train = np.mean(all_subs_train, axis=0)
avg_test = np.mean(all_subs_test, axis=0)

np.save(f'{SAVE_DIR}/eeg_train_subject_avg.npy', avg_train)
np.save(f'{SAVE_DIR}/eeg_test_subject_avg.npy', avg_test)

# 2. Append along Channel Dimension (Concatenation)
# Yields (1000, 762, N_TIMEPOINTS) and (102, 762, N_TIMEPOINTS)
print("Fusing subjects along the channel dimension (127 * 6 = 762)...")
fused_train = np.concatenate(all_subs_train, axis=1)
fused_test = np.concatenate(all_subs_test, axis=1)

# Final Saves
np.save(f'{SAVE_DIR}/concat_eeg_train_762.npy', fused_train)
np.save(f'{SAVE_DIR}/concat_eeg_test_762.npy', fused_test)

print(f"Finished! Fused Train Shape: {fused_train.shape}")
print(f"Finished! Fused Test Shape: {fused_test.shape}")


# End time
end_time = time.time()
execution_time = end_time - start_time

print(f"Execution complete! Time: {execution_time:.2f} seconds.")