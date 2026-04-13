import numpy as np
from tqdm import tqdm
import time
from tqdm import tqdm
import argparse
import random
import os
from scipy.stats import zscore

# Start time
start_time = time.time()

# Random seed for reproducibility
seed = 8
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--eeg_frequency', type=int, default=100)

args = parser.parse_args()

print('>>> Preparing single-trial EEG Data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


#################################################
# Loading the EEG data
#################################################

path1 = '/scratch/giffordale95/projects/eeg_moments/dataset/preprocessed_data/eeg/sub-01/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-'+format(args.eeg_frequency, '04')
path2 = '/scratch/giffordale95/projects/eeg_moments/dataset/preprocessed_data/eeg/sub-02/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-'+format(args.eeg_frequency, '04')
path3 = '/scratch/giffordale95/projects/eeg_moments/dataset/preprocessed_data/eeg/sub-03/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-'+format(args.eeg_frequency, '04')
path4 = '/scratch/giffordale95/projects/eeg_moments/dataset/preprocessed_data/eeg/sub-04/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-'+format(args.eeg_frequency, '04')
path5 = '/scratch/giffordale95/projects/eeg_moments/dataset/preprocessed_data/eeg/sub-05/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-'+format(args.eeg_frequency, '04')
path6 = '/scratch/giffordale95/projects/eeg_moments/dataset/preprocessed_data/eeg/sub-06/mvnn-time/baseline_correction-01/highpass-0.01_lowpass-100/sfreq-'+format(args.eeg_frequency, '04')

eeg_data_dict_1 = np.load(path1+'/preprocessed_data.npy', allow_pickle=True).item()
eeg_data_dict_2 = np.load(path2+'/preprocessed_data.npy', allow_pickle=True).item()
eeg_data_dict_3 = np.load(path3+'/preprocessed_data.npy', allow_pickle=True).item()
eeg_data_dict_4 = np.load(path4+'/preprocessed_data.npy', allow_pickle=True).item()
eeg_data_dict_5 = np.load(path5+'/preprocessed_data.npy', allow_pickle=True).item()
eeg_data_dict_6 = np.load(path6+'/preprocessed_data.npy', allow_pickle=True).item()


#############################################################################
# Formatting each EEG dataset into a suitable form for pairwise SVM decoding
############################################################################
def extract_single_trial_data(eeg_data_dict):

    test_data = []
    for session_number in range(8):
        test_indices_list = []
        presentation_order = eeg_data_dict['stimuli_presentation_order'][session_number]
        # Finding from the stimuli presentation indices when each test video appears
        for i in range(1001,1103):
            test_indices_list.append(np.where(presentation_order == i)[0].tolist())
        # zscore(eeg_data_list[ses], 0)
        eeg_data = eeg_data_dict['eeg_data'][session_number]
        eeg_data = zscore(eeg_data, 0) # Z-scoring the data for each session
        test_trials = []
        for l in test_indices_list:
            test_trials.append(eeg_data[l]) # Extracting all the trials corresponding to the test videos for each session
        test_data.append(test_trials)
    # Concatenating data from all 8 sessions
    test_data = np.concatenate([test_data[0], test_data[1], test_data[2], test_data[3], test_data[4], test_data[5], test_data[6], test_data[7]], axis=1)
    print("Shape of the test data: ", test_data.shape)
    return test_data

eeg1 = extract_single_trial_data(eeg_data_dict_1)
eeg2 = extract_single_trial_data(eeg_data_dict_2)
eeg3 = extract_single_trial_data(eeg_data_dict_3)
eeg4 = extract_single_trial_data(eeg_data_dict_4)
eeg5 = extract_single_trial_data(eeg_data_dict_5)
eeg6 = extract_single_trial_data(eeg_data_dict_6)

subjects_data = [eeg1, eeg2, eeg3, eeg4, eeg5, eeg6]
out_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/prepared_eeg/{args.eeg_frequency}_Hz'
os.makedirs(out_dir, exist_ok=True)

# 1. Save single subject EEG data
print("--- Saving single-subject matrices ---")
for i, sub_data in enumerate(subjects_data):
    sub_nr = i + 1
    save_path = os.path.join(out_dir, f'sub-0{sub_nr}_single_trial_eeg_test.npy')
    np.save(save_path, sub_data)
    print(f"Saved Subject {sub_nr} with shape: {sub_data.shape}")

# 2. Save subject-averaged EEG data
print("\n--- Computing subject-averaged responses ---")
subject_avg = np.mean(np.stack(subjects_data, axis=0), axis=0)
np.save(os.path.join(out_dir, 'subject_averaged_single_trial_eeg_test.npy'), subject_avg)
print(f"Saved subject-averaged data with shape: {subject_avg.shape}")

# --- NEW: Save Pseudo-trials for Subject-Averaged Data ---
# Axis 1 is trials. reps[::2] = trials 0,2,4... | reps[1::2] = trials 1,3,5...
sub_avg_even = subject_avg[:, ::2, :, :].mean(axis=1)
sub_avg_odd = subject_avg[:, 1::2, :, :].mean(axis=1)
np.save(os.path.join(out_dir, 'eeg_test_even_average.npy'), sub_avg_even)
np.save(os.path.join(out_dir, 'eeg_test_odd_average.npy'), sub_avg_odd)
print(f"Saved sub-avg pseudo-trials. Shape: {sub_avg_even.shape}")


# 3. Append data across all 6 subjects along the channel dimension
print("\n--- Fusing subjects along channel dimension (127 * 6 = 762) ---")
fused_subjects = np.concatenate(subjects_data, axis=2)
save_path_fused = os.path.join(out_dir, 'single_trial_eeg_test_762ch.npy')
np.save(save_path_fused, fused_subjects)
print(f"Final Fused Matrix Shape: {fused_subjects.shape}")

# --- NEW: Save Pseudo-trials for Fused-Channel Data ---
fused_even = fused_subjects[:, ::2, :, :].mean(axis=1)
fused_odd = fused_subjects[:, 1::2, :, :].mean(axis=1)
np.save(os.path.join(out_dir, 'eeg_test_even_append.npy'), fused_even)
np.save(os.path.join(out_dir, 'eeg_test_odd_append.npy'), fused_odd)
print(f"Saved fused pseudo-trials. Shape: {fused_even.shape}")

# End time
end_time = time.time()
execution_time = end_time - start_time

print(f"Execution complete! Time: {execution_time:.2f} seconds.")
