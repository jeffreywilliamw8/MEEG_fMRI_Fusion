import argparse
import os
import numpy as np
from tnsd_access import TrialHandler
from scipy.stats import zscore
from tqdm import tqdm
import time

# Start time
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--nsd_dir', default='/scratch/ccn_datasets/natural-scenes-dataset', type=str)
parser.add_argument('--tnsd_dir', default='/scratch/giffordale95/datasets/temporal-natural-scenes-dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train encoding fusion <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

SAVE_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# =============================================================================
# Loading the fMRI metadata
# =============================================================================
# Load the fMRI responses and metadata
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-nsd_fsaverage')
meta_file_name = f'metadata_subject-{args.subject}.npy'
metadata_fmri = np.load(os.path.join(data_dir, meta_file_name),
    allow_pickle=True).item()

# Select the fMRI responses for the training images, and average them across
# repeats
train_img_num = metadata_fmri['train_img_num']
train_img_num.sort()
train_img_num += 1 # since the EEG image numbers are 1 based

# Store the fMRI responses responses for the test images
test_img_num = np.append(metadata_fmri['test_img_num'],
    metadata_fmri['val_img_num'])
test_img_num.sort()

test_img_num += 1 # since the EEG image numbers are 1 based


# =============================================================================
# Load the EEG responses
# =============================================================================
# Initialize tNSD data loader
loader = TrialHandler(args.tnsd_dir)

# Get the time points # !!! Use official time points
n_times = 615
times = np.round(np.linspace(-200, 1000, n_times)).astype(int)

# Account for the 50ms shift in the EEG responses # !!!
shift = -50
times = times + shift

# Only select time points between -100ms and 600ms
t_start = np.where(times == -100)[0][0]
t_end = np.where(times == 600)[0][0]
times = times[t_start:t_end+1]

# Load and z-score the EEG responses at each session
if args.subject == 2 or args.subject == 3:
    sessions = 27
else:
    sessions = 36
conditions = []
eeg = []
for ses in tqdm(range(1, sessions+1)):
    trials_sess = loader.lookup_trials(subject=args.subject, session=ses)
    data_sess = loader.get_data(trials_sess)
    conditions.append(np.array(data_sess['metadata']['condition']))
    eeg.append(zscore(data_sess['data'][:,:-4,t_start:t_end+1], 0)) # !!! Select channels using official channels
    del data_sess

# Concatenate the data across sessions
conditions = np.concatenate(conditions)
eeg = np.concatenate(eeg, 0)

# =================================================================================
# Select the EEG responses for the training images and average them across repeats
# =================================================================================
eeg_train = []
eeg_train_even = []
eeg_train_odd = []

print("Processing Training Set...")
for img_num in tqdm(train_img_num):
    idx = np.where(conditions == img_num)
    
    #  Total average
    eeg_train.append(np.mean(eeg[idx], axis=0))
    
    #  Split-half averages
    # idx[::2] selects 0, 2, 4... (Even indices)
    # idx[1::2] selects 1, 3, 5... (Odd indices)
    eeg_train_even.append(np.mean(eeg[idx[::2]], axis=0))
    eeg_train_odd.append(np.mean(eeg[idx[1::2]], axis=0))
print('Shape of the EEG data (train):', np.array(eeg_train).shape)

eeg_train_trial_avg_all_dict = {
    'eeg_train': np.array(eeg_train),
    'train_img_num': train_img_num
}
eeg_train_trial_avg_even_dict = {
    'eeg_train_even': np.array(eeg_train_even),
    'eeg_train_odd': np.array(eeg_train_odd)
}
eeg_train_trial_avg_odd_dict = {
    'eeg_train_odd': np.array(eeg_train_odd),
    'train_img_num': train_img_num
}
np.save(os.path.join(SAVE_DIR, f'eeg_train_sub-{args.subject:02d}_trial_avg-all.npy'), eeg_train_trial_avg_all_dict)
np.save(os.path.join(SAVE_DIR, f'eeg_train_sub-{args.subject:02d}_trial_avg-even.npy'), eeg_train_trial_avg_even_dict)
np.save(os.path.join(SAVE_DIR, f'eeg_train_sub-{args.subject:02d}_trial_avg-odd.npy'), eeg_train_trial_avg_odd_dict)


del eeg_train, eeg_train_even, eeg_train_odd
# =============================================================================
# Store the EEG responses for the test images and average them across repeats
# =============================================================================
eeg_test = []
eeg_test_even = []
eeg_test_odd = []

print("Processing Test Set...")
for img_num in tqdm(test_img_num):
    idx = np.where(conditions == img_num)

    # Total average
    eeg_test.append(np.mean(eeg[idx], axis=0))
    
    # Split-half averages
    eeg_test_even.append(np.mean(eeg[idx[::2]], axis=0))
    eeg_test_odd.append(np.mean(eeg[idx[1::2]], axis=0))


eeg_test_dict = {
    'eeg_test': np.array(eeg_test),
    'eeg_test_even': np.array(eeg_test_even),
    'eeg_test_odd': np.array(eeg_test_odd),
    'test_img_num': test_img_num
}

np.save(os.path.join(SAVE_DIR, f'eeg_test_sub-{args.subject:02d}.npy'), eeg_test_dict)

# End time
end_time = time.time()
execution_time = end_time - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.") 