import numpy as np
import os
import random
import time
from tqdm import tqdm
import argparse


# Start time
start_time = time.time()

# Random seed for reproducibility

seed = 8
np.random.seed(seed)
random.seed(seed)

#========================
# Input arguments
#========================

parser = argparse.ArgumentParser()
parser.add_argument('--eeg_frequency', type=int, default=100)
parser.add_argument('--eeg_channel_policy', type=str, default='average', 
                    help='Policy for handling EEG channels: "append" to use all channels, "average" to average across subjects',
                    choices=['append', 'average'])
args = parser.parse_args()

print(f'>>> Aggregating Whole-Brain Encoding Fusion Correlations <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

subject_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']   # List of subjects to process
for subject in tqdm(subject_list):

    path_l = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/eeg2fmri_wb/{args.eeg_frequency}_Hz/eeg_channel_policy_{args.eeg_channel_policy}/fmri_sub-{subject}/left_hemisphere'
    path_r = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/eeg2fmri_wb/{args.eeg_frequency}_Hz/eeg_channel_policy_{args.eeg_channel_policy}/fmri_sub-{subject}/right_hemisphere'


    data_l = [[] for _ in range(370)] # This list will contain 370 lists of 163,842 correlations each
    for i in range(1,22):
        splits_path = path_l+'/fmri_split-'+str(i) #iterating over the split folders
        corrs = np.load(splits_path+'.npy', allow_pickle=True)
        for j in range(len(corrs)):
            data_l[j].extend(corrs[j]) # filling out

    data_r = [[] for _ in range(370)] # This list will contain 370 lists of 163,842 correlations each
    for i in range(1,22):
        splits_path = path_r+'/fmri_split-'+str(i) #iterating over the split folders
        corrs = np.load(splits_path+'.npy', allow_pickle=True)
        for j in range(len(corrs)):
            data_r[j].extend(corrs[j]) # filling out


    save_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/results/correlations/eeg2fmri_wb/{args.eeg_frequency}_Hz/eeg_channel_policy_{args.eeg_channel_policy}/fmri_sub-{subject}'
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)

    file_name_l = 'correlations_left.npy'
    file_name_r = 'correlations_right.npy'

    left_data = np.array(data_l)
    right_data = np.array(data_r)
    np.save(os.path.join(save_dir, file_name_l), left_data)
    np.save(os.path.join(save_dir, file_name_r), right_data)


    print(f"Correlations for subject {subject} saved!")

# End time
end_time = time.time()

print(f"Total time taken: {end_time - start_time:.2f} seconds.")
print("Execution complete!")