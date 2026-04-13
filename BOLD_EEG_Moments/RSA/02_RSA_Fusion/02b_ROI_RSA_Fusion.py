import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import argparse
import random
import pickle
import numpy as np
import os
import random
from scipy.stats import spearmanr
import time 
from utils import load_fmri_data

# Start time
start_time = time.time()

seed = 8
np.random.seed(seed)
random.seed(seed)

#=============================================================================
# Input arguments
#=============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=str, default='01')
parser.add_argument('--hemisphere', type=str, default='left')
parser.add_argument('--roi', type=str, default='V1v')
parser.add_argument('--eeg_rdm_policy', type=str, default='appended_channels',
                    choices=['appended_channels', 'subject_averaged'],
                    help="Policy for computing EEG RDMs: 'appended_channels' concatenates all subjects along channels, 'subject_averaged' averages RDMs across subjects")
parser.add_argument('--eeg_rdm_distance', type=str, default='correlation',
                    choices=['correlation', 'decoding_accuracy', 'cosine', 'euclidean'],)
parser.add_argument('--eeg_frequency', type=int, default=100)
args = parser.parse_args()

print('>>> ROI RSA Fusion <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

def flatten_rdm(rdm):
    return (rdm[np.triu_indices_from(rdm, k=1)]).astype(np.float32)  # k=1 excludes diagonal

#=================================================================================
# Loading the EEG RDMs or computing them on the fly (depending on distance metric)
#=================================================================================
eeg_data_path = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/prepared_eeg/{args.eeg_frequency}_Hz'
if args.eeg_rdm_policy == 'appended_channels':
    if args.eeg_rdm_distance in ['correlation', 'cosine', 'euclidean']:
        eeg_data = np.load(os.path.join(eeg_data_path, 'concat_eeg_test_762.npy')) # Shape: (102, 762, n_time_points)
        eeg_rdms = np.array([flatten_rdm(pairwise_distances(eeg_data[:,:,t], metric=args.eeg_rdm_distance)) for t in range(eeg_data.shape[2])], dtype=np.float32) # Shape: (n_time_points, 5151)
    elif args.eeg_rdm_distance == 'decoding_accuracy':
        eeg_rdms = np.load(os.path.join(eeg_data_path, 'concat_eeg_test_762_decoding_rdm.npy')) # Shape: (n_time_points, 5151)
else:
    eeg_rdm_list = []
    for sub in ['01', '02', '03', '04', '05', '06']:
        if args.eeg_rdm_distance in ['correlation', 'cosine', 'euclidean']:
            sub_data = np.load(os.path.join(eeg_data_path, f'sub-{sub}_test_z.npy')) # Shape: (102, 127, n_time_points)
            sub_rdm = np.array([flatten_rdm(pairwise_distances(sub_data[:,:,t], metric=args.eeg_rdm_distance)) for t in range(sub_data.shape[2])], dtype=np.float32) # Shape: (n_time_points, 5151)
        elif args.eeg_rdm_distance == 'decoding_accuracy':
            sub_rdm = np.load(os.path.join(eeg_data_path, f'sub-{sub}_test_decoding_rdm.npy')) # Shape: (n_time_points, 5151)
        eeg_rdm_list.append(sub_rdm)
    eeg_rdms = np.mean(eeg_rdm_list, axis=0, dtype=np.float32) # Shape: (n_time_points, 5151)
print("Shape of the EEG RDM: ", eeg_rdms.shape)

#=============================================================================
# Loading the fMRI data
#=============================================================================
_, fmri_data = load_fmri_data(args.fmri_subject, args.hemisphere, roi=args.roi, threshold=20.0)
fmri_rdm = flatten_rdm(pairwise_distances(fmri_data, metric='correlation')) # Shape: (5151,)
del fmri_data
print("Shape of the fMRI RDM: ", fmri_rdm.shape)

#=============================================================================
# Fusion loop: at each EEG time point, compute the Spearman correlation
# between the EEG RDM and the fMRI ROI RDM 
#=============================================================================
correlations = np.zeros(eeg_rdms.shape[0], dtype=np.float32)
for t in tqdm(range(eeg_rdms.shape[0])):
    correlations[t]= spearmanr(eeg_rdms[t], fmri_rdm).correlation
#=============================================================================
# Saving the results
#=============================================================================
import os

save_dir = os.path.join(
    '/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/results/',
    'correlations',
    'roi_fusion',
    f'{args.eeg_frequency}_Hz',
    f'eeg_rdm_policy-{args.eeg_rdm_policy}',
    f'eeg_rdm_distance-{args.eeg_rdm_distance}',
    f'fmri_sub-{args.fmri_subject}'
)
os.makedirs(save_dir, exist_ok=True)

file_name = f'{args.roi}_{args.hemisphere}.npy'
np.save(os.path.join(save_dir, file_name), correlations)

# End time
end_time = time.time()
execution_time = end_time - start_time

print(f"Execution complete! Time: {execution_time:.2f} seconds.")