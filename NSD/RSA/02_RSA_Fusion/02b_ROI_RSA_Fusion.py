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
from utils import load_fmri_roi_data2

# Start time
start_time = time.time()

seed = 8
np.random.seed(seed)
random.seed(seed)

#=============================================================================
# Input arguments
#=============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--eeg_rdm_metric', type=str, default='correlation', choices=['correlation', 'decoding_accuracy', 'cosine', 'euclidean'])
parser.add_argument('--fmri_rdm_metric', type=str, default='correlation', choices=['correlation', 'cosine', 'euclidean'])

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
if args.eeg_rdm_metric in ['correlation', 'cosine', 'euclidean']:
    data_path = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
    eeg_test = np.load(os.path.join(data_path, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True).item()['eeg_test'] # Shape: (515, 30, 160, 359)
    eeg_test = np.mean(eeg_test, axis=1) # Averaging across repeats to get shape (515, 160, 359)
    sub_sample_idx = np.random.choice(eeg_test.shape[0], size=100, replace=False)
    eeg_test = eeg_test[sub_sample_idx] # Shape after sub-sampling: (100, 160, 359)
    eeg_rdms = [flatten_rdm(pairwise_distances(eeg_test[:,:,t], metric=args.eeg_rdm_metric)) for t in range(eeg_test.shape[2])]
    eeg_rdms = np.array(eeg_rdms, dtype=np.float32)
    del eeg_test
elif args.eeg_rdm_metric == 'decoding_accuracy':
    data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/decoding_rdms'
    eeg_rdms = np.load(os.path.join(data_dir, f"decoding_rdm_eeg_sub-{args.subject}.npy"))

print("Shape of the EEG RDMs: ", eeg_rdms.shape)
#=============================================================================
# Loading the fMRI data
#=============================================================================
_, fmri_test = load_fmri_roi_data2(args.subject,  roi=args.roi, nc_threshold=0.2)
fmri_test = fmri_test[sub_sample_idx]
fmri_rdm = flatten_rdm(pairwise_distances(fmri_test, metric=args.fmri_rdm_metric)) # Shape: (4950,)
del fmri_test
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

save_dir = os.path.join(
    '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/',
    'correlations',
    'roi',
    f'eeg_rdm_metric-{args.eeg_rdm_metric}',
    f'fmri_rdm_metric-{args.eeg_rdm_metric}',
    f'subject-{args.subject}'
)
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
    
file_name = f'{args.roi}.npy'
np.save(os.path.join(save_dir, file_name), correlations)

# End time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution complete! Time: {execution_time:.2f} seconds.")