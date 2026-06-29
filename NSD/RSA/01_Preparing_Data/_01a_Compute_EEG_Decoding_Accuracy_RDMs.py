import numpy as np
import os
import argparse
import h5py
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
import random
import time

start_time = time.time()
seed = 8
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
args = parser.parse_args()

# Paths
data_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
out_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/decoding_rdms'
os.makedirs(out_dir, exist_ok=True)

# Load data
eeg_dict = np.load(os.path.join(data_dir, f'eeg_test_sub-{args.subject:02d}.npy'), allow_pickle=True).item()
eeg_data = eeg_dict['eeg_test']
sub_sample_idx = np.random.choice(eeg_data.shape[0], size=100, replace=False)
eeg_data = eeg_data[sub_sample_idx,:,:]
print(f"EEG data shape: {eeg_data.shape} (Stimuli, Trials, Channels, Time)")
n_stim, n_trials, n_chan, n_time = eeg_data.shape

# 1. PSEUDO-TRIALS (6 pseudo-trials)
n_pseudo = 6
trials_per_pseudo = n_trials // n_pseudo
pseudo_data = eeg_data.reshape(n_stim, n_pseudo, trials_per_pseudo, n_chan, n_time).mean(axis=2)
print(f"Data reshaped to pseudo-trials: {pseudo_data.shape} (Stimuli, Pseudo-trials, Channels, Time)")

# 2. SETUP PAIRS AND H5PY
rows, cols = np.triu_indices(n_stim, k=1)
n_pairs = len(rows)

h5_path = os.path.join(out_dir, f"decoding_rdm_eeg_sub-{args.subject}.h5")

# Open file in write mode ('w')
with h5py.File(h5_path, 'w') as f:
    # Pre-allocate the dataset: (Time, Pairs)
    dset = f.create_dataset("rdms", (n_time, n_pairs), dtype='float32', compression="gzip")
    
    clf = LinearSVC(C=1.0, max_iter=1000, tol=1e-3, random_state=8)
    cv = StratifiedKFold(n_splits=n_pseudo, shuffle=True, random_state=8)

    print(f"Starting decoding. Saving to HDF5: {h5_path}")
    
    for t in tqdm(range(n_time)):
        current_data = pseudo_data[:,:,t]
        timepoint_rdm = np.zeros(n_pairs) # Temporary storage for this row
        
        for p_idx, (i, j) in enumerate(zip(rows, cols)):
            X_i = current_data[i] 
            X_j = current_data[j] 

            X = np.concatenate([X_i, X_j], axis=0)
            y = np.concatenate([np.zeros(n_pseudo), np.ones(n_pseudo)])
            
            # Simple average of cross-validation scores
            scores = cross_val_score(clf, X, y, cv=cv)
            timepoint_rdm[p_idx] = np.mean(scores)
        
        # Incremental write: only this timepoint's row is sent to disk
        dset[t, :] = timepoint_rdm
        # Flush ensures data is written in case of a crash, but doesn't rewrite the whole file
        f.flush() 

print(f"Done! Total Time: {time.time() - start_time:.2f} seconds.")