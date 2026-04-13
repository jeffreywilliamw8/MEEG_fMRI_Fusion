import numpy as np
import os
import argparse
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
import time

# Start time
start_time = time.time()

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--eeg_subject', type=str, default='01',
                    choices=['01', '02', '03', '04', '05', '06', 'avg', 'app'],
                    help="Subject to decode: '01'-'06' for individual subjects, 'avg' for subject-averaged, 'app' for all subjects appended along channels")
parser.add_argument('--eeg_frequency', type=int, default=100)
args = parser.parse_args()

print('>>> Computing Decoding Accuracy for EEG RDMs <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
# =============================================================================
# Paths and Loading
# =============================================================================
data_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/prepared_eeg/{args.eeg_frequency}_Hz'
out_dir = f'/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/results/decoding_rdms/{args.eeg_frequency}_Hz'
os.makedirs(out_dir, exist_ok=True)

def load_data(sub_arg):
    if sub_arg in ['01', '02', '03', '04', '05', '06']:
        fname = f'sub-{sub_arg}_single_trial_eeg.npy'
    elif sub_arg == 'avg':
        fname = 'subject_averaged_single_trial_eeg.npy'
    elif sub_arg == 'app':
        fname = 'single_trial_eeg_762ch.npy'
    else:
        raise ValueError(f"Unknown subject argument: {sub_arg}")
    
    return np.load(os.path.join(data_dir, fname))

# Load data: (102, 24, n_channels, n_time)
eeg_data = load_data(args.eeg_subject)
n_stim, n_trials, n_chan, n_time = eeg_data.shape

# =============================================================================
# Decoding Pipeline
# =============================================================================
# 102x102 RDM has (102*101)/2 = 5151 unique pairs in upper triangle
rows, cols = np.triu_indices(n_stim, k=1)
n_pairs = len(rows)

# Storage for (Time, Pairs)
rdms = np.zeros((n_time, n_pairs))

# Linear SVM setup
clf = LinearSVC(C=1.0, max_iter=1000, tol=1e-3, random_state=42)
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

print("Starting time-resolved pairwise decoding...")
for t in tqdm(range(n_time)):
    # Slice data at current timepoint: (102, 24, n_channels)
    current_eeg = eeg_data[:, :, :, t]
    
    for p_idx, (i, j) in enumerate(zip(rows, cols)):
        # Observations: trials from stim i and stim j
        X_i = current_eeg[i] # (24, n_chan)
        X_j = current_eeg[j] # (24, n_chan)
        
        X = np.concatenate([X_i, X_j], axis=0)
        y = np.concatenate([np.zeros(n_trials), np.ones(n_trials)])
        
        # 4-fold cross-validation
        scores = cross_val_score(clf, X, y, cv=cv)
        rdms[t, p_idx] = np.mean(scores)
print("Decoding complete!")
# =============================================================================
# Save Results
# =============================================================================
save_name = f"decoding_rdm_eeg_sub-{args.eeg_subject}.npy"
save_path = os.path.join(out_dir, save_name)

np.save(save_path, rdms)

# End time
end_time = time.time()
execution_time = end_time - start_time

print(f"Execution complete! Time: {execution_time:.2f} seconds.")