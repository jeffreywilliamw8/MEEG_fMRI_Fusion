import argparse
import os
import numpy as np
import h5py
from berg import BERG
from tqdm import tqdm
import random
from sklearn.linear_model import RidgeCV
import time
import gc

# Start time
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--meg_subjects', default=[1, 2, 3, 4], type=list)
parser.add_argument('--meg_channel_policy', type=str, default='append')
parser.add_argument('--berg_dir', default='/scratch/jeffreykatab/Code/Encoding_Models/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print(f'>>> MEG-fMRI Encoding Fusion (ROI) <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 8
random.seed(seed)
np.random.seed(seed)

# =============================================================================
# Loading the fMRI data
# =============================================================================
# Loading the metadata
berg = BERG(berg_dir=args.berg_dir)
metadata_fmri = berg.get_model_metadata(
    'fmri-things_fmri_1-vit_b_32',
    subject=args.fmri_subject
    )

# Loading the fMRI responses
fmri_train_file = os.path.join('/scratch/jeffreykatab/berg', 'model_training_datasets',
    'train_dataset-things_fmri_1',
    f'fmri_sub-{args.fmri_subject:02d}_split-train.h5')
fmri_train = h5py.File(fmri_train_file, 'r')['neural_data']

fmri_test_file = os.path.join('/scratch/jeffreykatab/berg', 'model_training_datasets',
    'train_dataset-things_fmri_1',
    f'fmri_sub-{args.fmri_subject:02d}_split-test.h5')
fmri_test_all = h5py.File(fmri_test_file, 'r')['neural_data']

# Getting the image files names
train_stimuli_fmri = metadata_fmri['encoding_model']['train_stimuli']
test_stimuli_fmri = metadata_fmri['encoding_model']['test_stimuli']
unique_test_stimuli = np.unique(test_stimuli_fmri)

# Averaging the fMRI responses across repetitions of the same test stimulus
fmri_test = []
for stim in tqdm(unique_test_stimuli):
    idx = np.where(test_stimuli_fmri == stim)[0]
    fmri_test.append(fmri_test_all[idx].mean(0))
fmri_test = np.array(fmri_test)

# Selecting ROI voxels
roi_idx = metadata_fmri['roi'][args.roi]
whole_brain_nc = metadata_fmri['encoding_model']['noise_ceiling_testset']
roi_noise_ceilings = whole_brain_nc[roi_idx]
valid_voxels = np.where(roi_noise_ceilings > 20.0)[0] # Selecting voxels above noise ceiling threshold of 20%
fmri_train = fmri_train[:, valid_voxels].astype(np.float32) # Shape: (8640, n_voxels)
fmri_test = fmri_test[:, valid_voxels].astype(np.float32) # Shape: (8640, n_voxels)

eps = 1e-8
fmri_test_z = (fmri_test - fmri_test.mean(0)) / (fmri_test.std(0) + eps)
print("Shape of the fMRI data (train, test): ", fmri_train.shape, fmri_test.shape)

# ==============================================================================================
# Loading the MEG data
# ==============================================================================================
# Loop across subjects
for ms, msub in enumerate(tqdm(args.meg_subjects)):

    # Loading the MEG metadata
    metadata_meg = berg.get_model_metadata(
        'meg-things_meg_1-vit_b_32',
        subject=msub
    )

    # Time point selection
    tmax = 0.8 # We limit our analysis to the first 800 ms after stimulus onset
    times = metadata_meg['meg']['times']
    time_idx = np.zeros(len(times), dtype=int)
    time_idx[times <= tmax] = 1
    time_idx = np.where(time_idx == 1)[0]
    times = times[times <= tmax]

    # Loading the MEG responses
    meg_train_file = os.path.join('/scratch/jeffreykatab/berg', 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_all_training_splits.h5')
    meg_train_sub = h5py.File(meg_train_file, 'r')['neural_data']


    meg_test_file = os.path.join('/scratch/jeffreykatab/berg', 'model_training_datasets',
        'train_dataset-things_meg_1', f'meg_P{msub}_split-test.h5')
    meg_test_all = h5py.File(meg_test_file, 'r')['neural_data']
    meg_test_all = meg_test_all[:,:,time_idx].astype(np.float32)

    # Averaging the MEG responses across repetitions for the images shared with the fMRI data
    test_stimuli_meg = metadata_meg['encoding_model']['test_stimuli']
    meg_test_sub = []
    meg_test_sub_even = []
    meg_test_sub_odd = []
    for stim in unique_test_stimuli:
        idx = [i for i, x in enumerate(test_stimuli_meg) if x == stim]
        meg_test_sub.append(meg_test_all[idx].mean(0))
        # Repetitions for this stimulus: shape (12, sensors, time)
        reps = meg_test_all[idx]
        
        # Create pseudo-trials by averaging subsets
        # Even indices: 0, 2, 4, 6, 8, 10 (6 repeats)
        # Odd indices:  1, 3, 5, 7, 9, 11 (6 repeats)
        meg_test_sub_even.append(reps[::2].mean(0))
        meg_test_sub_odd.append(reps[1::2].mean(0))
        
    meg_test_sub = np.array(meg_test_sub)
    meg_test_sub_even = np.array(meg_test_sub_even)
    meg_test_sub_odd = np.array(meg_test_sub_odd)

    
    # Getting the MEG responses for the images shared with the fMRI
    train_stimuli_meg = metadata_meg['encoding_model']['all_training_splits']['train_stimuli']
    idx_meg = []
    for stim in train_stimuli_fmri:
        idx_meg.append(train_stimuli_meg.index(stim))
    idx_meg = np.array(idx_meg)
    meg_train_sub = meg_train_sub[:,:,time_idx][idx_meg].astype(np.float32)

    if args.meg_channel_policy == 'append':
        # Appending the MEG sensor responses across subjects
        if ms == 0:
            meg_train = meg_train_sub
            meg_test = meg_test_sub
            meg_test_even = meg_test_sub_even
            meg_test_odd = meg_test_sub_odd
        else:
            meg_train = np.append(meg_train, meg_train_sub, 1)
            meg_test = np.append(meg_test, meg_test_sub, 1)
            meg_test_even = np.append(meg_test_even, meg_test_sub_even, axis=1)
            meg_test_odd = np.append(meg_test_odd, meg_test_sub_odd, axis=1)
        del meg_train_sub, meg_test_all, meg_test_sub, meg_test_sub_even, meg_test_sub_odd
    
    elif args.meg_channel_policy == 'average':
        # Summing the MEG sensor responses across subjects
        if ms == 0:
            meg_train = meg_train_sub
            meg_test = meg_test_sub
            meg_test_even = meg_test_sub_even
            meg_test_odd = meg_test_sub_odd
        else:
            meg_train += meg_train_sub
            meg_test += meg_test_sub
            meg_test_even += meg_test_sub_even
            meg_test_odd += meg_test_sub_odd
        
        # If this is the last subject, divide by the total number of subjects
        if ms == len(args.meg_subjects) - 1:
            n_subs = len(args.meg_subjects)
            meg_train /= n_subs
            meg_test /= n_subs
            meg_test_even /= n_subs
            meg_test_odd /= n_subs

        del meg_train_sub, meg_test_all, meg_test_sub, meg_test_sub_even, meg_test_sub_odd


print("Shape of the MEG data (train, test): ", meg_train.shape, meg_test.shape) 

# =============================================================================
# Training and testing the encoding fusion models
# =============================================================================
alphas = np.logspace(-6, 10, 17)
n_test_stimluli = fmri_test.shape[0]
n_voxels = fmri_test.shape[1]
n_time_points = len(times)
correlations = np.zeros((n_time_points, n_voxels), dtype=np.float32)
t_fmri = np.zeros((n_time_points, n_test_stimluli, n_voxels), dtype=np.float32)
t_fmri_even = np.zeros((n_time_points, n_test_stimluli, n_voxels), dtype=np.float32)
t_fmri_odd = np.zeros((n_time_points, n_test_stimluli, n_voxels), dtype=np.float32)


print("Starting encoding fusion...")
for t in tqdm(range(len(times))):

    # Train the encoding fusion models
    meg2fmri = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    meg2fmri.fit(meg_train[:,:,t], fmri_train)

    # Generate the t-fMRI responses for the test images with in vivo MEG
    pred_fmri = meg2fmri.predict(meg_test[:,:,t])

    # Center and normalize the t-fMRI responses
    pred_fmri_z = (pred_fmri - pred_fmri.mean(0)) /  (pred_fmri.std(0) + eps)

    # Correlate the t-fMRI test responses with the fMRI test responses
    correlations[t,:] = np.diag(pred_fmri_z.T @ fmri_test_z) / len(pred_fmri_z)

    # Store t-fMRI responses
    t_fmri[t,:,:] = pred_fmri
    t_fmri_even[t,:,:] = meg2fmri.predict(meg_test_even[:,:,t])
    t_fmri_odd[t,:,:] = meg2fmri.predict(meg_test_odd[:,:,t])


    # Delete unused variables
    del pred_fmri, pred_fmri_z, meg2fmri
    gc.collect()
del fmri_test, fmri_test_z, meg_train, meg_test, meg_test_even, meg_test_odd
print("Encoding fusion complete!")
print("Shape of the t-fMRI responses: ", t_fmri.shape)

# =============================================================================
# Saving the encoding fusion correlations and t-fMRI responses
# =============================================================================
corrs_save_dir = f'/home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/correlations/encoding_fusion_roi/meg_channel_policy-{args.meg_channel_policy}/fmri_sub-{args.fmri_subject:02d}'
t_fmri_save_dir = f'/home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/results/t_fmri/meg_channel_policy-{args.meg_channel_policy}/fmri_sub-{args.fmri_subject:02d}'
os.makedirs(corrs_save_dir, exist_ok=True)
os.makedirs(t_fmri_save_dir, exist_ok=True)
file_name = f'{args.roi}.npy'

# Saving the correlations
np.save(os.path.join(corrs_save_dir, file_name), correlations)

# Saving the t-fMRI responses
file_name_even = f'{args.roi}_even.npy'
file_name_odd = f'{args.roi}_odd.npy'
np.save(os.path.join(t_fmri_save_dir, file_name), t_fmri)
np.save(os.path.join(t_fmri_save_dir, file_name_even), t_fmri_even)
np.save(os.path.join(t_fmri_save_dir, file_name_odd), t_fmri_odd)

# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")
