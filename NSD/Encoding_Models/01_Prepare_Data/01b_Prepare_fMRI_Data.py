"""Prepare the EEG and fMRI data later used for EEG-fMRI encoding fusion.

To reduce computational load, the EEG-fMRI fusion encoding models are only
trained, tested, and used for voxels falling within the NSD visual streams.

Parameters
----------
subject : int
    Subject identifiers. Valid subject identifiers are integers from 1, 4, 5, 6, 7 and 8 (EEG data for subjects 2 and 3 not yet available).
hemispheres : list
    List containing the hemispheres used for the analyses. Possible values 
    are: 'lh' (left hemisphere) and 'rh' (right hemisphere).
berg_dir : str
    Directory of the BERG.
tnsd_dir : str
    Directory of the Temporal Natural Scenes Dataset.

"""

import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm
import time

# Start time
start_time = time.time()

seed = 8
np.random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--hemispheres', default=['lh', 'rh'], type=list)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Prepare data <<<')
print('Input arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Create the save directories
# =============================================================================
save_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/prepared_data'
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Prepare and store the fMRI responses
# =============================================================================
# Load the fMRI train/test image numbers
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-nsd_fsaverage')
meta_file_name = f'metadata_subject-{args.subject}.npy'
metadata_fmri = np.load(os.path.join(data_dir, meta_file_name),
    allow_pickle=True).item()
img_presentation_order = metadata_fmri['img_presentation_order']
img_presentation_order += 1 # since the EEG image numbers are 1 based
train_img_num = metadata_fmri['train_img_num']
train_img_num.sort()
train_img_num += 1 # since the EEG image numbers are 1 based
test_img_num = metadata_fmri['test_img_num']
test_img_num.sort()
test_img_num += 1 # since the EEG image numbers are 1 based

# Loop across hemispheres
for hemi in tqdm(args.hemispheres):
    print("Processing data for hemisphere:", hemi)

    # Load the fMRI responses
    fmri_file_name = f'{hemi}_betas_subject-{args.subject}.h5'
    fmri = h5py.File(os.path.join(data_dir, fmri_file_name), 'r')['betas']

    # Store the fMRI responses for the training images, averaged across repeats
    fmri_train = []
    for img_num in train_img_num:
        idx = np.where(img_presentation_order == img_num)[0]
        fmri_train.append(np.nanmean(fmri[idx], 0))
    fmri_train = np.nan_to_num(np.array(fmri_train))
    print(f"Shape of fmri_train: {fmri_train.shape}")
    fmri_train_dict = {
        'fmri_train': fmri_train,
        'train_img_num': train_img_num
    }
    np.save(os.path.join(save_dir, (f'fmri_train_sub-{args.subject:02d}_'
        f'hemi-{hemi}.npy')), fmri_train_dict)
    del fmri_train, fmri_train_dict

    # Store the fMRI responses responses for the test images
    fmri_test = []
    for img_num in test_img_num:
        idx = np.where(img_presentation_order == img_num)[0]
        fmri_test.append(fmri[idx])
    fmri_test = np.nan_to_num(np.array(fmri_test))
    print(f"Shape of fmri_test: {fmri_test.shape}")
    fmri_test_dict = {
        'fmri_test': fmri_test,
        'test_img_num': test_img_num
    }
    np.save(os.path.join(save_dir, (f'fmri_test_sub-{args.subject:02d}_'
        f'hemi-{hemi}.npy')), fmri_test_dict)
    del fmri, fmri_test, fmri_test_dict

# End time
end_time = time.time()
execution_time = end_time - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.") 