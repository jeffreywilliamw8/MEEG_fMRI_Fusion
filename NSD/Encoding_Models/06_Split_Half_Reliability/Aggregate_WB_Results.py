"""
This script aggregates the split-half correlation results for each subject and hemisphere into a single file.
The final files are vertex-wise correlation time courses for each hemisphere, with the shape (n_shuffles, 2, n_time, 163842)
"""

import numpy as np
import os
import time
from tqdm import tqdm

# Start time
start_time = time.time()

subject_list = [1, 4, 5, 6, 7, 8]   # List of subjects to process
n_splits = 83                        # Number of fmri_split chunk files (163842 / 1974 = 83)

# /scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion_split_half_reliability/whole_brain/subject-1/hemisphere-lh


for subject in tqdm(subject_list):

    path_l = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion_split_half_reliability/whole_brain/subject-{subject}/hemisphere-lh'
    path_r = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion_split_half_reliability/whole_brain/subject-{subject}/hemisphere-rh'

    chunks_l = []
    for i in range(1, n_splits + 1):
        splits_path = path_l + f'/fmri_split-{i:02d}'  # iterating over the split files
        corrs = np.load(splits_path + '.npy')
        chunks_l.append(corrs)
    data_l = np.concatenate(chunks_l, axis=-1, dtype=np.float32)  # (n_shuffles, 2, n_time, 163842)

    chunks_r = []
    for i in range(1, n_splits + 1):
        splits_path = path_r + f'/fmri_split-{i:02d}'  # iterating over the split files
        corrs = np.load(splits_path + '.npy')
        chunks_r.append(corrs)
    data_r = np.concatenate(chunks_r, axis=-1, dtype=np.float32)  # (n_shuffles, 2, n_time, 163842)

    save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/encoding_fusion_split_half_reliability/whole_brain/subject-{subject}'
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)

    file_name_l = 'correlations_left.npy'
    file_name_r = 'correlations_right.npy'

    np.save(os.path.join(save_dir, file_name_l), data_l)
    np.save(os.path.join(save_dir, file_name_r), data_r)

    print(f"Split-half correlations for subject {subject} saved!")

# End time
end_time = time.time()

print(f"Total time taken: {end_time - start_time:.2f} seconds.")
print("Execution complete!")