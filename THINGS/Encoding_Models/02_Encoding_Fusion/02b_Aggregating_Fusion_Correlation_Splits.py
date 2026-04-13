import numpy as np
import os
import random
import time
from tqdm import tqdm
import os

# Start time
start_time = time.time()

# Random seed for reproducibility

seed = 8
np.random.seed(seed)
random.seed(seed)


subject_list = ['01', '02', '03']   # List of subjects to process
subject_ranges = {
    '01': 533,
    '02': 534,
    '03': 524
} # Number of fMRI splits per subject


for subject in tqdm(subject_list):

    corrs_dir = f'/home/jeffreykatabo/Projects/fusion/THINGS/Encoding_Models/results/correlations/encoding_fusion_wb/fmri_sub-{subject}'

    data = [[] for _ in range(181)] # Will be of shape (181 time points, n_voxels)
    for i in range(1,subject_ranges[subject]+1):
        corrs = np.load(os.path.join(corrs_dir, f'fmri_split-{i:03d}.npy'))
        for j in range(len(corrs)):
            data[j].extend(list(corrs[j])) # filling out
    data = np.array(data, dtype=np.float32)
    save_path = '/home/jeffreykatabo/Projects/fusion/THINGS/Encoding_Models/results/correlations/encoding_fusion_wb'
    np.save(os.path.join(save_path, f'sub-{subject}_correlation_time_courses.npy'), data)
    print("Shape of correlations for subject {}: {}".format(subject, data.shape))


# End time
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds.")
print("Execution complete!")