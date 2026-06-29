import numpy as np
import os
import random
import time
from tqdm import tqdm
import argparse


# Start time
start_time = time.time()

subject_list = [1,4,5,6,7,8]   # List of subjects to process
correlation_types = ["total_correlation", "vision_partial_correlation", "language_partial_correlation"]
for subject in tqdm(subject_list):
    for correlation_type in correlation_types:
        print("Processing correlation type: ", correlation_type)
        

        path_l = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/partial_correlation/wb/subject-{subject}/hemisphere-lh'
        #f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/partial_correlation/wb/subject-{args.subject}/hemisphere-{args.hemisphere}'


        data_l = [[] for _ in [51, 77, 102, 128, 153, 179, 205, 230, 256, 281, 307, 333, 358]] # This list will contain 359 lists of 163,842 correlations each
        for i in range(1,22):
            splits_path = path_l+f'/fmri_split-{i}' #iterating over the split folders
            corrs = np.load(splits_path+'.npy', allow_pickle=True).item()[correlation_type]
            for j in range(len(corrs)):
                data_l[j].extend(corrs[j]) # filling out

        """
        data_r = [[] for _ in range(359)] # This list will contain 359 lists of 163,842 correlations each
        for i in range(1,22):
            splits_path = path_r+f'/fmri_split-{i:02d}' #iterating over the split folders
            corrs = np.load(splits_path+'.npy', allow_pickle=True)
            for j in range(len(corrs)):
                data_r[j].extend(corrs[j]) # filling out
        """
        


        save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/partial_correlation/wb/subject-{subject}'
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)

        file_name_l = f'{correlation_type}_left.npy'
        #file_name_r = 'correlations_right.npy'

        left_data = np.array(data_l)
        #right_data = np.array(data_r)
        np.save(os.path.join(save_dir, file_name_l), left_data)
        #np.save(os.path.join(save_dir, file_name_r), right_data)


    print(f"Correlations for subject {subject} saved!")

# End time
end_time = time.time()

print(f"Total time taken: {end_time - start_time:.2f} seconds.")
print("Execution complete!")