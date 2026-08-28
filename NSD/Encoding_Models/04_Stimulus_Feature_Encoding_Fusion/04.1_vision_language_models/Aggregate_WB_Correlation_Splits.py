import numpy as np
import os
import random
import time
from tqdm import tqdm
import argparse


# Start time
start_time = time.time()

subject_list = [1,4,5,6,7,8]   # List of subjects to process
#dnn_types = ['vdnn', 'llm', 'both']
dnn_types = ['vdnn', 'llm']

for subject in tqdm(subject_list):
    
    for dnn_type in dnn_types:
        print("Processing DNN: ", dnn_type)
        

        path_l =  f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/stimulus_feature_encoding_fusion/phase_2/vision_language_models/dnn_type-{dnn_type}/wb/subject-{subject}/hemi-lh'
        path_r = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/stimulus_feature_encoding_fusion/phase_2/vision_language_models/dnn_type-{dnn_type}/wb/subject-{subject}/hemi-rh'


        data_l = [[] for _ in range(359)] # This list will contain 359 lists of 163,842 correlations each
        for i in range(1,22):
            splits_path_even = path_l+f'/fmri_split-{i}_cv_split-even.npy' #iterating over the split folders
            splits_path_odd = path_l+f'/fmri_split-{i}_cv_split-odd.npy' #iterating over the split folders
            corrs = (np.load(splits_path_even) + np.load(splits_path_odd))/2.0 # Averaging across even and odd splits
            for j in range(len(corrs)):
                data_l[j].extend(corrs[j]) # filling out

        data_r = [[] for _ in range(359)] # This list will contain 359 lists of 163,842 correlations each
        for i in range(1,22):
            splits_path_even = path_r+f'/fmri_split-{i}_cv_split-even.npy' #iterating over the split folders
            splits_path_odd = path_r+f'/fmri_split-{i}_cv_split-odd.npy' #iterating over the split folders
            corrs = (np.load(splits_path_even) + np.load(splits_path_odd))/2.0 # Averaging across even and odd splits
            for j in range(len(corrs)):
                data_r[j].extend(corrs[j]) # filling out


        save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/results/correlations/jefe_phase_2/wb/vision_language_models/dnn_type-{dnn_type}/subject-{subject}'
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