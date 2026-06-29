import numpy as np
import os
import time
from tqdm import tqdm


# Start time
start_time = time.time()

subject_list = [1,4,5,6,7,8]   # List of subjects to process
variance_types = ["shared_vision_language", "unique_vision", "unique_language"]
for subject in tqdm(subject_list):
    for variance_type in variance_types:
        print("Processing correlation type: ", variance_type)
        

        path_l = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/variance_partitioning/wb/subject-{subject}/hemisphere-lh'


        data_l = [[] for _ in range(359)] # This list will contain 359 lists of 163,842 correlations each
        for i in range(1,22):
            splits_path = path_l+f'/fmri_split-{i}' #iterating over the split folders
            corrs = np.load(splits_path+'.npy', allow_pickle=True).item()[variance_type]
            for j in range(len(corrs)):
                data_l[j].extend(corrs[j]) # filling out

        save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/variance_partitioning/wb/subject-{subject}'
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)

        file_name_l = f'{variance_type}_left.npy'

        left_data = np.array(data_l)
        np.save(os.path.join(save_dir, file_name_l), left_data)


    print(f"Correlations for subject {subject} saved!")

# End time
end_time = time.time()

print(f"Total time taken: {end_time - start_time:.2f} seconds.")
print("Execution complete!")