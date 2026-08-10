import numpy as np
import os
import time
from tqdm import tqdm

# Start time
start_time = time.time()

subject_list = [1, 4, 5, 6, 7, 8]  # List of subjects to process

alexnet_layers = [
    'features.2',    # Conv1 + Pool
    'features.5',    # Conv2 + Pool
    'features.7',    # Conv3
    'features.9',    # Conv4
    'features.12',   # Conv5 + Pool
    'classifier.2',  # FC6
    'classifier.5',  # FC7
    'classifier.6'   # FC8 (Output)
]

for subject in tqdm(subject_list):

    for layer in alexnet_layers:
        print("Processing AlexNet layer: ", layer)

        path_l = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/commonality_analysis/layerwise_alexnet/wb/subject-{subject}/layer-{layer}/hemisphere-lh'
        path_r = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/commonality_analysis/layerwise_alexnet/wb/subject-{subject}/layer-{layer}/hemisphere-rh'

        # Each split file is already a clean (359, n_vertices_in_split) matrix -- just
        # concatenate along the vertex axis across the 21 splits, no ragged-list rebuilding needed.
        splits_l = []
        for i in range(1, 22):
            splits_path = os.path.join(path_l, f'fmri_split-{i}.npy')
            corrs = np.load(splits_path, allow_pickle=True)
            splits_l.append(corrs)
        left_data = np.concatenate(splits_l, axis=1)  # (359, 163842)

        splits_r = []
        for i in range(1, 22):
            splits_path = os.path.join(path_r, f'fmri_split-{i}.npy')
            corrs = np.load(splits_path, allow_pickle=True)
            splits_r.append(corrs)
        right_data = np.concatenate(splits_r, axis=1)  # (359, 163842)

        save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/commonality_analysis/layerwise_alexnet/wb/subject-{subject}/layer-{layer}'
        os.makedirs(save_dir, exist_ok=True)

        file_name_l = 'correlations_left.npy'
        file_name_r = 'correlations_right.npy'

        np.save(os.path.join(save_dir, file_name_l), left_data)
        np.save(os.path.join(save_dir, file_name_r), right_data)

        print(f"Correlations for subject {subject}, layer {layer} saved! Shapes: {left_data.shape}, {right_data.shape}")

# End time
end_time = time.time()

print(f"Total time taken: {end_time - start_time:.2f} seconds.")
print("Execution complete!")