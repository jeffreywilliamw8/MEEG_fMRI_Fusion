import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms as trn
from torchvision.models.feature_extraction import create_feature_extractor
import os
import h5py
from utils import get_conditions_515
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

# Start time
start_time = time.time()


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--model', type=str, default='Qwen3')
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--nsd_dir', default='/scratch/ccn_datasets/natural-scenes-dataset', type=str)
args, unknown = parser.parse_known_args()

print(f'>>> Preparing {args.model} LLM embeddings <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 8

# =============================================================================
# Load the fMRI train/test image numbers
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-nsd_fsaverage')
meta_file_name = f'metadata_subject-{args.subject}.npy'
metadata_fmri = np.load(os.path.join(data_dir, meta_file_name),
    allow_pickle=True).item()

train_img_num = metadata_fmri['train_img_num']
train_img_num.sort()

test_img_num = metadata_fmri['test_img_num']
test_img_num.sort()

# =============================================================================
# Accessing the NSD images
# =============================================================================
# Access the NSD-core images
sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli', 'nsd', 'nsd_stimuli.hdf5'), 'r')
sdataset = sf.get('imgBrick')


# =============================================================================
# Extracting the LLM embeddings
# =============================================================================
stored_train_embeddings_file = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/{args.model}_train_embeddings_full.npy'
stored_test_embeddings_file = f'/scratch/singej96/universality_LLM/results/embeddings/{args.model}_test_embeddings.npy'
stored_train_embeddings = np.load(stored_train_embeddings_file)
stored_test_embeddings = np.load(stored_test_embeddings_file)
print(f"Loaded stored train embeddings from {stored_train_embeddings_file} with shape {stored_train_embeddings.shape}")
print(f"Loaded stored test embeddings from {stored_test_embeddings_file} with shape {stored_test_embeddings.shape}")

conditions515 = get_conditions_515(args.nsd_dir)
print("Range of conditions515:", min(conditions515), "to", max(conditions515))
conditions515 = np.array(conditions515)
conditions515 -= 1 # Converting to 0-based indexing to match the fMRI image numbers

stored_train_ids = np.setdiff1d(range(0,73000), conditions515)
print("Number of stored train embeddings:", len(stored_train_ids))
print("Range of stored train IDs:", min(stored_train_ids), "to", max(stored_train_ids))
# Training images
train_embeddings = []
for i in tqdm(train_img_num, leave=False):
    idx = np.where(stored_train_ids == i)[0]
    train_embeddings.append(stored_train_embeddings[idx][0])
embeddings_train = np.array(train_embeddings, dtype=np.float32)
print(f"Extracted train embeddings with shape {embeddings_train.shape}")

# Test images
test_embeddings = []
for i in tqdm(test_img_num, leave=False):
    idx = np.where(conditions515 == i)[0]
    test_embeddings.append(stored_test_embeddings[idx][0])
embeddings_test = np.array(test_embeddings, dtype=np.float32)
print(f"Extracted test embeddings with shape {embeddings_test.shape}")


# =============================================================================
# Downsampling the features using PCA
# =============================================================================
# Standardizing the features
scaler = StandardScaler()
scaler.fit(embeddings_train)
embeddings_train = scaler.transform(embeddings_train)
embeddings_test = scaler.transform(embeddings_test)

# Applying PCA
pca = PCA(n_components=250, random_state=seed)
pca.fit(embeddings_train)
pca_embeddings_train = pca.transform(embeddings_train)
pca_embeddings_train = pca_embeddings_train.astype(np.float32)

pca_embeddings_test = pca.transform(embeddings_test)
pca_embeddings_test = pca_embeddings_test.astype(np.float32)


# =============================================================================
# Saving the final LLM embeddings
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/LLMs/{args.model}'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
data_dict = {
    'pca_embeddings_train': pca_embeddings_train,
    'pca_embeddings_test': pca_embeddings_test,
    'embeddings_train': embeddings_train,
    'embeddings_test': embeddings_test
}
filename = f"fmri_sub-{args.subject:02d}_embeddings.npy"
np.save(os.path.join(save_dir, filename), data_dict)

# End time
end_time = time.time()
execution_time = end_time - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")