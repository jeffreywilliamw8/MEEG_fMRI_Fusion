"""
This script extracts image features from each
of the 8 AlexNet layers

"""

import os
import numpy as np
import torch
import torchextractor as tx
from tqdm import tqdm
from torchvision.models import alexnet, AlexNet_Weights
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gc
from PIL import Image
import time
from berg import BERG
import pandas as pd

# Start time
start_time = time.time()
metadata_df = pd.read_csv("/scratch/jeffreykatab/Code/Encoding_Models/THINGS/fMRI/betas_csv/sub-01_StimulusMetadata.csv")
############################################################################
# Core Functions
############################################################################

def load_alexnet_extractor(device):
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    model.to(device)
    model.eval()

    layer_names = {
        'features.2': 'conv1',    # Conv1 + Pool
        'features.5': 'conv2',    # Conv2 + Pool
        'features.7': 'conv3',    # Conv3
        'features.9': 'conv4',    # Conv4
        'features.12': 'conv5',   # Conv5 + Pool
        'classifier.2': 'fc6',    # FC6
        'classifier.5': 'fc7',    # FC7
        'classifier.6': 'fc8'     # FC8 (Output)
    }

    feature_extractor = tx.Extractor(model, list(layer_names.keys()))
    transform = weights.transforms()
    return feature_extractor, transform, layer_names

def extract_layerwise_features(stimuli_array, feature_extractor, transform, device, layer_map):
    images = transform(torch.from_numpy(stimuli_array))
    batch_size = 50 # Smaller batch for AlexNet spatial layers
    n_batches = int(np.ceil(len(images) / batch_size))

    # Initialize dictionary to hold lists of batches for each layer
    layer_data = {alias: [] for alias in layer_map.values()}

    with torch.no_grad():
        for b in tqdm(range(n_batches), desc="Extracting Layers"):
            idx_start = b * batch_size
            idx_end = min(idx_start + batch_size, len(images))
            img_batch = images[idx_start:idx_end].to(device)

            _, features = feature_extractor(img_batch)

            for internal_name, alias in layer_map.items():
                # Flatten spatial dimensions: (Batch, C, H, W) -> (Batch, C*H*W)
                flat = torch.flatten(features[internal_name], 1).cpu().numpy()
                layer_data[alias].append(flat)

    # Stack all batches for each layer
    for alias in layer_data:
        layer_data[alias] = np.vstack(layer_data[alias])

    return layer_data


def load_images(stimuli_names, images_dir, img_size):
    print("==== Loading {} Images ====".format(len(stimuli_names)))
    images_list = []
    for stimulus in tqdm(stimuli_names):
        concept = metadata_df[metadata_df['stimulus'] == stimulus]['concept'].values[0]
        img_path = os.path.join(images_dir, concept, stimulus)
        try:
            with Image.open(img_path).convert('RGB') as img:
                img = img.resize(img_size)
                img_array = np.array(img).transpose(2, 0, 1)
                images_list.append(img_array)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return np.stack(images_list).astype(np.uint8)

###############################################
# Configuration & Setup
###############################################

BERG_DIR = '/scratch/jeffreykatab/Code/Encoding_Models/brain-encoding-response-generator'
FMRI_SUBJECT = 1

berg = BERG(berg_dir=BERG_DIR)
metadata_fmri = berg.get_model_metadata('fmri-things_fmri_1-vit_b_32', subject=FMRI_SUBJECT)

train_stimuli_names = metadata_fmri['encoding_model']['train_stimuli']
test_stimuli_names = metadata_fmri['encoding_model']['test_stimuli']
test_stimuli_names = np.unique(test_stimuli_names)

SAVE_DIR = "/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/visual/alexnet_layerwise"
IMAGES_DIR = "/scratch/jeffreykatab/Code/Encoding_Models/THINGS/Images_and_Metadata/object_images"
IMG_SIZE = (224, 224)
N_PCS = 250 # Number of components per layer

os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor, transform, layer_map = load_alexnet_extractor(device)

###############################################
# Execution
###############################################

print("--- Phase 1: Image Loading ---")
train_stimulus = load_images(train_stimuli_names, IMAGES_DIR, IMG_SIZE)
test_stimulus = load_images(test_stimuli_names, IMAGES_DIR, IMG_SIZE)

print("\n--- Phase 2: Layer-wise Extraction ---")
train_layers_raw = extract_layerwise_features(train_stimulus, feature_extractor, transform, device, layer_map)
test_layers_raw = extract_layerwise_features(test_stimulus, feature_extractor, transform, device, layer_map)

# Clear memory of raw images
del train_stimulus, test_stimulus
gc.collect()

print("\n--- Phase 3: Dimensionality Reduction (PCA per Layer) ---")
final_features = {'train': {}, 'test': {}}

for layer_alias in layer_map.values():
    print(f"Processing {layer_alias}...")

    # Scale
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_layers_raw[layer_alias])
    test_scaled = scaler.transform(test_layers_raw[layer_alias])

    # PCA transformation
    pca = PCA(n_components=N_PCS)
    final_features['train'][layer_alias] = pca.fit_transform(train_scaled)
    final_features['test'][layer_alias] = pca.transform(test_scaled)

    # Clean up raw layer data to save RAM
    del train_layers_raw[layer_alias], test_layers_raw[layer_alias]
    gc.collect()

# Save output
output_dict = {
    'features': final_features, # Dictionary: ['train'/'test']['conv1'...'fc7']
    'train_stimuli': train_stimuli_names,
    'test_stimuli': test_stimuli_names,
    'layer_names': list(layer_map.values())
}

save_path = os.path.join(SAVE_DIR, f"alexnet_layerwise_features_{N_PCS}_pcs.npy")
np.save(save_path, output_dict)

print(f"\n Layer-wise features saved to {save_path}")
print(f"Total time: {time.time() - start_time:.2f} seconds.")