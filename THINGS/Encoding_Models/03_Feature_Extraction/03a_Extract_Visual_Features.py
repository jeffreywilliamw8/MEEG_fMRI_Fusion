"""
Extracts visual features from THINGS images 
using the ViT B32 model
"""

import os
import numpy as np
import torch
import torchextractor as tx
from tqdm import tqdm
from torchvision.models import vit_b_32, ViT_B_32_Weights
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gc
import pandas as pd
from PIL import Image
import time
from berg import BERG
from berg_exceptions import StimulusError

# Start time
start_time = time.time()

############################################################################
# Core Functions
############################################################################

def load_feature_extractor(device):
    weights = ViT_B_32_Weights.DEFAULT
    model = vit_b_32(weights=weights)
    model.to(device)
    model.eval()
    
    layer_names = [f'encoder.layers.encoder_layer_{i}' for i in range(12)]
    feature_extractor = tx.Extractor(model, layer_names)
    transform = weights.transforms()
    return feature_extractor, transform

def extract_raw_features(stimuli_array, feature_extractor, transform, device, desc="Extracting Features"):
    if not isinstance(stimuli_array, np.ndarray) or len(stimuli_array.shape) != 4:
        raise StimulusError("Stimulus must be a 4D numpy array (batch, channels, height, width)")

    images = transform(torch.from_numpy(stimuli_array))
    batch_size = 100
    n_batches = int(np.ceil(len(images) / batch_size))
    
    all_raw_ft = []
    with torch.no_grad():
        for b in tqdm(range(n_batches), desc=desc):
            idx_start = b * batch_size
            idx_end = min(idx_start + batch_size, len(images))
            img_batch = images[idx_start:idx_end].to(device)
            _, features = feature_extractor(img_batch)
            
            batch_features = []
            for layer_name in features.keys():
                layer_flat = features[layer_name].flatten(1, 2)
                batch_features.append(layer_flat)
            
            ft = torch.cat(batch_features, dim=-1).cpu().numpy()
            all_raw_ft.append(ft)
    return np.vstack(all_raw_ft)

def load_images_berg(stimuli_names, images_dir, img_size):
    """Loads images based on filenames provided by BERG metadata."""
    images_list = []
    for stimulus in tqdm(stimuli_names, desc="Loading Images"):
        # Note: Depending on your BERG setup, 'stimulus' might be 'concept/image.jpg' 
        # or just 'image.jpg'. Here we assume standard THINGS structure.
        img_path = os.path.join(images_dir, stimulus)
        
        try:
            with Image.open(img_path).convert('RGB') as img:
                img = img.resize(img_size)
                img_array = np.array(img).transpose(2, 0, 1) 
                images_list.append(img_array)
        except Exception as e:
            print(f"Warning: Could not load {img_path}. Error: {e}")
            
    return np.stack(images_list).astype(np.uint8)

def cleanup_resources(feature_extractor):
    if feature_extractor is not None:
        model = getattr(feature_extractor, 'model', None)
        if model is not None: model.to('cpu')
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

###############################################
# Configuration & BERG Setup
###############################################

# --- BERG Initialization ---
# Assuming these arguments come from your main script/argparse
BERG_DIR = '/scratch/jeffreykatab/Code/Encoding_Models/brain-encoding-response-generator'
FMRI_SUBJECT = 1      

berg = BERG(berg_dir=BERG_DIR)
metadata_fmri = berg.get_model_metadata(
    'fmri-things_fmri_1-vit_b_32',
    subject=FMRI_SUBJECT
)

# Extract stimulus filenames directly from BERG harmonization logic
train_stimuli_names = metadata_fmri['encoding_model']['train_stimuli']
test_stimuli_names = metadata_fmri['encoding_model']['test_stimuli']

# Paths
SAVE_DIR = "/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/visual/ViT_B_32"
IMAGES_DIR = "/scratch/jeffreykatab/Code/Encoding_Models/THINGS/Images_and_Metadata/object_images"
IMG_SIZE = (224, 224)
N_PCS = 768

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor, transform = load_feature_extractor(device)

###############################################
# Execution
###############################################

print("--- Phase 0: Loading BERG Aligned Images ---")
train_stimulus = load_images_berg(train_stimuli_names, IMAGES_DIR, IMG_SIZE)
test_stimulus = load_images_berg(test_stimuli_names, IMAGES_DIR, IMG_SIZE)

print("\n--- Phase 1: Raw Feature Extraction ---")
raw_train = extract_raw_features(train_stimulus, feature_extractor, transform, device, "Train Features")
raw_test = extract_raw_features(test_stimulus, feature_extractor, transform, device, "Test Features")

print("\n--- Phase 2: Scaling and PCA ---")
scaler = StandardScaler()
scaled_train = scaler.fit_transform(raw_train)
scaled_test = scaler.transform(raw_test)

pca = PCA(n_components=N_PCS)
final_train = pca.fit_transform(scaled_train)
final_test = pca.transform(scaled_test)

# 4. Save to Disk with BERG Order Preservation
output_data = {
    'train': final_train,
    'test': final_test,
    'train_stimuli': train_stimuli_names, # Preserves the filename list
    'test_stimuli': test_stimuli_names,
    'explained_variance': pca.explained_variance_ratio_
}

save_path = os.path.join(SAVE_DIR, f"ViT_B_32_BERG_aligned_{N_PCS}_PCs.npy")
np.save(save_path, output_data)

print(f"\n[✅] Mission Accomplished. Aligned features saved to {save_path}")
cleanup_resources(feature_extractor)

print(f"Execution complete! Time: {time.time() - start_time:.2f} seconds.")