"""
Extracts language features from THINGS image captions using
sentence-transformers/all-mpnet-base-v2
"""

import json
import pandas as pd
import numpy as np
import time
import os
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # only used if you uncomment the PCA block below

# Start time
start_time = time.time()

# --- Configuration ---
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
JSON_PATH = "/scratch/jeffreykatab/Code/Encoding_Models/THINGS/Images_and_Metadata/image_descriptions_qwen_coco_style.json"
SAVE_DIR = "/scratch/jeffreykatab/Code/Encoding_Models/THINGS/features/language/image_description_embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

N_TRAIN = 8640
N_TEST = 100
BATCH_SIZE = 64

#####################################################################
# 1. Load Data
#####################################################################
print("Loading descriptions...")
with open(JSON_PATH, 'r') as f:
    descriptions_dict = json.load(f)

all_stims = list(descriptions_dict.keys())
train_stims, test_stims = all_stims[:N_TRAIN], all_stims[N_TRAIN:]

#####################################################################
# 2. Batch Encoding
#####################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing {MODEL_NAME} on {device}...")

model = SentenceTransformer(MODEL_NAME, device=device)

all_descriptions_flat = []
stim_indices = []

for stim in all_stims:
    descs = descriptions_dict[stim]
    all_descriptions_flat.extend(descs)
    stim_indices.extend([stim] * len(descs))

print(f"Encoding {len(all_descriptions_flat)} descriptions...")

embeddings_flat = model.encode(
    all_descriptions_flat,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True
)

#####################################################################
# 3. Create Consensus Embeddings
#####################################################################
print("Averaging embeddings per stimulus...")
embed_df = pd.DataFrame(embeddings_flat)
embed_df['stimulus'] = stim_indices

# Maintain "Master Order" using sort=False
consensus_embeddings = embed_df.groupby('stimulus', sort=False).mean()
consensus_embeddings = consensus_embeddings.reindex(all_stims)

X_train_raw = consensus_embeddings.iloc[:N_TRAIN].values
X_test_raw = consensus_embeddings.iloc[N_TRAIN:].values

#####################################################################
# 4. Preprocessing (Z-scoring only -- no PCA needed, see module docstring)
#####################################################################
print(f"Z-scoring {X_train_raw.shape[1]}-dim mpnet embeddings (train-set statistics)...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# --- Optional: uncomment to also save a PCA-rotated version (768 -> 768,
# a same-size decorrelating rotation, not a dimensionality reduction --
# see module docstring) for interface parity with the Qwen3 script's
# saved keys. ---
# pca = PCA(n_components=768)
# X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)

#####################################################################
# 5. Save
#####################################################################
save_file = os.path.join(SAVE_DIR, 'language_features_mpnet_base_v2.npy')
np.save(save_file, {
    'train_features': X_train_scaled,
    'test_features': X_test_scaled,
    # 'pca_train_features': X_train_pca,
    # 'pca_test_features': X_test_pca,
    'train_stimuli_names': train_stims,
    'test_stimuli_names': test_stims
})

total_time = time.time() - start_time
print(f"\nSaved to: {save_file}. Time: {total_time:.2f} seconds.")