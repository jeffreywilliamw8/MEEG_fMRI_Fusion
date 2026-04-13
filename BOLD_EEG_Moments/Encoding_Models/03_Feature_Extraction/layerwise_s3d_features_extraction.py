"""
Extracts, normalizes, PCA-reduces, and saves S3D layerwise features.
PCA is set to 1000 components.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import time

# Start time
start_time = time.time()

#==============================
# settings
#==============================
N_TRAIN = 1000
N_TEST = 102
TOTAL = N_TRAIN + N_TEST
N_COMPONENTS = 1000
LAYERS_TO_USE = ['layer2', 'layer5', 'layer7', 'layer9', 'layer11', 'layer13']

# Paths
features_dir = '/scratch/giffordale95/projects/eeg_moments/results/stimulus_features/full_model_features/modality-visual/model-s3d' 
out_dir = '/home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/features/layerwise_s3d_features_pca'
os.makedirs(out_dir, exist_ok=True)

#==============================
# Helper functions
#==============================
def load_features(video_idx):
    """Load a single video feature dict."""
    fname = os.path.join(features_dir, f'stimulus_features_video-{video_idx:04d}.npy')
    try:
        return np.load(fname, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Warning: {fname} not found.")
        return None

def extract_layer_features(layer_name, start_idx, end_idx):
    """Extracts flattened features for one layer across specified video range."""
    feats = []
    for vid in range(start_idx, end_idx + 1):
        feat_dict = load_features(vid)
        if feat_dict is not None:
            arr = feat_dict[layer_name]  # access layer activations
            feats.append(arr.flatten())
        if vid % 200 == 0:
            print(f"   - Processed video {vid}/{end_idx} for {layer_name}")
    return np.array(feats)

#==============================
# Main processing loop
#==============================
for layer in LAYERS_TO_USE:
    print(f"\n=== Processing {layer} ===")

    # 1. Extract raw features
    print(f"Extracting features...")
    train_feats = extract_layer_features(layer_name=layer, start_idx=1, end_idx=N_TRAIN)
    test_feats = extract_layer_features(layer_name=layer, start_idx=N_TRAIN + 1, end_idx=TOTAL)

    # 2. Standardize (Scaling is required before PCA)
    print(f"Standardizing...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feats)
    test_scaled = scaler.transform(test_feats)

    # 3. PCA Reduction to 1000 PCs
    print(f"Applying PCA (n_components={N_COMPONENTS})...")
    pca = PCA(n_components=N_COMPONENTS)
    
    # Fit PCA on training set only
    train_pca = pca.fit_transform(train_scaled)
    # Transform test set using the same components
    test_pca = pca.transform(test_scaled)

    print(f"Original shape: {train_feats.shape} features")
    print(f"Reduced shape:  {train_pca.shape} PCs")
    print(f"Explained Variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 4. Save everything
    np.save(os.path.join(out_dir, f'{layer}_train_pca1000.npy'), train_pca)
    np.save(os.path.join(out_dir, f'{layer}_test_pca1000.npy'), test_pca)
    np.save(os.path.join(out_dir, f'{layer}_pca_model.npy'), pca)

    print(f"✅ Saved PCA-reduced data for {layer}")

print("\nAll layers processed successfully!")

# End time
end_time = time.time()
execution_time = end_time - start_time

print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")