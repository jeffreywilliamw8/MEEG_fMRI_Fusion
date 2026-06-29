import argparse
import numpy as np
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import os
import h5py
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
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--nsd_dir', default='/scratch/ccn_datasets/natural-scenes-dataset', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> Extracting Layerwise Feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 8

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Load the image presentation order and the training/testing splits
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'model_training_datasets',
    'train_dataset-nsd_fsaverage')

metadata = np.load(os.path.join(data_dir, 'metadata_subject-'+
    str(args.subject)+'.npy'), allow_pickle=True).item()

img_presentation_order = metadata['img_presentation_order']
train_img_num = metadata['train_img_num']
test_img_num = metadata['test_img_num']


# =============================================================================
# Access the NSD images
# =============================================================================
# Access the NSD-core images
sf = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli', 'nsd',
    'nsd_stimuli.hdf5'), 'r')
sdataset = sf.get('imgBrick')

# Access the NSD-synthetic images (stimuli)
stimuli = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
    'nsdsynthetic', 'nsdsynthetic_stimuli.hdf5'), 'r').get('imgBrick')[:]
# Access the NSD-synthetic images (colorstimuli)
colorstimuli = h5py.File(os.path.join(args.nsd_dir, 'nsddata_stimuli',
    'stimuli', 'nsdsynthetic','nsdsynthetic_colorstimuli_subj0'+
    str(args.subject)+'.hdf5'), 'r').get('imgBrick')[:]
images_nsdsynthetic = np.append(stimuli, colorstimuli, 0)


# ... (Keep your imports and argparsing as is)

# =============================================================================
# Vision model setup
# =============================================================================
model = torchvision.models.alexnet(weights='DEFAULT')

model_layers = [
    'features.2',    # Conv1 + Pool
    'features.5',    # Conv2 + Pool
    'features.7',    # Conv3
    'features.9',    # Conv4
    'features.12',   # Conv5 + Pool
    'classifier.2',  # FC6
    'classifier.5',  # FC7
    'classifier.6'   # FC8 (Output)
]

feature_extractor = create_feature_extractor(model, return_nodes=model_layers)
feature_extractor.to(device)
feature_extractor.eval()

transform = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()

# Initialize dictionaries to hold lists of features per layer
raw_features = {layer: {'train': [], 'test': [], 'test_ood': []} for layer in model_layers}

# =============================================================================
# Extract the image features
# =============================================================================
with torch.no_grad():
    # 1. Training images
    print("Extracting Training Features...")
    for i in tqdm(train_img_num):
        img = transform(Image.fromarray(sdataset[i]).convert('RGB')).unsqueeze(0).to(device)
        ft = feature_extractor(img)
        for layer, feat in ft.items():
            raw_features[layer]['train'].append(feat.flatten().cpu().numpy())

    # 2. Test images (NSD-core)
    print("Extracting Test Features...")
    for i in tqdm(test_img_num):
        img = transform(Image.fromarray(sdataset[i]).convert('RGB')).unsqueeze(0).to(device)
        ft = feature_extractor(img)
        for layer, feat in ft.items():
            raw_features[layer]['test'].append(feat.flatten().cpu().numpy())

    # 3. Test images (NSD-synthetic)
    print("Extracting OOD Test Features...")
    for img_arr in tqdm(images_nsdsynthetic):
        img_arr = (np.sqrt(img_arr/255)*255).astype(np.uint8)
        img = transform(Image.fromarray(img_arr).convert('RGB')).unsqueeze(0).to(device)
        ft = feature_extractor(img)
        for layer, feat in ft.items():
            raw_features[layer]['test_ood'].append(feat.flatten().cpu().numpy())

# =============================================================================
# Process Each Layer Separately (Z-score + PCA)
# =============================================================================
final_results = {}

print("\nProcessing Layers (Scaling + PCA)...")
for layer in tqdm(model_layers):
    # Convert lists to numpy arrays
    X_train = np.array(raw_features[layer]['train'])
    X_test = np.array(raw_features[layer]['test'])
    X_ood = np.array(raw_features[layer]['test_ood'])
    
    # 1. Z-scoring (StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_ood = scaler.transform(X_ood)
    
    # 2. PCA
    pca = PCA(n_components=250, random_state=seed)
    X_train = pca.fit_transform(X_train).astype(np.float32)
    X_test = pca.transform(X_test).astype(np.float32)
    X_ood = pca.transform(X_ood).astype(np.float32)
    
    # Store in final structure
    final_results[layer] = {
        'train': X_train,
        'test': X_test,
        'test_ood': X_ood
    }

# =============================================================================
# Saving the results
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/{args.model}'
os.makedirs(save_dir, exist_ok=True)

# Save as a single .npy file containing the dictionary of all layers
filename = f"sub-{args.subject:02d}_layerwise_fmaps.npy"
np.save(os.path.join(save_dir, filename), final_results)

# End time
end_time = time.time()
execution_time = end_time - start_time
print("Layerwise extraction complete!")
print(f"Execution time: {execution_time:.2f} seconds.")