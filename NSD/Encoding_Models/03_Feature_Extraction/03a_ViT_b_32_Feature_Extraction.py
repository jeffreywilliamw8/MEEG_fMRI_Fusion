import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms as trn
from torchvision.models.feature_extraction import create_feature_extractor
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
parser.add_argument('--model', type=str, default='vit_b_32')
parser.add_argument('--nsd_dir', default='/scratch/ccn_datasets/natural-scenes-dataset', type=str)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
args, unknown = parser.parse_known_args()

print('>>> ViT B 32 Feature Extraction <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 8

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


# =============================================================================
# Vision model
# =============================================================================
# Load the model
model = torchvision.models.vit_b_32(weights='DEFAULT')

# Select the used layers for feature extraction
#nodes, _ = get_graph_node_names(model)
model_layers = [
    'encoder.layers.encoder_layer_0.add_1',
    'encoder.layers.encoder_layer_1.add_1',
    'encoder.layers.encoder_layer_2.add_1',
    'encoder.layers.encoder_layer_3.add_1',
    'encoder.layers.encoder_layer_4.add_1',
    'encoder.layers.encoder_layer_5.add_1',
    'encoder.layers.encoder_layer_6.add_1',
    'encoder.layers.encoder_layer_7.add_1',
    'encoder.layers.encoder_layer_8.add_1',
    'encoder.layers.encoder_layer_9.add_1',
    'encoder.layers.encoder_layer_10.add_1',
    'encoder.layers.encoder_layer_11.add_1'
    ]

feature_extractor = create_feature_extractor(model, return_nodes=model_layers)
feature_extractor.to(device)
feature_extractor.eval()


# =============================================================================
# Defining the image preprocessing
# =============================================================================
transform = trn.Compose([
    trn.Lambda(lambda img: trn.CenterCrop(min(img.size))(img)),
    trn.Resize((224,224)),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# Extracting the image features
# =============================================================================
# Training images
fmaps_train = []
with torch.no_grad():
    for i in tqdm(train_img_num, leave=False):
        # Preprocess the images
        img = sdataset[i]
        img = Image.fromarray(img).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        fmaps_train.append(np.squeeze(ft.detach().cpu().numpy()))
        del ft
fmaps_train = np.asarray(fmaps_train)

# Test images (NSD-core)
fmaps_test = []
with torch.no_grad():
    for i in tqdm(test_img_num, leave=False):
        # Preprocess the images
        img = sdataset[i]
        img = Image.fromarray(img).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        fmaps_test.append(np.squeeze(ft.detach().cpu().numpy()))
        del ft
fmaps_test = np.asarray(fmaps_test)

# Test images (NSD-synthetic)
fmaps_test_ood = []
with torch.no_grad():
    for img in tqdm(images_nsdsynthetic, leave=False):
        # Preprocess the images
        img = (np.sqrt(img/255)*255).astype(np.uint8)
        img = Image.fromarray(img).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        fmaps_test_ood.append(np.squeeze(ft.detach().cpu().numpy()))
        del ft
fmaps_test_ood = np.asarray(fmaps_test_ood)


# =============================================================================
# Downsampling the features using PCA
# =============================================================================
# Standardizing the features
scaler = StandardScaler()
scaler.fit(fmaps_train)
fmaps_train = scaler.transform(fmaps_train)
fmaps_test = scaler.transform(fmaps_test)
fmaps_test_ood = scaler.transform(fmaps_test_ood)

# Applying PCA
pca = PCA(n_components=250, random_state=seed)
pca.fit(fmaps_train)
fmaps_train = pca.transform(fmaps_train)
fmaps_train = fmaps_train.astype(np.float32)

fmaps_test = pca.transform(fmaps_test)
fmaps_test = fmaps_test.astype(np.float32)

fmaps_test_ood = pca.transform(fmaps_test_ood)
fmaps_test_ood = fmaps_test_ood.astype(np.float32)


# =============================================================================
# Saving the feature maps
# =============================================================================
save_dir = f'/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features/vision_models/{args.model}'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
data_dict = {
    'fmaps_train': fmaps_train,
    'fmaps_test': fmaps_test,
    'fmaps_test_ood': fmaps_test_ood
}
filename = f"fmri_sub-{args.subject:02d}_fmaps.npy"
np.save(os.path.join(save_dir, filename), data_dict)

# End time
end_time = time.time()
execution_time = end_time - start_time
print("Execution complete!")
print(f"Execution time: {execution_time:.2f} seconds.")