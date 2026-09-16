"""Extract and save layer-wise visual features from the EEG Moments videos using r3d_18, a 3D
ResNet-18 trained on Kinetics-400 action recognition, applied as a genuine spatiotemporal model.
Each video's sampled frames are passed through r3d_18 jointly as ONE clip (N, C, T, H, W), so
every layer's feature map already integrates information across frames -- there is no per-frame
averaging step here (unlike the frame-wise AlexNet extraction), because the model itself fuses
time within each 3D convolution.

https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html

"""

import os
import argparse

import numpy as np
import torch
import torchvision
from torchvision.io.video import read_video
from torch.utils.data import Dataset, DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='r3d_18', type=str)
parser.add_argument('--num_frames', default=16, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Extract vision features -- r3d_18, spatiotemporal <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =============================================================================
# GPU and workers
# =============================================================================
# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# Load the r3d_18 model and video transform
# =============================================================================
model = torchvision.models.video.r3d_18(weights='KINETICS400_V1')
# The Kinetics-400 video classification preset (resize -> center crop 112 -> to float ->
# normalize -> permute (T,C,H,W) -> (C,T,H,W)). It expects one whole (T, C, H, W) clip, unlike
# the AlexNet image preset which broadcasts frame-wise over a leading dimension.
transform = torchvision.models.video.R3D_18_Weights.KINETICS400_V1.transforms()

# =============================================================================
# Define the layers from which to extract features
# =============================================================================
# r3d_18's own stages, in order of increasing spatiotemporal abstraction: the stem (first joint
# space-time conv), the four residual stages (Layer1-4, each roughly doubling channels and --
# from Layer2 on -- halving the spatial AND temporal extent), the final global-average-pooled
# representation (AvgPool), and the 400-way Kinetics action logits (FC), the task-driven semantic
# analogue of AlexNet's FC8.
fe_nodes = {
    'stem': 'Stem',
    'layer1': 'Layer1',
    'layer2': 'Layer2',
    'layer3': 'Layer3',
    'layer4': 'Layer4',
    'avgpool': 'AvgPool',
    'fc': 'FC',
}
layer_names = list(fe_nodes.values())

# =============================================================================
# Video dataset class
# =============================================================================
class VideoDataset(Dataset):
    def __init__(self, video_dir, num_samples, device, transform=None):
        self.video_dir = video_dir
        self.video_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4'))])
        assert len(self.video_files) == 1102
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def sample_frames(self, video_frames, num_samples):
        num_frames = video_frames.shape[0]
        if num_samples > num_frames:
            raise ValueError("The number of samples requested is greater than the number of frames in the video.")
        indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)
        sampled_frames = video_frames[indices]
        return sampled_frames

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_frames, _, _ = read_video(video_path, pts_unit='sec',
            output_format='TCHW')
        try:
            sampled_frames = self.sample_frames(video_frames, self.num_samples)
        except ValueError:
            last_frame = video_frames[-1].unsqueeze(0).repeat(
                self.num_samples - video_frames.shape[0], 1, 1, 1)
            sampled_frames = torch.cat([video_frames, last_frame], dim=0)
        if self.transform:
            # (T, C, H, W) -> (C, T, H, W), the whole clip transformed/normalized jointly.
            sampled_frames = self.transform(sampled_frames)
            sampled_frames = sampled_frames.to(device)
        return idx, sampled_frames


# =============================================================================
# Create the dataset and dataloader
# =============================================================================
video_dir = os.path.join(args.emd_dir, 'stimuli', 'mp4_h264')

# Create the dataset
dataset = VideoDataset(video_dir=video_dir, num_samples=args.num_frames,
    device=device, transform=transform)

# Create a DataLoader without shuffling
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
    pin_memory=False)  # num_workers=num_workers

# =============================================================================
# Create the feature extractor
# =============================================================================
feature_extractor = create_feature_extractor(model, fe_nodes)
# Set the model in evaluation mode, on the current device
feature_extractor.eval()
feature_extractor.to(device)

# =============================================================================
# Extract the vision features -- one clip per video, so one feature map per video per layer
# =============================================================================
features = {layer: [] for layer in layer_names}

with torch.no_grad():
    for indices, batch in tqdm(dataloader):

        # batch: (n_videos, C, T, H, W) 
        ft_batch = feature_extractor(batch)

        for layer in layer_names:
            act = ft_batch[layer]                         # (n_videos, C, [T, H, W])
            act = act.reshape(act.shape[0], -1)            # flatten -> (n_videos, n_features)
            features[layer].append(act.cpu().numpy().astype(np.float32))

        del ft_batch, batch
        print('batch {} - {} / 1101 done'.format(indices[0].item(), indices[-1].item()))

for layer in layer_names:
    features[layer] = np.concatenate(features[layer], axis=0).astype(np.float32)
    print(f"{layer}: extracted features {features[layer].shape} (videos, features)")


# =============================================================================
# PCA functions
# =============================================================================
def fit_pca_float32(X_train):
    """Fit PCA in sample space on float32 data.
    Assumes X_train is already z-scored so no centering needed."""
    # Covariance in sample space: (n_samples, n_samples)
    cov = (X_train @ X_train.T) / (X_train.shape[0] - 1)     # (1000, 1000), float32
    # Eigen-decomposition — eigh returns ascending order
    eigenvalues, eigenvectors = eigh(cov)
    # Reverse to descending order
    eigenvalues  = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    # Clip tiny negative eigenvalues caused by float32 numerical noise
    eigenvalues = np.maximum(eigenvalues, 0)
    # Principal axes in feature space: (n_features, n_samples)
    principal_axes = X_train.T @ eigenvectors                  # (n_features, n_samples)
    principal_axes /= np.linalg.norm(principal_axes, axis=0)   # normalize → V
    # Explained variance ratio (after clipping)
    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    return principal_axes, explained_variance_ratio


def transform_pca_float32(X, principal_axes, n_components):
    """Project data onto the top n_components principal axes."""
    return X @ principal_axes[:, :n_components]               # (n_samples, n_components)


# =============================================================================
# Downsample the vision features using PCA, and save -- separately per layer
# =============================================================================
save_dir = os.path.join('/scratch/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models', 'results', 'stimulus_features',
    'layerwise_vision_features', args.model_name)
os.makedirs(save_dir, exist_ok=True)

for layer in layer_names:

    print(f"\n>>> {layer} <<<")

    # Divide the 1102 videos into the 1000 training videos and 102 test videos
    features_train = features[layer][:1000].astype(np.float32)
    features_test = features[layer][1000:].astype(np.float32)
    del features[layer]

    # Z-score the features
    scaler = StandardScaler()
    scaler.fit(features_train)
    features_train = scaler.transform(features_train)
    features_test = scaler.transform(features_test)

    # Fit the PCA on the train data
    principal_axes, explained_variance_ratio = fit_pca_float32(features_train)

    # Find the number of components explaining 95% variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
    print(f"Components explaining 95% variance: {n_components_95}")

    # Transform train and test
    features_train = transform_pca_float32(features_train, principal_axes,
        n_components_95)
    features_test  = transform_pca_float32(features_test,  principal_axes,
        n_components_95)

    file_name_train = f'vision_features_train_layer-{layer}.npy'
    file_name_test = f'vision_features_test_layer-{layer}.npy'
    np.save(os.path.join(save_dir, file_name_train), features_train.astype(np.float32))
    np.save(os.path.join(save_dir, file_name_test), features_test.astype(np.float32))
    print(f"Saved: {file_name_train} {features_train.shape}, "
          f"{file_name_test} {features_test.shape}")

    del features_train, features_test, principal_axes

print(f"\nAll layers saved to: {save_dir}")