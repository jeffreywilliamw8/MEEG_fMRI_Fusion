import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from utils import get_eeg_times
import time

# Start time
start_time = time.time()


# =============================================================================
# Setup Directories and Meta Parameters
# =============================================================================
subject_list = [1,4,5,6,7,8]
streams = ['early', 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal']
stream_labels = ['Early', 'Midventral', 'Midlateral', 'Midparietal', 'Ventral', 'Lateral', 'Parietal']
n_streams = len(stream_labels)

PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Isolate post-stimulus window only
times = get_eeg_times()
post_stim_idx = np.where(times >= 0)

# =============================================================================
# Run Extraction and Compute Similarity Matrices
# =============================================================================
print(">>> Extracting Post-Stimulus Curves for Encoding Models and RSA <<<")

em_curves_dict = np.load('/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/plots/em_stream_curves.npy', allow_pickle=True).item()
rsa_curves_dict = np.load('/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots/rsa_stream_curves.npy', allow_pickle=True).item()
em_curves = []
rsa_curves = []
for stream in streams:
    em_curve = np.squeeze(np.mean(em_curves_dict[stream][:, post_stim_idx], axis=0))
    rsa_curve = np.squeeze(np.mean(rsa_curves_dict[stream][:, post_stim_idx], axis=0))
    print("Shape of the curves: (encoding, rsa):", em_curve.shape, rsa_curve.shape)
    em_curves.append(em_curve)
    rsa_curves.append(rsa_curve)



# Initialize blank similarity matrix arrays
em_matrix = np.zeros((n_streams, n_streams))
rsa_matrix = np.zeros((n_streams, n_streams))

print(">>> Computing Between-stream Similarity Matrices (Spearman's R) <<<")
for i in range(n_streams):
    for j in range(n_streams):
        em_matrix[i, j] = spearmanr(em_curves[i], em_curves[j]).correlation
        rsa_matrix[i, j] = spearmanr(rsa_curves[i], rsa_curves[j]).correlation

# =============================================================================
# Statistically Correlate the Two Matrices 
# =============================================================================
# Extract indices for the strict upper triangle (excluding the 1.0 diagonal entries)
upper_tri_indices = np.triu_indices(n_streams, k=1)

em_vector = em_matrix[upper_tri_indices]
rsa_vector = rsa_matrix[upper_tri_indices]

# Compute second-order Spearman rank correlation between the profiles
third_order_r, p_value = spearmanr(em_vector, rsa_vector)
print(f"Third-order Spearman correlation between EM and RSA matrices: R = {third_order_r:.4f}")

# =============================================================================
# Plotting the Side-by-Side Similarity Matrices
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# Determine uniform color bounds for the heatmaps
vmin_val = min(em_matrix.min(), rsa_matrix.min())
vmax_val = max(em_matrix.max(), rsa_matrix.max())

# Plot 1: Encoding Matrix
sns.heatmap(
    em_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="viridis", 
    xticklabels=stream_labels, 
    yticklabels=stream_labels,
    vmin=vmin_val, 
    vmax=vmax_val, 
    cbar=False, 
    ax=axes[0],
    square=True
)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, ha="right")
axes[0].set_title("Encoding Similarity Matrix", fontsize=22, pad=12)
axes[0].tick_params(axis='both', labelsize=18)


# Plot 2: RSA Matrix
sns.heatmap(
    rsa_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="viridis", 
    xticklabels=stream_labels, 
    yticklabels=stream_labels,
    vmin=vmin_val, 
    vmax=vmax_val, 
    cbar=True, 
    ax=axes[1],
    square=True,
    cbar_kws={'label': "Spearman's R"}
)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, ha="right")
axes[1].set_title("RSA Similarity Matrix", fontsize=22, pad=12)
axes[1].tick_params(axis='both', labelsize=22)

# Add overarching statistical metric to the figure
plt.suptitle(
    #f"Correlation of Similarity Matrices: Spearman's R = {third_order_r:.3f} (p = {p_value:.1e})", 
    f"Correlation of Similarity Matrices: Spearman's R = {third_order_r:.3f}",
    fontsize=22, 
    y=1.02
)

plt.tight_layout()

# Save final matrix asset
output_filename = 'em_vs_rsa_stream_similarity_matrices.svg'
plt.savefig(os.path.join(PLOTS_DIR, output_filename), dpi=300, bbox_inches='tight')
print(f"plot saved to: {os.path.join(PLOTS_DIR, output_filename)}")
plt.close()

print(f"Execution complete! Total Time: {time.time() - start_time:.2f} seconds.")