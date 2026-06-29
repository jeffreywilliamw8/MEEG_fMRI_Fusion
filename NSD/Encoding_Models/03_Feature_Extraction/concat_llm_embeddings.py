import numpy as np
import os

# Configuration
input_dir = '/scratch/singej96/universality_LLM/results/embeddings'
output_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/Encoding_Models/stimulus_features'
base_name = 'Qwen3_train_embeddings_chunk'
n_chunks = 10

# Construct file paths using string formatting to match '0000' through '0009'
file_paths = [os.path.join(input_dir, f"{base_name}{i:04d}.npy") for i in range(n_chunks)]

print(f"Loading and concatenating {n_chunks} chunks...")

# Load all arrays into a list and concatenate along axis 0 (stimulus dimension)
# np.vstack is a shorthand for concatenating along the first axis
final_array = np.vstack([np.load(f) for f in file_paths])

# Save the result
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'Qwen3_train_embeddings_full.npy')
np.save(save_path, final_array)

print(f"Mission accomplished! Final shape: {final_array.shape}")
print(f"Saved to: {save_path}")