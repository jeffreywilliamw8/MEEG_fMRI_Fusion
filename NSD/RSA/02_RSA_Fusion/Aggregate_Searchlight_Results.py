import numpy as np
import os
from tqdm import tqdm
import argparse
import time

# Start time
start_time = time.time()

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--n_neighbours', type=int, default=100) 
args = parser.parse_args()


print('>>> Searchlight RSA Results Aggregation <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/searchlight_fusion/n_neighbours-{args.n_neighbours}/metric_correlation'
OUTPUT_DIR = f'{BASE_DIR}/aggregated_results'
N_SUBJECTS = 3
N_TIMEPOINTS = 359
HEMISPHERES = ['lh_hemisphere', 'rh_hemisphere']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Aggregation Loop
# =============================================================================
print(">>> Starting RSA Fusion Aggregation <<<")

for sub_id in tqdm([4, 5, 7]):
    sub_name = f"subject-{sub_id}"
    print(f"\nProcessing {sub_name}...")
    
    for hemi in HEMISPHERES:
        hemi_path = os.path.join(BASE_DIR, sub_name, hemi)
        
        time_course_list = []
        
        for t in range(N_TIMEPOINTS):
            file_name = f"time_point_{t:04d}.npy"
            file_path = os.path.join(hemi_path, file_name)
            
            try:
                # Load the correlation coefficients (163842,)
                data = np.load(file_path)
                time_course_list.append(data)
            except FileNotFoundError:
                print(f"Warning: {file_path} not found. Skipping...")
                continue

        # Convert list to a single array of shape (359, 163842)
        if time_course_list:
            final_array = np.stack(time_course_list, axis=0)
            print(f"  Aggregated array shape for {hemi}: {final_array.shape}")
            
            # Create subject-specific output folder
            sub_output_dir = os.path.join(OUTPUT_DIR, sub_name)
            os.makedirs(sub_output_dir, exist_ok=True)
            
            # Save the aggregated time course
            save_name = f"{sub_name}_{hemi}_timecourse.npy"
            np.save(os.path.join(sub_output_dir, save_name), final_array)
            
            print(f"  Saved {hemi} aggregated array with shape: {final_array.shape}")
        else:
            print(f"  Error: No data found for {sub_name} {hemi}")

print("\n>>> All subjects aggregated. <<<")


# End time
execution_time = time.time() - start_time
print(f"Total Execution time: {execution_time:.2f} seconds.")