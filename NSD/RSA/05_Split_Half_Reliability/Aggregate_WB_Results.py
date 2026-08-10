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

print('>>> Searchlight RSA Split-Half Reliability Aggregation <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = f'/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/correlations/split_half_reliability/n_neighbours-{args.n_neighbours}'
OUTPUT_DIR = f'{BASE_DIR}/aggregated_results'
SUBJECTS = [1, 4, 5]
N_TIMEPOINTS = 359
HEMISPHERES = ['lh_hemisphere', 'rh_hemisphere']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Aggregation Loop
# =============================================================================
print(">>> Starting Searchlight RSA Split-Half Reliability Aggregation <<<")

for sub_id in tqdm(SUBJECTS):
    sub_name = f"subject-{sub_id}"
    print(f"\nProcessing {sub_name}...")

    for hemi in HEMISPHERES:
        hemi_path = os.path.join(BASE_DIR, sub_name, hemi)

        time_course_list = []

        for t in range(N_TIMEPOINTS):
            file_name = f"time_point_{t:04d}.npy"
            file_path = os.path.join(hemi_path, file_name)

            try:
                # Each file: (n_shuffles, 2, n_vertices) -- shuffles x halves x vertices, unlike
                # the plain (n_vertices,) per-timepoint file in the non-split-half version.
                data = np.load(file_path)
                time_course_list.append(data)
            except FileNotFoundError:
                print(f"Warning: {file_path} not found. Skipping...")
                continue

        # Stack across time -> (n_timepoints, n_shuffles, 2, n_vertices). Stacked ONCE at the end
        # (not extended element-by-element) since this array is one dimension deeper than the
        # non-split-half version, matching the convention used for the whole-brain encoding
        # split-half aggregation script.
        if time_course_list:
            final_array = np.stack(time_course_list, axis=0)
            print(f"  Aggregated array shape for {hemi}: {final_array.shape}")

            # Create subject-specific output folder
            sub_output_dir = os.path.join(OUTPUT_DIR, sub_name)
            os.makedirs(sub_output_dir, exist_ok=True)

            # Save the aggregated time course
            save_name = f"{sub_name}_{hemi}_split_half_timecourse.npy"
            np.save(os.path.join(sub_output_dir, save_name), final_array)

            print(f"  Saved {hemi} aggregated array with shape: {final_array.shape}")
        else:
            print(f"  Error: No data found for {sub_name} {hemi}")

print("\n>>> All subjects aggregated. <<<")

# End time
execution_time = time.time() - start_time
print(f"Total Execution time: {execution_time:.2f} seconds.")