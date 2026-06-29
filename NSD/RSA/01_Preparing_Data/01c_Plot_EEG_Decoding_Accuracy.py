import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from utils import sign_permutation_cluster_test, get_eeg_times
import time

# Start time
start_time = time.time()

# --- Configuration ---
subject_list = [1,4,5,6,7,8]
n_pairs = 4950

# Input/Output Directories
results_dir = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/results/decoding_rdms'
PLOTS_DIR = '/scratch/jeffreykatab/Projects/fusion/NSD/RSA/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Time Vector Logic ---
times = get_eeg_times()
n_timepoints = len(times)

# Storage for subject-level timecourses: Shape (n_subjects, n_timepoints)
subject_accuracies = []

# =============================================================================
# Load and Aggregate Data Across Subjects
# =============================================================================
print(">>> Loading and Processing Image-Wise Decoding RDMs <<<")
for sub in subject_list:
    filepath = os.path.join(results_dir, f"decoding_rdm_eeg_sub-{sub}.npy")
    
    if os.path.exists(filepath):
        # Load raw data -> Shape: (n_timepoints, n_pairs) -> e.g., (359, 4950)
        rdm_data = np.load(filepath)
        
        # 1. Average across all 4950 image pairs to get 1 value per timepoint
        sub_timecourse = np.mean(rdm_data, axis=1)
        
        # 2. Subtract 0.5 to center the baseline around 0 (Chance)
        centered_timecourse = sub_timecourse - 0.5
        
        subject_accuracies.append(centered_timecourse)
    else:
        print(f" Warning: Missing decoding file for Subject {sub}. Skipping.")

# Convert to matrix -> Shape: (n_subjects, n_timepoints)
X_acc = np.array(subject_accuracies)
n_subs = X_acc.shape

print(f"Aggregated matrix ready for stats: {X_acc.shape} (Subjects, Timepoints)")

# =============================================================================
# Run Group Statistics & Plotting
# =============================================================================
m_group = np.mean(X_acc, axis=0)
s_err = sem(X_acc, axis=0)

# Cluster Permutation Test on the centered data (testing against 0 chance level)
print("\n>>> Computing Sign-Permutation Cluster Test (against 0) <<<")
res = sign_permutation_cluster_test(X_acc, n_permutations=10000)
sig_mask = np.zeros(n_timepoints, dtype=bool)
for c_idx, _, _ in res['significant_clusters']:
    sig_mask[c_idx] = True

# Plot Setup
plt.figure(figsize=(14, 8))
ax = plt.gca()

# Plot Mean Curve and SEM Ribbon
ax.plot(times, m_group, color='#365D8D', lw=3, zorder=3)
ax.fill_between(times, m_group - s_err, m_group + s_err, color='#365D8D', alpha=0.15, zorder=2)

# Plot Significance Indicators (Staggered layout bar right above the curves)
sig_y_level = 0.22  # Adjusted height for the decoding range
if np.any(sig_mask):
    ax.scatter(times[sig_mask], [sig_y_level] * np.sum(sig_mask), 
               color='#365D8D', s=10, marker='s', alpha=0.8, edgecolors='none', 
               label='p < 0.05 (Cluster-Corrected)')

# Neuroimaging plot aesthetics
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.axhline(0, color='black', lw=1.2, alpha=0.6, linestyle='-')  # Chance line now at 0

# Custom Aesthetic Styling
ax.set_title('Image-Wise EEG Decoding Accuracy', fontweight='bold', fontsize=18, pad=30)
ax.set_xlabel('Time (ms)', fontsize=26)
ax.set_ylabel('Accuracy minus Chance', fontsize=26)

ax.set_xlim(-100, 600)
ax.set_ylim(bottom=-0.01, top=0.25)  # Focuses window tightly on above-chance boundaries

ax.legend(loc='upper right', frameon=False, fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=22)

plt.tight_layout()

# Save final asset
save_path = os.path.join(PLOTS_DIR, 'eeg_decoding_accuracy.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSuccess! Plot saved to: {save_path}")
print(f"Total processing time: {time.time() - start_time:.2f} seconds.")