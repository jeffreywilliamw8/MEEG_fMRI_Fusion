#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=compute_decoding_accuracy_eeg_rdms
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=06:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a eeg_subject_all
index=0
for s in "01" "02" "03" "04" "05" "06" "avg" "app"; do
    eeg_subject_all[$index]=$s
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
eeg_subject=${eeg_subject_all[$SLURM_ARRAY_TASK_ID]}
echo eeg_subject: $eeg_subject

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/code/01_Preparing_Data

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 01c_Compute_Decoding_Accuracy_EEG_RDMs.py --eeg_subject $eeg_subject