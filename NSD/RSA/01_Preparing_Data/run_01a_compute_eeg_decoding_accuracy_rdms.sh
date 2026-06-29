#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=compute_eeg_decoding_accuracy_rdms
#SBATCH --mail-type=end
#SBATCH --mem=12000
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
index=0
for s in 1 4 5 6 7 8; do
    subject_all[$index]=$s
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/RSA/code/01_Preparing_Data

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 01a_Compute_EEG_Decoding_Accuracy_RDMs.py --subject $subject