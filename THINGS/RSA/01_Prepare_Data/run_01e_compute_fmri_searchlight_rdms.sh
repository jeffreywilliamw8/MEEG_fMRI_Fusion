#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=compute_fmri_searchlight_rdms
#SBATCH --mail-type=end
#SBATCH --mem=40000
#SBATCH --time=3:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
index=0
for sub in $(seq 1 3); do
    subject_all[$index]=$sub
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/THINGS/RSA/code/01_Prepare_Data

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 01e_Compute_fMRI_Searchlight_RDMs.py --subject $subject