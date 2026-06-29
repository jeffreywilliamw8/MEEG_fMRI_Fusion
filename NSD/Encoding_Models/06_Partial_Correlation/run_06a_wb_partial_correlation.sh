#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=wb_variance_partitioning
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=02:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
declare -a fmri_hemi_all
declare -a fmri_split_all
index=0
for s in 1 4 5 6 7 8; do
    for h in 'lh'; do
        for f in $(seq 1 21) ; do
            subject_all[$index]=$s
            fmri_hemi_all[$index]=$h
            fmri_split_all[$index]=$f
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
fmri_hemi=${fmri_hemi_all[$SLURM_ARRAY_TASK_ID]}
fmri_split=${fmri_split_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo fmri_hemi: $fmri_hemi
echo fmri_split: $fmri_split

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/Encoding_Models/code/06_Partial_Correlation

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 06a_WB_Partial_Correlation.py --subject $subject --hemisphere $fmri_hemi --fmri_split $fmri_split