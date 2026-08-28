#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=wb_sfef_phase_1
#SBATCH --mail-type=end
#SBATCH --mem=25000
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --qos=standard

# Create the parameters combinations
declare -a fmri_subject_all
declare -a fmri_split_all
declare -a half_all

index=0
for s in 1 2 3; do
    for f in $(seq 1 125) ; do
        for h in 1 2; do
            fmri_subject_all[$index]=$s
            fmri_split_all[$index]=$f
            half_all[$index]=$h
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
fmri_split=${fmri_split_all[$SLURM_ARRAY_TASK_ID]}
half=${half_all[$SLURM_ARRAY_TASK_ID]}

echo fmri_subject: $fmri_subject
echo fmri_split: $fmri_split
echo half: $half


# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/code/02_Encoding_Fusion

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 02a_Whole_Brain_Encoding_Fusion.py --fmri_subject $fmri_subject --fmri_split $fmri_split --half $half