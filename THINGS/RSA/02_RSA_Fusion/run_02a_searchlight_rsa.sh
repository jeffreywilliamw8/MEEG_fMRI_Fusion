#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=searchlight_rsa_fusion
#SBATCH --mail-type=end
#SBATCH --mem=12000
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
declare -a time_point_split_all

index=0
for s in 1 2 3; do
    for t in 0 1 2; do
        subject_all[$index]=$s
        time_point_split_all[$index]=$t
        ((index=index+1))
    done    
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
time_point_split=${time_point_split_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo time_point_split: $time_point_split


# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/THINGS/RSA/code/02_RSA_Fusion

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 02a_Searchlight_RSA_Fusion.py --subject $subject  --time_point_split $time_point_split