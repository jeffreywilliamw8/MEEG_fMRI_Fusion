#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=324_358_searchlight_rsa_fusion
#SBATCH --mail-type=end
#SBATCH --mem=110000
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
declare -a hemi_all
declare -a time_point_all

index=0
for s in 1 4 5 6 7 8; do
    for h in 'lh' 'rh'; do
        for t in $(seq 324 358); do
            subject_all[$index]=$s
            hemi_all[$index]=$h
            time_point_all[$index]=$t
            ((index=index+1))
        done    
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
hemi=${hemi_all[$SLURM_ARRAY_TASK_ID]}
time_point=${time_point_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo hemi: $hemi
echo time_point: $time_point


# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/RSA/code/02_RSA_Fusion

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 02a_Searchlight_RSA_Fusion.py --subject $subject --hemisphere $hemi --time_point $time_point