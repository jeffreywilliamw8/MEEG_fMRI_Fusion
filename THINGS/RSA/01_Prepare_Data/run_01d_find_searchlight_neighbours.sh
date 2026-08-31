#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=find_searchlight_neighbours
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=02:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
index=0
for s in "01" "02" "03"; do
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
cd /home/jeffreykatab/Projects/fusion/THINGS/RSA/code/01_Prepare_Data

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 01d_Find_Searchlight_Neighbours.py --subject $subject