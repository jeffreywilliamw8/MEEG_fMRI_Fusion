#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=merge_geodesic_vertex_distances
#SBATCH --mail-type=end
#SBATCH --mem=250000
#SBATCH --time=01:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a hemisphere_all
index=0
for h in 'lh' 'rh' ; do
    hemisphere_all[$index]=$h
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
echo hemisphere: $hemisphere

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/code/01_Preparing_Data/berg_code

# Run the job
python 02_merge_geodesic_vertex_distances.py --hemisphere $hemisphere