#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=compute_geodesic_vertex_distances
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=18:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a hemisphere_all
declare -a vertex_split_all
index=0
for h in 'lh' 'rh' ; do
    for v in `seq 0 80` ; do
        hemisphere_all[$index]=$h
        vertex_split_all[$index]=$v
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
hemisphere=${hemisphere_all[$SLURM_ARRAY_TASK_ID]}
vertex_split=${vertex_split_all[$SLURM_ARRAY_TASK_ID]}
echo hemisphere: $hemisphere
echo vertex_split: $vertex_split

# Wait a bit so it doesn't crash
sleep 8

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/RSA/code/01_Preparing_Data/berg_code

# Run the job
python 01_compute_geodesic_vertex_distances.py --hemisphere $hemisphere --vertex_split $vertex_split