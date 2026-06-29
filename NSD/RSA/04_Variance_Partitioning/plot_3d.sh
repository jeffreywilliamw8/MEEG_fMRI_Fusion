#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=plot_3d_wb_rsa_variance_partitioning
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=6:00:00
#SBATCH --qos=standard


# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/RSA/code/04_Variance_Partitioning

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 04d_Plot_3D_Variance_Partitioning_Results.py