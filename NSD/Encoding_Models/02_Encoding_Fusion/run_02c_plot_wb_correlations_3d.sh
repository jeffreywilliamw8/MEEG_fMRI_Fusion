#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=plot_em_3d_wb_correlations
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=4:00:00
#SBATCH --qos=standard


# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/Encoding_Models/code/02_Encoding_Fusion

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 02c_Plot_WB_Correlations_3D.py