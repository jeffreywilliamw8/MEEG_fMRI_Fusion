#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=plot_wb_correlations
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=4:00:00
#SBATCH --qos=standard


# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/RSA/code/02_RSA_Fusion

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 02d_Plot_WB_Correlations.py