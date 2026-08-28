#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=3d_jefe_phase_2
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=03:00:00
#SBATCH --qos=standard

# Create the parameters combinations

declare -a dnn_type_all
index=0

for t in 'vdnn' 'llm' ; do
    dnn_type_all[$index]=$t
    ((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
dnn_type=${dnn_type_all[$SLURM_ARRAY_TASK_ID]}
echo dnn_type: $dnn_type

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/Encoding_Models/code/04_Joint_EEG_Feature_Encoding/04.1_vision_language_models

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 04g_WB_3D_Plot.py --dnn_type $dnn_type