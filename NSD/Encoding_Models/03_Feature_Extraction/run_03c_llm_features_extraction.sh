#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=llm_feature_extraction
#SBATCH --mail-type=end
#SBATCH --mem=75000
#SBATCH --time=01:00:00
#SBATCH --qos=standard


# Create the parameters combinations
declare -a subject_all

index=0
for s in 1 4 5 6 7 8; do
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
cd /home/jeffreykatab/Projects/fusion/NSD/Encoding_Models/code/03_Feature_Extraction

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 03c_LLM_Features_Extraction.py --subject $subject