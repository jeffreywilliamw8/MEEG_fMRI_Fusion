#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=layerwise_alexnet_commonality_analysis
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=1:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
declare -a roi_all
declare -a layer_all
index=0

for sub in 1 4 5 6 7 8; do
    for roi in 'V1' 'V2' 'V3' 'hV4' 'ventral'; do
        for l in 'features.2' 'features.5' 'features.7' 'features.9' 'features.12' 'classifier.2' 'classifier.5' 'classifier.6' ; do
            subject_all[$index]=$sub
            roi_all[$index]=$roi
            layer_all[$index]=$l
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
layer=${layer_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo roi: $roi
echo layer: $layer

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/RSA/code/03_Commonality_Analysis/03.2_layerwise_alexnet

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 03g_Layerwise_AlexNet_Commonality_Analysis.py --subject $subject --roi $roi --layer $layer