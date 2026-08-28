#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=wb_layerwise_alexnet_jefe_phase_2
#SBATCH --mail-type=end
#SBATCH --mem=60000
#SBATCH --time=04:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
declare -a fmri_hemi_all
declare -a fmri_split_all
declare -a layer_all
index=0
for s in 1 4; do
    for h in 'lh' 'rh'; do
        for f in $(seq 1 21) ; do
            for l in 'features.2' 'features.5' 'features.7' 'features.9' 'features.12' 'classifier.2' 'classifier.5' 'classifier.6'; do
                subject_all[$index]=$s
                fmri_hemi_all[$index]=$h
                fmri_split_all[$index]=$f
                layer_all[$index]=$l
                ((index=index+1))
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
fmri_hemi=${fmri_hemi_all[$SLURM_ARRAY_TASK_ID]}
fmri_split=${fmri_split_all[$SLURM_ARRAY_TASK_ID]}
layer=${layer_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo fmri_hemi: $fmri_hemi
echo fmri_split: $fmri_split
echo layer: $layer

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/Encoding_Models/code/04_Joint_EEG_Feature_Encoding/04.2_layerwise_alexnet

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 04f_WB_AlexNet_Layerwise_JEFE_Phase_2.py --subject $subject --hemisphere $fmri_hemi --fmri_split $fmri_split --layer $layer