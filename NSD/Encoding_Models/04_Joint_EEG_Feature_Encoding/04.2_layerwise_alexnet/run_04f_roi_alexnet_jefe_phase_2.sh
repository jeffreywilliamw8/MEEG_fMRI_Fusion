#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=roi_layerwise_alexnet_jefe_phase_2
#SBATCH --mail-type=end
#SBATCH --mem=60000
#SBATCH --time=8:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
declare -a hemi_all
declare -a roi_all
declare -a layer_all
index=0

for sub in 1 4 5 6 7 8; do
    for h in 'lh' 'rh' ; do
        for roi in 'V1v' 'V1d' 'V2v' 'V2d' 'V3v' 'V3d' 'hV4' 'ventral'; do
            for l in 'features.2' 'features.5' 'features.7' 'features.9' 'features.12' 'classifier.2' 'classifier.5' 'classifier.6'; do
                subject_all[$index]=$sub
                hemi_all[$index]=$h
                roi_all[$index]=$roi
                layer_all[$index]=$l
                ((index=index+1))
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
hemi=${hemi_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
layer=${layer_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo hemi: $hemi
echo roi: $roi
echo layer: $layer

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/Encoding_Models/code/04_Joint_EEG_Feature_Encoding/04.2_layerwise_alexnet

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 04f_ROI_AlexNet_Layerwise_JEFE_Phase_2.py --subject $subject --hemisphere $hemi --roi $roi --layer $layer