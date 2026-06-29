#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=wb_jefe_phase_2
#SBATCH --mail-type=end
#SBATCH --mem=80000
#SBATCH --time=1:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
declare -a fmri_hemi_all
declare -a fmri_split_all
declare -a dnn_type_all
index=0
for s in 1 4 5 6 7 8; do
    for h in 'lh' 'rh'; do
        for f in $(seq 1 21) ; do
            for t in 'vdnn' 'llm' 'both' ; do
                subject_all[$index]=$s
                fmri_hemi_all[$index]=$h
                fmri_split_all[$index]=$f
                dnn_type_all[$index]=$t
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
dnn_type=${dnn_type_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo fmri_hemi: $fmri_hemi
echo fmri_split: $fmri_split
echo dnn_type: $dnn_type

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/Encoding_Models/code/04_Joint_EEG_Feature_Encoding/04.1_vision_language_models

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 04c_Whole_Brain_JEFE_Phase_2.py --subject $subject --hemisphere $fmri_hemi --fmri_split $fmri_split --dnn_type $dnn_type