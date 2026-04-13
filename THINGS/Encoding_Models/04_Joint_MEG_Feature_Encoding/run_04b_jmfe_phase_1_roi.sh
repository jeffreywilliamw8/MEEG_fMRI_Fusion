#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=jmfe_phase_1_roi
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=02:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a fmri_subject_all
declare -a fmri_roi_all

index=0
for s in $(seq 1 3) ; do
    for r in 'V1' 'V2' 'V3' 'hV4' 'lFFA' 'rFFA' 'lOFA' 'rOFA' 'lEBA' 'rEBA' 'lPPA' 'rPPA' 'lLOC' 'rLOC' 'IT'; do
        fmri_subject_all[$index]=$s
        fmri_roi_all[$index]=$r
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${fmri_roi_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo roi: $roi

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/code/04_Joint_MEG_Feature_Encoding

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 04b_JMFE_Phase_1_ROI.py --fmri_subject $fmri_subject --roi $roi