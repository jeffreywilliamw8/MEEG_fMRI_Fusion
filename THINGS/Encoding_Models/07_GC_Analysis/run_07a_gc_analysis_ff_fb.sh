#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=gc_analysis_ff_fb
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=00:30:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a fmri_subject_all
declare -a cv_all
declare -a roi_pairs=(
     "V1-V2" "V1-V3" "V1-V4" "V1-IT"
    "V2-V3" "V2-V4" "V2-IT"
    "V3-V4" "V3-IT"
    "V4-IT"
) # 5 ROIs => 10 unique pairs
index=0

for s in $(seq 1 3); do
    for r in "${roi_pairs[@]}"; do
        for cv in "True" "False"; do
            fmri_subject_all[$index]=$s
            roi_pairs_all[$index]=$r
            cv_all[$index]=$cv
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
roi_pair=${roi_pairs_all[$SLURM_ARRAY_TASK_ID]}
cv_state=${cv_all[$SLURM_ARRAY_TASK_ID]}

echo fmri_subject: $fmri_subject
echo roi_pair: $roi_pair
echo cross_validate: $cv_state

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/code/07_GC_Analysis

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 07a_GC_Analysis_FF_FB_2.py --fmri_subject $fmri_subject --rois $roi_pair --cross_validate $cv_state