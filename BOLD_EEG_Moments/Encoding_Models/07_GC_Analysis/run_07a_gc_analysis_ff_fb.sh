#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=gc_analysis_ff_fb
#SBATCH --mail-type=end
#SBATCH --mem=8500
#SBATCH --time=00:10:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a fmri_subject_all
declare -a cv_all
declare -a roi_pairs=(
    "V1-V2" "V1-V3" "V1-V4" "V1-FFA" "V1-OFA" "V1-EBA" "V1-PPA" "V1-LOC"
    "V2-V3" "V2-V4" "V2-FFA" "V2-OFA" "V2-EBA" "V2-PPA" "V2-LOC"
    "V3-V4" "V3-FFA" "V3-OFA" "V3-EBA" "V3-PPA" "V3-LOC"
    "V4-FFA" "V4-OFA" "V4-EBA" "V4-PPA" "V4-LOC"
    "FFA-OFA" "FFA-EBA" "FFA-PPA" "FFA-LOC"
    "OFA-EBA" "OFA-PPA" "OFA-LOC"
    "EBA-PPA" "EBA-LOC"
    "PPA-LOC"
) # 9 ROIs => 36 unique pairs
index=0

for s in "01" "02" "03" "04" "05" "06" "07" "08" "09" "10"; do
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
cd /home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/code/07_GC_Analysis

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
# Note: Ensure your .py script arguments match these new flags
python 07a_GC_Analysis_FF_FB.py --fmri_subject $fmri_subject --rois $roi_pair --cross_validate $cv_state