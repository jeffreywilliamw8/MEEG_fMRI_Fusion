#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=encoding_fusion_roi
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=02:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a fmri_subject_all
declare -a fmri_hemi_all
declare -a roi_all
index=0
for s in "01" "02" "03" "04" "05" "06" "07" "08" "09" "10"; do
    for h in 'left' 'right' ; do
        for r in 'V1v' 'V1d' 'V2v' 'V2d' 'V3v' 'V3d' 'V3ab' 'hV4' 'FFA' 'OFA' 'EBA' 'PPA' 'LOC'; do
            fmri_subject_all[$index]=$s
            fmri_hemi_all[$index]=$h
            roi_all[$index]=$r
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
fmri_hemi=${fmri_hemi_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo fmri_hemi: $fmri_hemi
echo roi: $roi

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/code/02_Encoding_Fusion

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 02c_Encoding_Fusion_ROI.py --fmri_subject $fmri_subject --hemisphere $fmri_hemi --roi $roi 