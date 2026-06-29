#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=roi_jefe_phase_1
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=06:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
declare -a hemi_all
declare -a roi_all
declare -a cv_split_all

index=0
for sub in 1 4 5 6 7 8; do
    for h in 'lh' 'rh' ; do
        for r in 'V1v' 'V1d' 'V2v' 'V2d' 'V3v' 'V3d' 'hV4' 'ventral'; do
            for cv_split in 'even' 'odd' ; do
                subject_all[$index]=$sub
                hemi_all[$index]=$h
                roi_all[$index]=$r
                cv_split_all[$index]=$cv_split
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
cv_split=${cv_split_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo hemi: $hemi
echo roi: $roi
echo cv_split: $cv_split

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/Encoding_Models/code/04_Joint_EEG_Feature_Encoding

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 04b_ROI_JEFE_Phase_1.py --subject $subject  --hemisphere $hemi --roi $roi --cv_split $cv_split