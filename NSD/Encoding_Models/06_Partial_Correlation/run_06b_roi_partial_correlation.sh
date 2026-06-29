#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=roi_partial_correlation
#SBATCH --mail-type=end
#SBATCH --mem=70000
#SBATCH --time=3:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
declare -a hemi_all
declare -a roi_all

index=0
for sub in 1 4 5 6 7 8; do
    for h in 'lh' 'rh' ; do
        for r in 'V1v' 'V1d' 'V2v' 'V2d' 'V3v' 'V3d' 'hV4' 'ventral'; do
            subject_all[$index]=$sub
            hemi_all[$index]=$h
            roi_all[$index]=$r
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
hemi=${hemi_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo hemi: $hemi
echo roi: $roi

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/Encoding_Models/code/06_Partial_Correlation

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 06b_ROI_Partial_Correlation.py --subject $subject  --hemisphere $hemi --roi $roi