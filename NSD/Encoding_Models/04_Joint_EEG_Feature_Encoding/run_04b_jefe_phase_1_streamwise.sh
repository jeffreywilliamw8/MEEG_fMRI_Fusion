#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=jefe_phase_1_streamwise
#SBATCH --mail-type=end
#SBATCH --mem=70000
#SBATCH --time=08:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
declare -a fmri_hemi_all
declare -a stream_all
index=0
for sub in 1 4 5 6 7 8; do
    for h in 'lh' 'rh' ; do
        for st in 'early' 'midventral' 'midlateral' 'midparietal' 'ventral' 'lateral' 'parietal'; do
            subject_all[$index]=$sub
            fmri_hemi_all[$index]=$h
            stream_all[$index]=$st
            ((index=index+1))
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
fmri_hemi=${fmri_hemi_all[$SLURM_ARRAY_TASK_ID]}
stream=${stream_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo fmri_hemi: $fmri_hemi
echo stream: $stream

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/Encoding_Models/code/04_Joint_EEG_Feature_Encoding

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 04b_Streamwise_JEFE_Phase_1.py --subject $subject  --hemisphere $fmri_hemi --stream $stream