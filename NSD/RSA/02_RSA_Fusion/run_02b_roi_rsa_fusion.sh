#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=roi_rsa_fusion
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=00:45:00
#SBATCH --qos=standard


# Create the parameters combinations
declare -a subject_all
declare -a roi_all
declare -a eeg_metric_all
declare -a fmri_metric_all

index=0
for sub in 1 4 5 6 7 8; do
    for roi in 'V1' 'V2' 'V3' 'hV4' 'ventral'; do
        for eeg_metric in 'correlation' 'cosine' 'euclidean'; do
            for fmri_metric in 'correlation' 'cosine' 'euclidean'; do
                subject_all[$index]=$sub
                roi_all[$index]=$roi
                eeg_metric_all[$index]=$eeg_metric
                fmri_metric_all[$index]=$fmri_metric
                ((index=index+1))
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
eeg_metric=${eeg_metric_all[$SLURM_ARRAY_TASK_ID]}
fmri_metric=${fmri_metric_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo roi: $roi
echo eeg_metric: $eeg_metric
echo fmri_metric: $fmri_metric

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/RSA/code/02_RSA_Fusion

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 02b_ROI_RSA_Fusion.py --subject $subject --roi $roi --eeg_rdm_metric $eeg_metric --fmri_rdm_metric $fmri_metric