#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=roi_fr_rsa_fusion
#SBATCH --mail-type=end
#SBATCH --mem=50000
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=standard


# Create the parameters combinations
declare -a subject_all
declare -a roi_all
index=0
for sub in 1 4 5 6 7 8; do
    for roi in 'V1' 'V2' 'V3' 'hV4' 'ventral'; do
        subject_all[$index]=$sub
        roi_all[$index]=$roi
        ((index=index+1))
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo roi: $roi

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/NSD/RSA/code/02_RSA_Fusion

# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 02c_ROI_Feature_Reweighted_RSA_Fusion.py --subject $subject --roi $roi