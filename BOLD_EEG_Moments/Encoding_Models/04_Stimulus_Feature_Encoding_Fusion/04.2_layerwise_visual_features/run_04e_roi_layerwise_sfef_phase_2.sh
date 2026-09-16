#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=roi_layerwise_sfef_phase_2
#SBATCH --mail-type=end
#SBATCH --mem=40000
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --qos=standard

# Create the parameters combinations
declare -a subject_all
declare -a hemi_all
declare -a roi_all
declare -a cv_split_all
declare -a layer_all
index=0

for sub in 8 9 10; do
    for h in 'left' 'right' ; do
        for roi in 'V1v' 'V1d' 'hV4' 'PPA'; do
            for l in 'Stem' 'Layer1' 'Layer2' 'Layer3' 'Layer4' 'AvgPool' 'FC'; do
                for cv in 'even' 'odd'; do
                    subject_all[$index]=$sub
                    hemi_all[$index]=$h
                    roi_all[$index]=$roi
                    layer_all[$index]=$l
                    cv_split_all[$index]=$cv
                    ((index=index+1))
                done
            done
        done
    done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
hemi=${hemi_all[$SLURM_ARRAY_TASK_ID]}
roi=${roi_all[$SLURM_ARRAY_TASK_ID]}
cv=${cv_split_all[$SLURM_ARRAY_TASK_ID]}
layer=${layer_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo hemi: $hemi
echo roi: $roi
echo cv: $cv
echo layer: $layer

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/code/04_Stimulus_Feature_Encoding_Fusion/04.2_layerwise_visual_features
# Activate the Anaconda environment
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the job
python 04e_ROI_Layerwise_SFEF_Phase_2.py --subject $subject --hemisphere $hemi --roi $roi --layer $layer --cv_split $cv