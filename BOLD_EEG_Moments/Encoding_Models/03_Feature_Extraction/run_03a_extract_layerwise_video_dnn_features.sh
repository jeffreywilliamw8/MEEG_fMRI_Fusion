#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de
#SBATCH --job-name=extract_layerwise_video_dnn_features
#SBATCH --mail-type=end
#SBATCH --mem=80000
#SBATCH --time=00:45:00
#SBATCH --qos=standard

PYTHON_SCRIPT="03a_Extract_Layerwise_Video_DNN_Features.py"
cd /home/jeffreykatab/Projects/fusion/BOLD_EEG_Moments/Encoding_Models/code/03_Feature_Extraction
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv
python3 $PYTHON_SCRIPT
