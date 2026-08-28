#!/bin/bash
#SBATCH --mail-user=jeffreykatab@zedat.fu-berlin.de 
#SBATCH --job-name=generate_image_captions
#SBATCH --mail-type=end
#SBATCH --mem=25000
#SBATCH --time=72:00:00
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module add CUDA/12.4.0
PYTHON_SCRIPT="03b_Generate_Image_Captions.py"
cd /home/jeffreykatab/Projects/fusion/THINGS/Encoding_Models/code/03_Feature_Extraction
source /home/jeffreykatab/anaconda3/etc/profile.d/conda.sh
conda activate myenv
python3 $PYTHON_SCRIPT
