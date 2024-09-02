#!/bin/bash

#SBATCH --job-name=train_cutout_multilabel_patchcleanser
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --nodes=1                # node count (number of different machine)
#SBATCH --ntasks-per-node=1      # number of tasks per-node (choose equal to gpus) [make sure ntasks and ngpus are equal]
#SBATCH --gpus-per-node=1        # gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=30:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=djacob@princeton.edu

# Start the conda environment
module purge
module load anaconda3/2022.10
source activate torch-env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/djacob/.conda/envs/torch-env/lib 

# Model parameters
ARCH="ViT"

if [[ "$ARCH" == "ViT" ]]; then
    IMAGE_SIZE=384
    BATCH_SIZE=64
    PATCH_SIZE=55
elif [[ "$ARCH" == "resnet" ]]; then
    IMAGE_SIZE=448
    BATCH_SIZE=64
    PATCH_SIZE=64
fi

MASK_NUMBER_FR=3
MASK_NUMBER_SR=3

# GPU info
WORLD_GPU_ID=0
TOTAL_NUM_GPU=1

# Misc.
TRIAL=2
TRIAL_TYPE="vanilla"

# File/path locations
DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/coco/"

# Model weight locations
if [[ "$ARCH" == "ViT" ]]; then
    # ViT weights
    MODEL_NAME="Q2L-CvT_w24-384"
    MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/transformer/checkpoint.pkl"
elif [[ "$ARCH" == "resnet" ]]; then
    # ResNet weights
    MODEL_NAME="tresnet_l"
    MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/MS_COCO_TRresNet_L_448_86.6.pth"
fi

CONFIG="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/transformer/config_new.json"

python ml_greedy_cutout_generation.py $DATA_DIR --image-size $IMAGE_SIZE --batch-size $BATCH_SIZE --model-path $MODEL_PATH --model-name $MODEL_NAME --config $CONFIG --patch-size $PATCH_SIZE --mask-number-fr $MASK_NUMBER_FR --mask-number-sr $MASK_NUMBER_SR --world-gpu-id $WORLD_GPU_ID --total-num-gpu $TOTAL_NUM_GPU --trial $TRIAL --trial-type $TRIAL_TYPE