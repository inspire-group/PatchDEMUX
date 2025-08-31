#!/bin/bash

# Greedy Cutout Script
# 
# This script generates greedy cutout masks for a multi-label classifier using the method in https://arxiv.org/abs/2306.12610.
# Greedy cutout is a defense fine-tuning technique that was designed specifically to improve the robustness of
# the single-label CDPA PatchCleanser. We extend it here to the multi-classification domain to improve the
# robustness of PatchDEMUX. 
#
# Features:
# - Can be configured to run different architectures (i.e., Resnet, ViT)
#
# Usage: sbatch greedy_cutout_generation.slurm

#SBATCH --job-name=greedy_cutout_generation
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --nodes=1                # node count (number of different machine)
#SBATCH --ntasks-per-node=1      # number of tasks per-node (choose equal to gpus) [make sure ntasks and ngpus are equal]
#SBATCH --gpus-per-node=1        # gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --constraint="nomig&gpu40"  # request 40GB GPU without MIG partitioning
#SBATCH --time=30:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=YOUR_EMAIL_HERE

# Start the conda environment
module purge
module load anaconda3/2022.10
source activate torch-env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch-env/lib 

# Replace with your project path, e.g., "/scratch/gpfs/username/patchdemux"
SCRATCH_DIR="YOUR_PATH_HERE"

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

MASK_NUMBER=6

# Dataset info
DATASET_NAME="mscoco"

if [[ "$DATASET_NAME" == "mscoco" ]]; then
    DATA_DIR="${SCRATCH_DIR}/coco/"
    CONFIG="${SCRATCH_DIR}/checkpoints/mscoco/transformer/config_new.json"
    NUM_CLASSES=80
elif [[ "$DATASET_NAME" == "pascalvoc" ]]; then
    DATA_DIR="${SCRATCH_DIR}/pascal-voc/"
    CONFIG="${SCRATCH_DIR}/checkpoints/pascalvoc/transformer/config_new.json"
    NUM_CLASSES=20
fi

# GPU info
WORLD_GPU_ID=0
TOTAL_NUM_GPU=1

# Misc.
TRIAL=2
TRIAL_TYPE="vanilla"

# Model weight locations - paths should correspond to valid checkpoints for the desired dataset
if [[ "$ARCH" == "ViT" ]]; then
    # ViT weights - use vanilla checkpoints for greedy cutout generation
    MODEL_NAME="Q2L-CvT_w24-384"
    MODEL_PATH="${SCRATCH_DIR}/checkpoints/mscoco/transformer/checkpoint.pkl"
elif [[ "$ARCH" == "resnet" ]]; then
    # ResNet weights - use vanilla checkpoints for greedy cutout generation
    MODEL_NAME="tresnet_l"
    MODEL_PATH="${SCRATCH_DIR}/checkpoints/mscoco/MS_COCO_TRresNet_L_448_86.6.pth"
fi

# Build the python command with conditional config parameter
PYTHON_CMD="python preprocessing/generate_greedy_cutouts.py $DATA_DIR --dataset-name $DATASET_NAME --num-classes $NUM_CLASSES --image-size $IMAGE_SIZE --batch-size $BATCH_SIZE --model-path $MODEL_PATH --model-name $MODEL_NAME"

# Add config parameter only for ViT models
if [[ "$ARCH" == "ViT" ]]; then
    PYTHON_CMD="$PYTHON_CMD --config $CONFIG"
fi

PYTHON_CMD="$PYTHON_CMD --patch-size $PATCH_SIZE --mask-number $MASK_NUMBER --world-gpu-id $WORLD_GPU_ID --total-num-gpu $TOTAL_NUM_GPU --trial $TRIAL --trial-type $TRIAL_TYPE"

# Execute the command
$PYTHON_CMD