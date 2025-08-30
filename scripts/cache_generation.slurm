#!/bin/bash

# Caching Script
# 
# This script caches model outputs, which can be used later when sweeping model thresholds
#
# Features:
# - Can be configured to run different architectures (i.e., Resnet, ViT) and different datasets (i.e., MSCOCO, PASCALVOC)
# - Caches model outputs relevant to certification, inference, and clean settings. 
#
# Prerequisites: Requires ~50GB of disk space.
# Usage: sbatch cache_generation.slurm

#SBATCH --job-name=cache_outputs_patchdemux
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --nodes=1                # node count (number of different machine)
#SBATCH --ntasks-per-node=1      # number of tasks per-node (choose equal to gpus) [make sure ntasks and ngpus are equal]
#SBATCH --gpus-per-node=1        # gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=35:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-15
#SBATCH --constraint="nomig&gpu40"  # request 40GB GPU without MIG partitioning
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
    BATCH_SIZE=16
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
    DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/coco/"
    CONFIG="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/transformer/config_new.json"
    NUM_CLASSES=80
elif [[ "$DATASET_NAME" == "pascalvoc" ]]; then
    DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/pascal-voc/"
    CONFIG="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/pascalvoc/transformer/config_new.json"
    NUM_CLASSES=20
fi

# GPU info
WORLD_GPU_ID=$SLURM_ARRAY_TASK_ID
TOTAL_NUM_GPU=16

# Misc.
TRIAL=1
CACHE_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser"

# Trained model specifics
DEF_FINETUNE="true" 

if [[ "$DEF_FINETUNE" == "true" ]]; then
    EPOCH=0
    CUTOUT_TYPE="greedycutout"
    CUTOUT_INFO="patch_55_masknum_6"
    TRAIN_INFO="training_onecyclelr_mixedprec_ema"

    TRIAL_TYPE="${CUTOUT_TYPE}_${CUTOUT_INFO}_${TRAIN_INFO}_epoch${EPOCH}"
elif [[ "$DEF_FINETUNE" == "false" ]]; then
    TRIAL_TYPE="vanilla"
fi

# Model weight locations - paths should correspond to valid checkpoints for the desired dataset
if [[ "$ARCH" == "ViT" ]]; then
    # ViT weights
    MODEL_NAME="Q2L-CvT_w24-384"

    if [[ "$DEF_FINETUNE" == "true" ]]; then
        MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/ViT_trained/${CUTOUT_TYPE}/${CUTOUT_INFO}/${TRAIN_INFO}/08-31-2024/trial_2/epoch_${EPOCH}/ema-model-epoch-${EPOCH}.pth"
    elif [[ "$DEF_FINETUNE" == "false" ]]; then
        MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/transformer/checkpoint.pkl"
    fi
    
elif [[ "$ARCH" == "resnet" ]]; then
    # ResNet weights
    MODEL_NAME="tresnet_l"

    if [[ "$DEF_FINETUNE" == "true" ]]; then
        MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/resnet_trained/${CUTOUT_TYPE}/${CUTOUT_INFO}/${TRAIN_INFO}/08-15-2024/trial_1/epoch_${EPOCH}/ema-model-epoch-${EPOCH}.pth"
    elif [[ "$DEF_FINETUNE" == "false" ]]; then
        MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/MS_COCO_TRresNet_L_448_86.6.pth"
    fi

fi

# Build the python command with conditional config parameter
PYTHON_CMD="python preprocessing/generate_cached_outputs.py $DATA_DIR --dataset-name $DATASET_NAME --num-classes $NUM_CLASSES --image-size $IMAGE_SIZE --batch-size $BATCH_SIZE --model-name $MODEL_NAME --model-path $MODEL_PATH"

# Add config parameter only for ViT models
if [[ "$ARCH" == "ViT" ]]; then
    PYTHON_CMD="$PYTHON_CMD --config $CONFIG"
fi

PYTHON_CMD="$PYTHON_CMD --patch-size $PATCH_SIZE --mask-number $MASK_NUMBER --world-gpu-id $WORLD_GPU_ID --total-num-gpu $TOTAL_NUM_GPU --trial $TRIAL --trial-type $TRIAL_TYPE --cache-dir $CACHE_DIR"

# Execute the command
$PYTHON_CMD