#!/bin/bash

# Defense Finetuning with Cutout Script
# 
# This script finetunes a multi-label classifier with different cutout techniques in order to
# achieve stronger performance when PatchDEMUX is initialized with the single-label CDPA PatchCleanser.
#
# Features:
# - Finetunes model according to desired cutout technique 
# - Can be configured to run different model architectures
# - Can be configured with different training hyperparameters
#
# Prerequisites: A multi-label classifier that has already done some initial training on the dataset
# Usage: sbatch train_cutout.slurm

#SBATCH --job-name=defense_finetuning_cutout
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
    BATCH_SIZE=8
elif [[ "$ARCH" == "resnet" ]]; then
    IMAGE_SIZE=448
    BATCH_SIZE=64
fi

# Training specifics
LR=0.00005
AMP="--amp"
LR_SCHEDULER="onecyclelr"
EMA_DECAY_RATE=0.9997

# MSCOCO dataset info
DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/coco/"
CONFIG="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/transformer/config_new.json"
NUM_CLASSES=80

# Misc.
TRIAL=2
CACHE_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser"

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

# Cutout info - configure cutout type here
CUTOUT_TYPE="greedycutout"

if [[ "$CUTOUT_TYPE" == "randomcutout" ]]; then
    CUTOUT_SIZE=192
elif [[ "$CUTOUT_TYPE" == "greedycutout" ]]; then
    GREEDY_CUTOUT_PATH="/home/djacob/multi-label-patchcleanser/dump/greedy_cutout/mscoco/patch_55_masknumfr_6_masknumsr_6/ViT/08-25-2024/trial_2_vanilla/gpu_world_id_0/greedy_cutout_dict.json"
fi

# Build the python command with conditional config parameter
PYTHON_CMD="python train/cutout_defense_finetuning.py $DATA_DIR --image-size $IMAGE_SIZE --batch-size $BATCH_SIZE --model-path $MODEL_PATH --model-name $MODEL_NAME"

# Add config parameter only for ViT models
if [[ "$ARCH" == "ViT" ]]; then
    PYTHON_CMD="$PYTHON_CMD --config $CONFIG"
fi

# Add cutout-specific parameters based on cutout type
PYTHON_CMD="$PYTHON_CMD --lr $LR --cutout-type $CUTOUT_TYPE"

if [[ "$CUTOUT_TYPE" == "randomcutout" ]]; then
    PYTHON_CMD="$PYTHON_CMD --cutout-size $CUTOUT_SIZE"
elif [[ "$CUTOUT_TYPE" == "greedycutout" ]]; then
    PYTHON_CMD="$PYTHON_CMD --greedy-cutout-path $GREEDY_CUTOUT_PATH"
fi

PYTHON_CMD="$PYTHON_CMD $AMP --lr-scheduler $LR_SCHEDULER --ema-decay-rate $EMA_DECAY_RATE --trial $TRIAL --cache-dir $CACHE_DIR"

# Execute the command
$PYTHON_CMD