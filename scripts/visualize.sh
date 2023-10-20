#!/bin/bash

# Model parameters
PATCH_SIZE=64
MASK_NUMBER=6

# Visualization parameters
IMAGE_INDEX=2
FIRST_MASK_INDEX=25
SECOND_MASK_INDEX=26
#CLEAN_IM="--no-clean-im"
CLEAN_IM="--clean-im"

# File/path locations
DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/coco/"

# Baseline path
MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/MS_COCO_TRresNet_L_448_86.6.pth"

python ml_pc_visualization.py $DATA_DIR --model-path $MODEL_PATH --patch-size $PATCH_SIZE --mask-number $MASK_NUMBER --image-index $IMAGE_INDEX --first-mask-index $FIRST_MASK_INDEX --second-mask-index $SECOND_MASK_INDEX $CLEAN_IM