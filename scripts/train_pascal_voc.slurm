#!/bin/bash

# Transfer Learning for PASCAL VOC
# 
# This script takes existing checkpoints and transfer learns them for PASCAL VOC.
#
# Features:
# - Can be configured with different training hyperparameters
#
# Prerequisites: A multi-label classifier that has already been trained on another dataset (i.e., MSCOCO)
# Usage: sbatch train_pascal_voc.slurm

#SBATCH --job-name=transfer_learn_pascalvoc
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
IMAGE_SIZE=384
BATCH_SIZE=8

# Training specifics
LR=0.002
AMP="--amp"
LR_SCHEDULER="onecyclelr"
EMA_DECAY_RATE=0.9997

# PASCAL VOC dataset info
DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/pascal-voc/"

# Misc.
TRIAL=2
CACHE_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser"

# ViT model weight locations - start with base MSCOCO checkpoints and then finetune for the PASCAL VOC task
MODEL_NAME="Q2L-CvT_w24-384"
MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/transformer/checkpoint.pkl"
CONFIG="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/transformer/config_new.json"

python train/pascalvoc_transfer_learning.py $DATA_DIR --image-size $IMAGE_SIZE --batch-size $BATCH_SIZE --model-path $MODEL_PATH --model-name $MODEL_NAME --config $CONFIG --lr $LR $AMP --lr-scheduler $LR_SCHEDULER --ema-decay-rate $EMA_DECAY_RATE --trial $TRIAL --cache-dir $CACHE_DIR