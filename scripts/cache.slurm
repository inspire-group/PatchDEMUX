#!/bin/bash

#SBATCH --job-name=cache_multilabel_patchcleanser_6mask_and_6mask
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --nodes=1                # node count (number of different machine)
#SBATCH --ntasks-per-node=1      # number of tasks per-node (choose equal to gpus) [make sure ntasks and ngpus are equal]
#SBATCH --gpus-per-node=1        # gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=32:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-15
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
    #PATCH_SIZE=55
    PATCH_SIZE=28
elif [[ "$ARCH" == "resnet" ]]; then
    IMAGE_SIZE=448
    BATCH_SIZE=64
    PATCH_SIZE=64
fi

MASK_NUMBER_FR=6
MASK_NUMBER_SR=6

# GPU info
WORLD_GPU_ID=$SLURM_ARRAY_TASK_ID
TOTAL_NUM_GPU=16

# Misc.
TRIAL=1

# Trained model specifics
PRETRAINED="false"

if [[ "$PRETRAINED" == "true" ]]; then
    EPOCH=0
    CUTOUT_TYPE="randomcutout"
    CUTOUT_INFO="size_192"
    TRAIN_INFO="training_onecyclelr_mixedprec_ema"

    TRIAL_TYPE="${CUTOUT_TYPE}_${CUTOUT_INFO}_${TRAIN_INFO}_epoch${EPOCH}"
elif [[ "$PRETRAINED" == "false" ]]; then
    TRIAL_TYPE="vanilla"
fi

# File/path locations
DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/coco/"

# Model weight locations
if [[ "$ARCH" == "ViT" ]]; then
    # ViT weights
    MODEL_NAME="Q2L-CvT_w24-384"

    if [[ "$PRETRAINED" == "true" ]]; then
        MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/ViT_trained/${CUTOUT_TYPE}/${CUTOUT_INFO}/${TRAIN_INFO}/08-29-2024/trial_2/epoch_${EPOCH}/ema-model-epoch-${EPOCH}.pth"
    elif [[ "$PRETRAINED" == "false" ]]; then
        MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/transformer/checkpoint.pkl"
    fi
    
elif [[ "$ARCH" == "resnet" ]]; then
    # ResNet weights
    MODEL_NAME="tresnet_l"

    if [[ "$PRETRAINED" == "true" ]]; then
        MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/resnet_trained/${CUTOUT_TYPE}/${CUTOUT_INFO}/${TRAIN_INFO}/08-15-2024/trial_1/epoch_${EPOCH}/ema-model-epoch-${EPOCH}.pth"
    elif [[ "$PRETRAINED" == "false" ]]; then
        MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/MS_COCO_TRresNet_L_448_86.6.pth"
    fi

fi

CONFIG="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/transformer/config_new.json"

python ml_pc_cache_generation.py $DATA_DIR --image-size $IMAGE_SIZE --batch-size $BATCH_SIZE --model-name $MODEL_NAME --model-path $MODEL_PATH --config $CONFIG --patch-size $PATCH_SIZE --mask-number-fr $MASK_NUMBER_FR --mask-number-sr $MASK_NUMBER_SR --world-gpu-id $WORLD_GPU_ID --total-num-gpu $TOTAL_NUM_GPU --trial $TRIAL --trial-type $TRIAL_TYPE