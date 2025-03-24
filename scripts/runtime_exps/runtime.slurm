#!/bin/bash

#SBATCH --job-name=validate_multilabel_patchcleanser_6mask_and_12mask
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --nodes=1                # node count (number of different machine)
#SBATCH --ntasks-per-node=1      # number of tasks per-node (choose equal to gpus) [make sure ntasks and ngpus are equal]
#SBATCH --gpus-per-node=1        # gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-3
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
PATCH_CLEANSER="--patchcleanser"
ARCH="ViT"

IMAGE_SIZE=384
BATCH_SIZE=1
PATCH_SIZE=55
THRE=0.8

#mask 6 run
MASK_NUMBER_FR=6
MASK_NUMBER_SR=6

# GPU info
WORLD_GPU_ID=$SLURM_ARRAY_TASK_ID
TOTAL_NUM_GPU=4

# Misc.
TRIAL=2

# Trained model specifics
PRETRAINED="true"

if [[ "$PRETRAINED" == "true" ]]; then
    EPOCH=0
    CUTOUT_TYPE="greedycutout"
    CUTOUT_INFO="patch_55_masknum_6"
    TRAIN_INFO="training_onecyclelr_mixedprec_ema"

    TRIAL_TYPE="${CUTOUT_TYPE}_${CUTOUT_INFO}_${TRAIN_INFO}_epoch${EPOCH}"
elif [[ "$PRETRAINED" == "false" ]]; then
    TRIAL_TYPE="vanilla"
fi

# File/path locations
DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/coco/"

# ViT weights
MODEL_NAME="Q2L-CvT_w24-384"

if [[ "$PRETRAINED" == "true" ]]; then
    MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/ViT_trained/${CUTOUT_TYPE}/${CUTOUT_INFO}/${TRAIN_INFO}/08-31-2024/trial_2/epoch_${EPOCH}/ema-model-epoch-${EPOCH}.pth"
elif [[ "$PRETRAINED" == "false" ]]; then
    MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/transformer/checkpoint.pkl"
fi

CONFIG="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/transformer/config_new.json"

python ml_pc_clean_images_ViT_RUNTIME.py $DATA_DIR --patchcleanser $PATCH_CLEANSER --batch-size $BATCH_SIZE --model-name $MODEL_NAME --model-path $MODEL_PATH --config $CONFIG --thre $THRE --patch-size $PATCH_SIZE --mask-number-fr $MASK_NUMBER_FR --mask-number-sr $MASK_NUMBER_SR --world-gpu-id $WORLD_GPU_ID --total-num-gpu $TOTAL_NUM_GPU --trial $TRIAL --trial-type $TRIAL_TYPE
