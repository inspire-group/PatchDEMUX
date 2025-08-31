#!/bin/bash

# Cache Interpolation Script
# 
# This script interpolates between provided threshold values to achieve
# some target recall value. 
#
# Features:
# - Can be configured to run certification, defended clean, and undefended clean settings
#
# Prerequisites: Run cache generation script first to create cached outputs. Then run cache thresholding to compute a reasonable set of recall values that will aid interpolation. 
# Usage: sbatch run_cache_interpolation.slurm

#SBATCH --job-name=cache_interpolation
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --nodes=1                # node count (number of different machine)
#SBATCH --ntasks-per-node=1      # number of tasks per-node (choose equal to gpus) [make sure ntasks and ngpus are equal]
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)
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

# Path locations and thresholds
CACHE_DIR="${SCRATCH_DIR}"
CACHED_OUTPUTS=${CACHE_DIR}/cached_outputs/mscoco/ViT/patch_55_masknum_6/08-22-2025/trial_1_greedycutout_patch_55_masknum_6_training_onecyclelr_mixedprec_ema_epoch0/cached_outputs

# Interpolation parameters - for each element in the target recall array, add values below corresponding to recalls directly below and above the target
#FIRST_THRE_ARR=()
#FIRST_RECALL_ARR=()
#SECOND_THRE_ARR=()
#SECOND_RECALL_ARR=()

# certification example
#FIRST_THRE_ARR=(0.999 0.86 0.60)
#FIRST_RECALL_ARR=(19.35 48.67 73.59)
#SECOND_THRE_ARR=(0.99 0.84 0.58)
#SECOND_RECALL_ARR=(30.9 50.61 75.43)

TARGET_RECALL_ARR=(25.0 50.0 75.0)

# Model parameters
ARCH="ViT"

if [[ "$ARCH" == "ViT" ]]; then
    IMAGE_SIZE=384
    PATCH_SIZE=55
elif [[ "$ARCH" == "resnet" ]]; then
    IMAGE_SIZE=448
    PATCH_SIZE=64
fi

MASK_NUMBER=6

# Dataset info
DATASET_NAME="mscoco"
if [[ "$DATASET_NAME" == "mscoco" ]]; then
    NUM_CLASSES=80
elif [[ "$DATASET_NAME" == "pascalvoc" ]]; then
    NUM_CLASSES=20
fi

# Misc.
TRIAL=2

# Type of run
RUN_TYPE="defended"
HIST="no_hist"

if [ "$RUN_TYPE" == "certification" ] && [ "$HIST" == "no_hist" ]; then
    ATTACKER_TYPE="none"
elif [ "$RUN_TYPE" == "certification" ] && [ "$HIST" == "hist" ]; then
    ATTACKER_TYPE="worst_case"
elif [[ "$RUN_TYPE" == "defended" ]]; then
    DEFENSE_STATUS="--defense"
elif [[ "$RUN_TYPE" == "undefended" ]]; then
    DEFENSE_STATUS="--no-defense"
fi

# Run interpolation
for i in {0..2}
do
    FIRST_THRE=${FIRST_THRE_ARR[$i]}
    FIRST_RECALL=${FIRST_RECALL_ARR[$i]}
    SECOND_THRE=${SECOND_THRE_ARR[$i]}
    SECOND_RECALL=${SECOND_RECALL_ARR[$i]}

    TARGET_RECALL=${TARGET_RECALL_ARR[$i]}

    if [[ "$RUN_TYPE" == "certification" ]]; then
        python certify/pd_certify_cached_interpolate.py --cache-location $CACHED_OUTPUTS --dataset-name $DATASET_NAME --num-classes $NUM_CLASSES --image-size $IMAGE_SIZE --attacker-type $ATTACKER_TYPE --first-thre $FIRST_THRE --first-recall $FIRST_RECALL --second-thre $SECOND_THRE --second-recall $SECOND_RECALL --target-recall $TARGET_RECALL --patch-size $PATCH_SIZE --mask-number $MASK_NUMBER --trial $TRIAL
    else
        python inference/pd_infer_cached_interpolate.py $DEFENSE_STATUS --cache-location $CACHED_OUTPUTS --dataset-name $DATASET_NAME --num-classes $NUM_CLASSES --image-size $IMAGE_SIZE --first-thre $FIRST_THRE --first-recall $FIRST_RECALL --second-thre $SECOND_THRE --second-recall $SECOND_RECALL --target-recall $TARGET_RECALL --patch-size $PATCH_SIZE --mask-number $MASK_NUMBER --trial $TRIAL
    fi
done