#!/bin/bash

#SBATCH --job-name=cache_multilabel_patchcleanser_6mask_and_6mask
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --nodes=1                # node count (number of different machine)
#SBATCH --ntasks-per-node=1      # number of tasks per-node (choose equal to gpus) [make sure ntasks and ngpus are equal]
#SBATCH --gpus-per-node=1        # gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=5:00:00          # total run time limit (HH:MM:SS)
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

IMAGE_SIZE=384
BATCH_SIZE=16
PATCH_SIZE=55

#mask 6 run
MASK_NUMBER_FR=6
MASK_NUMBER_SR=6

# Dataset specifics
DATASET_NAME="pascalvoc"
NUM_CLASSES=20

# GPU info
WORLD_GPU_ID=$SLURM_ARRAY_TASK_ID
TOTAL_NUM_GPU=16

# Misc.
TRIAL=1

# Trained model specifics
PRETRAINED="true"
TRIAL_TYPE="vanilla"

# File/path locations
DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/pascal-voc/"

# Model weight locations
MODEL_NAME="Q2L-CvT_w24-384"
MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/pascalvoc/transformer/checkpoint_lr_0_002_epoch_13.pth"
CONFIG="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/pascalvoc/transformer/config_new.json"

python ml_pc_cache_generation.py $DATA_DIR --dataset-name $DATASET_NAME --num-classes $NUM_CLASSES --image-size $IMAGE_SIZE --batch-size $BATCH_SIZE --model-name $MODEL_NAME --model-path $MODEL_PATH --config $CONFIG --patch-size $PATCH_SIZE --mask-number-fr $MASK_NUMBER_FR --mask-number-sr $MASK_NUMBER_SR --world-gpu-id $WORLD_GPU_ID --total-num-gpu $TOTAL_NUM_GPU --trial $TRIAL --trial-type $TRIAL_TYPE