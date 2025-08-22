#!/bin/bash

#SBATCH --job-name=train_cutout_multilabel_patchcleanser
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

# Misc.
TRIAL=2

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

# Cutout info
CUTOUT_SIZE=192
#CUTOUT_TYPE="randomcutout"
CUTOUT_TYPE="greedycutout"
#GREEDY_CUTOUT_PATH="/home/djacob/multi-label-patchcleanser/dump/greedy_cutout/mscoco/patch_64_masknumfr_3_masknumsr_3/resnet/08-15-2024/trial_2_vanilla/gpu_world_id_0/greedy_cutout_dict.json"
GREEDY_CUTOUT_PATH="/home/djacob/multi-label-patchcleanser/dump/greedy_cutout/mscoco/patch_55_masknumfr_3_masknumsr_3/ViT/08-25-2024/trial_2_vanilla/gpu_world_id_0/greedy_cutout_dict.json"

python ml_ASL_cutout_train.py $DATA_DIR --image-size $IMAGE_SIZE --batch-size $BATCH_SIZE --model-path $MODEL_PATH --model-name $MODEL_NAME --config $CONFIG --lr $LR --cutout-size $CUTOUT_SIZE --cutout-type $CUTOUT_TYPE --greedy-cutout-path $GREEDY_CUTOUT_PATH $AMP --lr-scheduler $LR_SCHEDULER --ema-decay-rate $EMA_DECAY_RATE --trial $TRIAL