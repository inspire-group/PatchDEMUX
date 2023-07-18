#!/bin/bash

#SBATCH --job-name=train_cutout_multilabel_patchcleanser
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --nodes=1                # node count (number of different machine)
#SBATCH --ntasks-per-node=1      # number of tasks per-node (choose equal to gpus) [make sure ntasks and ngpus are equal]
#SBATCH --gpus-per-node=1        # gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=djacob@princeton.edu

# Start the conda environment
module purge
module load anaconda3/2022.10
source activate torch-env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/djacob/.conda/envs/torch-env/lib 

# File/path locations
DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/coco/"
MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/MS_COCO_TRresNet_L_448_86.6.pth"

# Model parameters
BATCH_SIZE=64

# Training specifics
LR=0.00005
CUTOUT_SIZE=224
AMP="--amp"
LR_SCHEDULER="onecyclelr"
EMA_DECAY_RATE=0.9997

# Misc.
TRIAL=1

python ml_ASL_cutout_train.py $DATA_DIR --batch-size $BATCH_SIZE --model-path $MODEL_PATH --lr $LR --cutout-size $CUTOUT_SIZE $AMP --lr-scheduler $LR_SCHEDULER --ema-decay-rate $EMA_DECAY_RATE --trial $TRIAL