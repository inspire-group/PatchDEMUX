#!/bin/bash

#SBATCH --job-name=certify_multilabel_patchcleanser_6mask_and_12mask
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --nodes=1                # node count (number of different machine)
#SBATCH --ntasks-per-node=1      # number of tasks per-node (choose equal to gpus) [make sure ntasks and ngpus are equal]
#SBATCH --gpus-per-node=1        # gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-7
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
BATCH_SIZE=64
PATCH_SIZE=64
MASK_NUMBER_FR=6
MASK_NUMBER_SR=6
#THRE_ARR=(0.99 0.995)
THRE_ARR=(0.8)
ATTACKER_TYPE="FP_attacker"

# GPU info
WORLD_GPU_ID=$SLURM_ARRAY_TASK_ID
TOTAL_NUM_GPU=8

# Misc.
TRIAL=2
#TRIAL_TYPE="cutout_224_onecyclelr_mixedprec_ema"
TRIAL_TYPE="vanilla"
EPOCH=0

# File/path locations
DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/coco/"
#MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/${TRIAL_TYPE}/07-13-2023/trial_1_FULL_RUN/epoch_${EPOCH}/ema-model-epoch-${EPOCH}.pth"
MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/MS_COCO_TRresNet_L_448_86.6.pth"

for THRE in ${THRE_ARR[@]};
do
    #python ml_pc_certification.py $DATA_DIR --batch-size $BATCH_SIZE --model-path $MODEL_PATH --thre $THRE --patch-size $PATCH_SIZE --mask-number-fr $MASK_NUMBER_FR --mask-number-sr $MASK_NUMBER_SR --world-gpu-id $WORLD_GPU_ID --total-num-gpu $TOTAL_NUM_GPU --trial $TRIAL --trial-type $TRIAL_TYPE
    python ml_pc_certification_residual_robustness.py $DATA_DIR --batch-size $BATCH_SIZE --model-path $MODEL_PATH --thre $THRE --attacker-type $ATTACKER_TYPE --patch-size $PATCH_SIZE --mask-number-fr $MASK_NUMBER_FR --mask-number-sr $MASK_NUMBER_SR --world-gpu-id $WORLD_GPU_ID --total-num-gpu $TOTAL_NUM_GPU --trial $TRIAL --trial-type $TRIAL_TYPE
done