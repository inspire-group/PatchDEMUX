#!/bin/bash

#SBATCH --job-name=certify_multilabel_patchcleanser_12mask
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --nodes=1                # node count (number of different machine)
#SBATCH --ntasks-per-node=4      # number of tasks per-node (choose equal to gpus) [make sure ntasks and ngpus are equal]
#SBATCH --gpus-per-node=4        # gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=50:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-4
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=djacob@princeton.edu

#export WORLD_SIZE=1 # set it equal to total number of gpus across all nodes (=total number of tasks across all nodes)

# Borrowed from https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
#export MASTER_PORT=8130
#echo "NODELIST="${SLURM_JOB_NODELIST}
#echo "MASTER_ADDR="$MASTER_ADDR
#echo $(date '+%d/%m/%Y %H:%M:%S')


# Start the conda environment
module purge
module load anaconda3/2022.10
source activate torch-env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/djacob/.conda/envs/torch-env/lib 

# File/path locations
DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/coco/"
MODEL_PATH="ASL/checkpoints/MS_COCO_TRresNet_L_448_86.6.pth"

# Model parameters
BATCH_SIZE=64
PATCH_SIZE=64
MASK_NUMBER=12

# GPU info
NODE_NUMBER=$SLURM_ARRAY_TASK_ID
TOTAL_NUM_GPU=20
GPU_PER_NODE=4

# Misc.
TRIAL=1

for i in {0..3}; do
CUDA_VISIBLE_DEVICES=$i python ml_pc_certification.py $DATA_DIR --batch-size $BATCH_SIZE --model-path $MODEL_PATH --patch-size $PATCH_SIZE --mask-number $MASK_NUMBER --node-number $NODE_NUMBER --total-num-gpu $TOTAL_NUM_GPU --gpu-per-node $GPU_PER_NODE --trial $TRIAL &
done
wait