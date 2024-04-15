#!/bin/bash

#SBATCH --job-name=cache_thresholding
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file

#SBATCH --nodes=1                # node count (number of different machine)
#SBATCH --ntasks-per-node=1      # number of tasks per-node (choose equal to gpus) [make sure ntasks and ngpus are equal]
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=djacob@princeton.edu

# Start the conda environment
module purge
module load anaconda3/2022.10
source activate torch-env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/djacob/.conda/envs/torch-env/lib 

THRE_ARR=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99)
#THRE_ARR=(0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05)

# Model security parameters
ATTACKER_TYPE="FP_attacker"
PATCH_SIZE=64
MASK_NUMBER=6
PATCH_CLEANSER="--no-patchcleanser"

# File/path locations
CACHED_OUTPUTS=/scratch/gpfs/djacob/multi-label-patchcleanser/cached_outputs/mscoco/patch_64_masknumfr_6_masknumsr_6/resnet/03-23-2024/trial_2_vanilla/cached_outputs
#CACHED_OUTPUTS=/scratch/gpfs/djacob/multi-label-patchcleanser/cached_outputs/mscoco/patch_64_masknumfr_6_masknumsr_6/ViT/03-20-2024/trial_2_vanilla/cached_outputs
#CACHED_OUTPUTS=/scratch/gpfs/djacob/multi-label-patchcleanser/cached_outputs/mscoco/patch_64_masknumfr_6_masknumsr_6/resnet/04-13-2024/trial_2_cutout_224_onecyclelr_mixedprec_ema_epoch9/cached_outputs

for THRE in ${THRE_ARR[@]};
do
    python ml_pc_cache_certification.py --cache-location $CACHED_OUTPUTS --attacker-type $ATTACKER_TYPE --thre $THRE --patch-size $PATCH_SIZE --mask-number-fr $MASK_NUMBER
    #python ml_pc_cache_clean_images.py $PATCH_CLEANSER --cache-location $CACHED_OUTPUTS --thre $THRE --patch-size $PATCH_SIZE --mask-number-fr $MASK_NUMBER
done