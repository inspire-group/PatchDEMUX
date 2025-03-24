#!/bin/bash

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
#SBATCH --mail-user=djacob@princeton.edu

# Start the conda environment
module purge
module load anaconda3/2022.10
source activate torch-env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/djacob/.conda/envs/torch-env/lib 

# File/path locations
#CACHED_OUTPUTS=/scratch/gpfs/djacob/multi-label-patchcleanser/cached_outputs/mscoco/ViT/patch_55_masknumfr_6_masknumsr_6/08-30-2024/trial_2_greedycutout_patch_55_masknum_6_training_onecyclelr_mixedprec_ema_epoch0/cached_outputs
CACHED_OUTPUTS=/scratch/gpfs/djacob/multi-label-patchcleanser/cached_outputs/pascalvoc/ViT/patch_55_masknumfr_6_masknumsr_6/03-21-2025/trial_1_vanilla/cached_outputs

# Interpolation parameters
#FIRST_THRE_ARR=()
#FIRST_RECALL_ARR=()
#SECOND_THRE_ARR=()
#SECOND_RECALL_ARR=()

#undefended
FIRST_THRE_ARR=(0.999 0.999 0.96)
FIRST_RECALL_ARR=(16.84 16.84 74.12)
SECOND_THRE_ARR=(0.99 0.99 0.95)
SECOND_RECALL_ARR=(53.29 53.29 76.81)

# no hist
#FIRST_THRE_ARR=(0.99 0.91 0.56)
#FIRST_RECALL_ARR=(21.25 49.83 74.25)
#SECOND_THRE_ARR=(0.98 0.90 0.54)
#SECOND_RECALL_ARR=(30.18 51.35 75.18)

TARGET_RECALL_ARR=(25.0 50.0 75.0)

# Model parameters
ARCH="ViT"
DATASET_NAME="pascalvoc"
NUM_CLASSES=20

if [[ "$ARCH" == "ViT" ]]; then
    IMAGE_SIZE=384
    PATCH_SIZE=55
    #PATCH_SIZE=218
elif [[ "$ARCH" == "resnet" ]]; then
    IMAGE_SIZE=448
    PATCH_SIZE=64
    #PATCH_SIZE=64
fi

MASK_NUMBER=6

# Misc.
TRIAL=2

# Type of run
RUN_TYPE="undefended"
HIST="no_hist"

#RUN_TYPE="defended"
#HIST="no_hist"

#RUN_TYPE="certification"
#HIST="no_hist"

#RUN_TYPE="certification"
#HIST="hist"

if [ "$RUN_TYPE" == "certification" ] && [ "$HIST" == "no_hist" ]; then
    ATTACKER_TYPE="none"
elif [ "$RUN_TYPE" == "certification" ] && [ "$HIST" == "hist" ]; then
    ATTACKER_TYPE="worst_case"
elif [[ "$RUN_TYPE" == "defended" ]]; then
    PATCH_CLEANSER="--patchcleanser"
elif [[ "$RUN_TYPE" == "undefended" ]]; then
    PATCH_CLEANSER="--no-patchcleanser"
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
        python ml_pc_cache_certification_interpolator.py --cache-location $CACHED_OUTPUTS --dataset-name $DATASET_NAME --num-classes $NUM_CLASSES --image-size $IMAGE_SIZE --attacker-type $ATTACKER_TYPE --first-thre $FIRST_THRE --first-recall $FIRST_RECALL --second-thre $SECOND_THRE --second-recall $SECOND_RECALL --target-recall $TARGET_RECALL --patch-size $PATCH_SIZE --mask-number-fr $MASK_NUMBER --trial $TRIAL
    else
        python ml_pc_cache_clean_images_interpolator.py $PATCH_CLEANSER --cache-location $CACHED_OUTPUTS --dataset-name $DATASET_NAME --num-classes $NUM_CLASSES --image-size $IMAGE_SIZE --first-thre $FIRST_THRE --first-recall $FIRST_RECALL --second-thre $SECOND_THRE --second-recall $SECOND_RECALL --target-recall $TARGET_RECALL --patch-size $PATCH_SIZE --mask-number-fr $MASK_NUMBER --trial $TRIAL
    fi
done