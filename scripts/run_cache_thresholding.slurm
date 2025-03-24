#!/bin/bash

#SBATCH --job-name=cache_thresholding
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
#CACHED_OUTPUTS=/scratch/gpfs/djacob/multi-label-patchcleanser/cached_outputs/mscoco/patch_64_masknumfr_6_masknumsr_6/resnet/03-23-2024/trial_2_vanilla/cached_outputs
#CACHED_OUTPUTS=/scratch/gpfs/djacob/multi-label-patchcleanser/cached_outputs/mscoco/patch_55_masknumfr_6_masknumsr_6/ViT/08-30-2024/trial_2_greedycutout_patch_55_masknum_6_training_onecyclelr_mixedprec_ema_epoch0/cached_outputs
#CACHED_OUTPUTS=/scratch/gpfs/djacob/multi-label-patchcleanser/cached_outputs/mscoco/patch_32_masknumfr_6_masknumsr_6/resnet/08-29-2024/trial_2_vanilla/cached_outputs

#CACHED_OUTPUTS=/scratch/gpfs/djacob/multi-label-patchcleanser/cached_outputs/mscoco/ViT/patch_55_masknumfr_6_masknumsr_6/08-30-2024/trial_2_greedycutout_patch_55_masknum_6_training_onecyclelr_mixedprec_ema_epoch0/cached_outputs
CACHED_OUTPUTS=/scratch/gpfs/djacob/multi-label-patchcleanser/cached_outputs/pascalvoc/ViT/patch_55_masknumfr_6_masknumsr_6/03-21-2025/trial_1_vanilla/cached_outputs

THRE_ARR_STD=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99)
THRE_ARR_LOW1=(0.00005 0.0001 0.0005 0.001 0.005)
THRE_ARR_LOW2=(0.01 0.05)
THRE_ARR_MID=(0.52 0.54 0.56 0.58 0.62 0.64 0.66 0.68 0.72 0.74 0.76 0.78 0.82 0.84 0.86 0.88)
THRE_ARR_HIGH=(0.999 0.9999 0.99999)

THRE_COMBINED=("${THRE_ARR_STD[@]}" "${THRE_ARR_LOW1[@]}" "${THRE_ARR_LOW2[@]}" "${THRE_ARR_MID[@]}" "${THRE_ARR_HIGH[@]}")

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

# Type of run
RUN_TYPE="undefended"

if [[ "$RUN_TYPE" == "certification" ]]; then
    ATTACKER_TYPE="worst_case"
elif [[ "$RUN_TYPE" == "defended" ]]; then
    PATCH_CLEANSER="--patchcleanser"
elif [[ "$RUN_TYPE" == "undefended" ]]; then
    PATCH_CLEANSER="--no-patchcleanser"
fi

# Run evaluations over different thresholds
for THRE in ${THRE_COMBINED[@]};
do
    if [[ "$RUN_TYPE" == "certification" ]]; then
        python ml_pc_cache_certification.py --cache-location $CACHED_OUTPUTS --dataset-name $DATASET_NAME --num-classes $NUM_CLASSES --image-size $IMAGE_SIZE --attacker-type $ATTACKER_TYPE --thre $THRE --patch-size $PATCH_SIZE --mask-number-fr $MASK_NUMBER
    else
        python ml_pc_cache_clean_images.py $PATCH_CLEANSER --cache-location $CACHED_OUTPUTS --dataset-name $DATASET_NAME --num-classes $NUM_CLASSES --image-size $IMAGE_SIZE --thre $THRE --patch-size $PATCH_SIZE --mask-number-fr $MASK_NUMBER
    fi
done

# Writing bash script: https://stackoverflow.com/questions/49516592/easily-creating-a-bash-script-in-python
# for i in {0..7}
# do
#   mv gpu_world_id_$i/*.npz cached_outputs/
# done
# then remove all the created folders....