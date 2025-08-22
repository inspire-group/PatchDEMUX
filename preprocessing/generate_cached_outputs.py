# Adopted from: 
# - https://github.com/Alibaba-MIIL/ASL/blob/main/validate.py
# - https://github.com/SlongLiu/query2labels/blob/main/q2l_infer.py 

import argparse
import numpy as np
import torch
import os

from pathlib import Path
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

# Add parent directory to path so we can import from utils
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from defenses.patchcleanser.pc_utils import gen_mask_set, cache_masked_outputs
from utils.datasets import split_dataset_gpu
from utils.common import file_print, load_model, load_eval_dataset

parser = argparse.ArgumentParser(description='Generate cached outputs for sweeping model thresholds')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', choices=["mscoco", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size (default: 32)')

# Model specifics
available_models = ['tresnet_l', 'Q2L-CvT_w24-384']
parser.add_argument('--model-name', choices=available_models, default='tresnet_l')
parser.add_argument('--model-path', default='./TRresNet_L_448_86.6.pth', type=str)
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model. default is False. ')

# * Transformer config file (optional, required for ViT models)
parser.add_argument('--config', type=str, default=None, help='config file containing all ViT parameters')

# Mask set specifics
parser.add_argument('--patch-size', default=64, type=int, help='patch size (default: 64)')
parser.add_argument('--mask-number', default=6, type=int, help='mask number (default: 6)')

# GPU info for parallelism
parser.add_argument('--world-gpu-id', default=0, type=int, help='overall GPU id (default: 0)')
parser.add_argument('--total-num-gpu', default=1, type=int, help='total number of GPUs (default: 1)')

# Miscellaneous
parser.add_argument('--trial', default=1, type=int, help='trial (default: 1)')
parser.add_argument('--trial-type', default="vanilla", type=str, help='type of checkpoints used with the trial (default: vanilla/unmodified)')
parser.add_argument('--print-freq', '-p', default=64, type=int, help='print frequency (default: 64)')
parser.add_argument('--cache-dir', default='/scratch/gpfs/djacob/multi-label-patchcleanser', type=str, help='directory for cached outputs (default: /scratch/gpfs/djacob/multi-label-patchcleanser)')

def main():
    args = parser.parse_args()
    args.batch_size = args.batch_size   

    if args.model_name == "Q2L-CvT_w24-384":
        is_ViT = True
        if args.config is None:
            raise ValueError("--config parameter is required when using ViT models (Q2L-CvT_w24-384)")
    else:
        is_ViT = False

    # Get GPU id
    world_gpu_id = args.world_gpu_id

    # Construct file path for saving cached outputs
    foldername = os.path.join(args.cache_dir, f"cached_outputs/{args.dataset_name}/{'ViT' if is_ViT else 'resnet'}/patch_{args.patch_size}_masknum_{args.mask_number}/{todaystring}/trial_{args.trial}_{args.trial_type}/gpu_world_id_{args.world_gpu_id}/")
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging.txt"
    
    # Setup model
    model, args = load_model(args, is_ViT)

    # Data loading code
    val_dataset = load_eval_dataset(args)

    # Create GPU specific dataset
    gpu_val_dataset, start_idx, end_idx = split_dataset_gpu(val_dataset, args.batch_size, args.total_num_gpu, world_gpu_id)
    file_print(args.logging_file, "listing out info about this GPU process...")
    file_print(args.logging_file, f"length of gpu_val_dataset: {len(gpu_val_dataset)}\nbatch is currently at: {(int)(start_idx / args.batch_size)}\nstart_idx: {start_idx}\nend_idx: {end_idx}")

    val_loader = torch.utils.data.DataLoader(
        gpu_val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Create R-covering set of masks
    im_size = [args.image_size, args.image_size]
    patch_size = [args.patch_size, args.patch_size]
    mask_number = [args.mask_number, args.mask_number]
    mask_list, mask_size, mask_stride = gen_mask_set(im_size, patch_size, mask_number)

    generate_cached_outputs(model, val_loader, mask_list, args)

def generate_cached_outputs(model, val_loader, mask_list, args):
    file_print(args.logging_file, "starting output generation...")

    Sig = torch.nn.Sigmoid()

    # target shape: [batch_size, object_size_channels, number_classes]
    for batch_index, batch in enumerate(val_loader):
        input_data = batch[0]
        target = batch[1]
        file_print(args.logging_file, f'\nBatch: [{batch_index}/{len(val_loader)}]')
        output_dict = {}

        # torch.max returns (values, indices), additionally squeezes along the dimension dim
        if args.dataset_name == "mscoco":
            target = target.max(dim=1)[0]

        im = input_data
        target = target.cpu().numpy()
        output_dict["target"] = target

        # Compute output for the undefended model
        with torch.no_grad():
            clean_output = Sig(model(im.cuda()).cuda()).cpu().numpy()
        output_dict["clean_output"] = clean_output

        # When using PatchCleanser as the single-label CDPA backbone, we can cache double masked outputs for each image
        # in the dataset and prevent redundant computation when sweeping over model thresholds
        output_dict["masked_output"] = cache_masked_outputs(model, mask_list, im, args, lambda msg: file_print(args.logging_file, msg))

        # Save outputs for this batch as numpy arrays
        np.savez(args.save_dir + f"gpu_{args.world_gpu_id}_batch_{batch_index}_outputs", **output_dict)

    return

if __name__ == '__main__':
    main()
