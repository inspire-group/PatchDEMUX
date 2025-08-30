# Adopted from: 
# - https://github.com/Alibaba-MIIL/ASL/blob/main/validate.py
# - https://github.com/SlongLiu/query2labels/blob/main/q2l_infer.py 

import argparse
import numpy as np
import torch
import os
import json

from pathlib import Path
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

# Add parent directory to path so we can import from utils
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from defenses.patchcleanser.pc_utils import gen_mask_set
from utils.datasets import split_dataset_gpu
from utils.common import file_print, load_model, load_train_dataset

sys.path.append(os.path.join(parent_dir, "packages/ASL/"))
from packages.ASL.src.loss_functions.losses import AsymmetricLoss

parser = argparse.ArgumentParser(description='Greedy cutout generation')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', choices=["mscoco", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80, type=int, help='number of classes (default: 80)')
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

    # Construct file path for saving metrics
    args.metadata = f"patch_{args.patch_size}_masknum_{args.mask_number}"
    foldername = f"dump/greedy_cutout/{args.dataset_name}/{args.metadata}/{'ViT' if is_ViT else 'resnet'}/{todaystring}/trial_{args.trial}_{args.trial_type}/gpu_world_id_{args.world_gpu_id}/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging.txt"
    
    # Setup model
    model, args = load_model(args, is_ViT)

    # Data loading code - use the common function
    train_dataset = load_train_dataset(args)

    # Create GPU specific dataset
    gpu_train_dataset, start_idx, end_idx = split_dataset_gpu(train_dataset, args.batch_size, args.total_num_gpu, world_gpu_id)
    file_print(args.logging_file, "listing out info about this GPU process...")
    file_print(args.logging_file, f"length of gpu_val_dataset: {len(gpu_train_dataset)}\nbatch is currently at: {(int)(start_idx / args.batch_size)}\nstart_idx: {start_idx}\nend_idx: {end_idx}")

    train_loader = torch.utils.data.DataLoader(
        gpu_train_dataset, batch_size=args.batch_size, shuffle=False,   # No need to shuffle, we generate optimal masks in order
        num_workers=args.workers, pin_memory=True)

    # Create R-covering set of masks
    im_size = [args.image_size, args.image_size]
    patch_size = [args.patch_size, args.patch_size]
    mask_number = [args.mask_number, args.mask_number]
    mask_list, mask_size, mask_stride = gen_mask_set(im_size, patch_size, mask_number)

    greedy_cutout_generation(model, train_loader, mask_list, args)

# Determines the mask which results in the worst loss for a batch of images
def mask_set_optimum_loss(model, im_data, targets_data, mask_list, criterion):
    data_len = len(im_data)
    mask_optima = np.zeros(data_len,)
    losses = np.zeros(data_len,)

    # Iterate through the set of individual masks
    for i, mask in enumerate(mask_list):
        mask = mask.reshape(1, 1, *mask.shape).cuda()            
        masked_im = torch.where(mask, im_data.cuda(), torch.tensor(0.0).cuda())
            
        # Compute output
        with torch.no_grad():
            output = model(masked_im).cuda()

        # Determine the loss for each image, and determine if the optimum mask location has changed
        for j in range(data_len):
            loss = criterion(torch.Tensor(output[j]).cuda(), torch.Tensor(targets_data[j]).cuda())

            if loss.item() > losses[j]: 
                losses[j] = loss
                mask_optima[j] = i

    return mask_optima, losses

def greedy_cutout_generation(model, train_loader, mask_list, args):
    file_print(args.logging_file, "starting greedy cutout generation...")

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    
    # Create a dictionary to store all of the "greedy cutout" mask indices corresponding to each individual image
    greedy_cutout_dict = {}

    # target shape: [batch_size, object_size_channels, number_classes]
    for batch_index, batch in enumerate(train_loader):
        
        input_data = batch[0]
        target = batch[1]
        paths = batch[2]

        file_print(args.logging_file, f'\nBatch: [{batch_index}/{len(train_loader)}]')

        # torch.max returns (values, indices), additionally squeezes along the dimension dim
        if args.dataset_name == "mscoco":
            target = target.max(dim=1)[0]

        im = input_data
        target = target.cpu().numpy()

        # Determine the first round masks which result in the worst loss
        fr_mask_optima, _ = mask_set_optimum_loss(model, im, target, mask_list, criterion)
        file_print(args.logging_file, f'first round masks found...')

        # For each single mask i, find the subset of images from the batch which have mask i as their 
        # optimum mask; then apply the second round of masking to find the double mask pair which
        # results in the worst loss
        file_print(args.logging_file, f'starting search for second round masks...')
        cumulative_im_indices = []
        sr_mask_optima = np.zeros(len(im),)
        losses = np.zeros(len(im),)
        for i, mask in enumerate(mask_list):
            if (i % 10 == 0): file_print(args.logging_file, f'computing images associated with first round mask {i} out of {len(mask_list)}')
            subset_im_indices = np.nonzero(fr_mask_optima == i)[0]
            if len(subset_im_indices) == 0: continue   # If this mask is not the optimum for any image in the batch, we can skip it
            cumulative_im_indices.extend(subset_im_indices)

            subset_images = im[subset_im_indices]
            subset_targets = target[subset_im_indices]

            # Apply the optimum first round masks to the image subset
            mask = mask.reshape(1, 1, *mask.shape).cuda()            
            masked_subset_images = torch.where(mask, subset_images.cuda(), torch.tensor(0.0).cuda())

            # Determine second round masks which result in the worst loss for this image subset
            subset_sr_mask_optima, subset_losses = mask_set_optimum_loss(model, masked_subset_images, subset_targets, mask_list, criterion)

            # Update the global arrays corresponding to second round mask optima
            sr_mask_optima[subset_im_indices] = subset_sr_mask_optima
            losses[subset_im_indices] = subset_losses

        cumulative_im_indices = sorted(cumulative_im_indices)
        assert len(cumulative_im_indices) == len(im)
        for i in range(len(cumulative_im_indices)):
            assert cumulative_im_indices[i] == i

        # Update the global dictionary
        file_print(args.logging_file, f'second round masks found...')
        greedy_cutout_dict = greedy_cutout_dict | {paths[i]: {"fr_mask": fr_mask_optima[i], "sr_mask": sr_mask_optima[i], "loss": losses[i]} for i in range (len(im))}

        # Save the current state of the global dictionary every batch
        with open(args.save_dir + f"greedy_cutout_dict.json", "w") as f:
            json.dump(greedy_cutout_dict, f, indent=2)

    # Save the final state of the global dictionary along with metadata
    metadata = {"image_size": args.image_size, "patch_size": args.patch_size, "mask_number": args.mask_number}
    greedy_cutout_dict = {"metadata": metadata, "data": greedy_cutout_dict}
    with open(args.save_dir + f"greedy_cutout_dict.json", "w") as f:
        json.dump(greedy_cutout_dict, f, indent=2)

    return

if __name__ == '__main__':
    main()
