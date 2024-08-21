# Adopted from: 
# - https://github.com/Alibaba-MIIL/ASL/blob/main/validate.py
# - https://github.com/SlongLiu/query2labels/blob/main/q2l_infer.py 

import argparse
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os
import json
from collections import OrderedDict

from pathlib import Path
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

from utils.defense import gen_mask_set
from utils.metrics import PerformanceMetrics
from utils.datasets import CocoDetection, split_dataset_gpu, TransformWrapper
from utils.training_helpers import Cutout

import sys
sys.path.append("packages/ASL/")
from packages.ASL.src.models import create_model
from packages.ASL.src.loss_functions.losses import AsymmetricLoss

sys.path.append("packages/query2labels/lib")
from packages.query2labels.lib.models.query2label import build_q2l

parser = argparse.ArgumentParser(description='Multi-Label PatchCleanser Certification')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', choices=["mscoco", "nuswide", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80)
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size (default: 32)')

# Model specifics
available_models = ['tresnet_l', 'Q2L-CvT_w24-384']
parser.add_argument('--model-name', choices=available_models, default='tresnet_l')
parser.add_argument('--model-path', default='./TRresNet_L_448_86.6.pth', type=str)
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')

# * Transformer
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--enc_layers', default=1, type=int, 
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=2, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=256, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=128, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=4, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--pre_norm', action='store_true')
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--backbone', choices=["resnet101", "CvT_w24"], default='CvT_w24', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                    help='keep the other self attention modules in transformer decoders, which will be removed default.')
parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                    help='keep the first self attention module in transformer decoders, which will be removed default.')
parser.add_argument('--keep_input_proj', action='store_true', 
                    help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

# Mask set specifics
parser.add_argument('--patch-size', default=64, type=int, help='patch size (default: 64)')
parser.add_argument('--mask-number-fr', default=6, type=int, help='mask number first round (default: 6)')
parser.add_argument('--mask-number-sr', default=6, type=int, help='mask number second round (default: 6)')

# GPU info for parallelism
parser.add_argument('--world-gpu-id', default=0, type=int, help='overall GPU id (default: 0)')
parser.add_argument('--total-num-gpu', default=1, type=int, help='total number of GPUs (default: 1)')

# Miscellaneous
parser.add_argument('--trial', default=1, type=int, help='trial (default: 1)')
parser.add_argument('--trial-type', default="vanilla", type=str, help='type of checkpoints used with the trial (default: vanilla/unmodified)')
parser.add_argument('--print-freq', '-p', default=64, type=int, help='print frequency (default: 64)') # is this even used anywhere????

def file_print(file_path, msg):
    with open(file_path, "a") as f:
        print(msg, flush=True, file=f) 

# Clean the state dict associated with ViT model
def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

# Load in the multi-label classifier
def load_model(args, is_ViT):
    args.do_bottleneck_head = False

    # Create model
    model = build_q2l(args).cuda() if is_ViT else create_model(args).cuda()

    # Setup depends on whether architecture is based on transformer or ResNet
    file_print(args.logging_file, f"setting up the model...{'ViT' if is_ViT else 'resnet'}")
    state = torch.load(args.model_path, map_location='cpu')
    if is_ViT:
        state_dict = clean_state_dict(state['state_dict'])
        classes_list = np.ones((80, 1))
    else:
        state_dict = state['model']
        classes_list = np.array(list(state['idx_to_class'].values()))
        args.do_bottleneck_head = False
    
    # Load model
    model.load_state_dict(state_dict, strict=True)
    args.rank = 0
    model = model.eval()

    # Cleanup intermediate variables
    del state
    del state_dict
    torch.cuda.empty_cache()
    file_print(args.logging_file, 'done\n')

    return model, args, classes_list

def main():
    args = parser.parse_args()
    args.batch_size = args.batch_size   

    is_ViT = False
    if args.model_name == "Q2L-CvT_w24-384":
        is_ViT = True

    # update Transformer parameters with pre-defined config file
    if args.config and is_ViT:
        with open(args.config, 'r') as f:
            cfg_dict = json.load(f)
        for k,v in cfg_dict.items():
            setattr(args, k, v)

        # Update parameters corresponding to this script
        args.image_size = args.img_size

    # Get GPU id
    world_gpu_id = args.world_gpu_id

    # Construct file path for saving metrics
    args.metadata = f"patch_{args.patch_size}_masknumfr_{args.mask_number_fr}_masknumsr_{args.mask_number_sr}"
    foldername = f"dump/greedy_cutout/{args.dataset_name}/{args.metadata}/{'ViT' if is_ViT else 'resnet'}/{todaystring}/trial_{args.trial}_{args.trial_type}/gpu_world_id_{args.world_gpu_id}/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging.txt"
    
    # Setup model
    model, args, classes_list = load_model(args, is_ViT)

    # Data loading code
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])


    instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    data_path_train = os.path.join(args.data, 'images/train2014')  # args.data
    train_dataset = CocoDetection(data_path_train,
                                instances_path_train,
                                transforms.Compose([
                                    TransformWrapper(transforms.Resize((args.image_size, args.image_size))),
                                    TransformWrapper(transforms.ToTensor()),
                                    # normalize, # no need, toTensor does normalization
                                ]))


    # Create GPU specific dataset
    gpu_train_dataset, start_idx, end_idx = split_dataset_gpu(train_dataset, args.batch_size, args.total_num_gpu, world_gpu_id)
    file_print(args.logging_file, "listing out info about this GPU process...")
    file_print(args.logging_file, f"length of gpu_val_dataset: {len(gpu_train_dataset)}\nbatch is currently at: {(int)(start_idx / args.batch_size)}\nstart_idx: {start_idx}\nend_idx: {end_idx}")

    train_loader = torch.utils.data.DataLoader(
        gpu_train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Create R-covering set of masks for both the first and second rounds
    im_size = [args.image_size, args.image_size]
    patch_size = [args.patch_size, args.patch_size]
    mask_number_fr = [args.mask_number_fr, args.mask_number_fr]
    mask_list_fr, mask_size_fr, mask_stride_fr = gen_mask_set(im_size, patch_size, mask_number_fr)

    mask_round_equal = False
    if(args.mask_number_fr != args.mask_number_sr):
        mask_number_sr = [args.mask_number_sr, args.mask_number_sr]
        mask_list_sr, mask_size_sr, mask_stride_sr = gen_mask_set(im_size, patch_size, mask_number_sr)
    else:
        mask_list_sr = mask_list_fr
        mask_round_equal = True

    greedy_cutout_generation(model, train_loader, classes_list, mask_list_fr, mask_list_sr, mask_round_equal, args)

# Essentially, we are going to work with batches of data and in this function we will loop over masks. For each mask, we input the size 64 batch into the model
# and then ONE-BY-ONE (i.e., the ASL loss function only returns a single value) compute the loss for each of these images. Have an array which for each of the 64
# images maintains the current mask index with the worst loss. Once done, we return this array (and optionally the corresponding loss values) back to the caller

# In the main function, after doing this once we have the worst single mask for each of the 64 images. Now, we find the worst double mask. Specifically, we consider
# each single mask one at a time and determine which set of the 64 images have this as their worst mask. We then call this function on the subset of images (with the first round mask applied).
# Note that the function should still work, although now the subset of data is of size <= 64. After computing worst mask index + loss, return to main. Main should have the image indices corresponding
# to the subset, so we can update a global array with the worst case two-mask pairs. Do this for each possible first round mask (i.e., using a loop) and then we will have a size 
# 64 array with the first + second round worst case masks and loss. Finally, we convert this to a dictionary with the file name being the key. 
def mask_set_optimum_loss(model, im_data, targets_data, mask_list, criterion, args):
    data_len = len(im_data)
    mask_optima = np.zeros(data_len,)
    losses = np.zeros(data_len,)

    # Iterate through the set of individual masks
    for i, mask in enumerate(mask_list):
        mask = mask.reshape(1, 1, *mask.shape).to(args.rank)            
        masked_im = torch.where(mask, im_data.to(args.rank), torch.tensor(0.0).to(args.rank))
            
        # Compute output
        with torch.no_grad():
            output = model(masked_im).to(args.rank)

        # Determine the loss for each image, and determine if the optimum mask location has changed - check validation loss here and make sure it is about 67!!! (train loss is about 20)
        for j in range(data_len):
            loss = criterion(torch.Tensor(output[j]).to(args.rank), torch.Tensor(targets_data[j]).to(args.rank))

            if loss.item() > losses[j]: 
                losses[j] = loss
                mask_optima[j] = i

    return mask_optima, losses

def greedy_cutout_generation(model, train_loader, classes_list, mask_list_fr, mask_list_sr, mask_round_equal, args):
    file_print(args.logging_file, "starting actual validation...")

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)

    preds = []
    targets = []
    num_masks_fr, num_masks_sr = len(mask_list_fr), len(mask_list_sr)
    num_classes = len(classes_list)

    # Create a dictionary to store all of the "greedy cutout" mask indices corresponding to each individual image
    greedy_cutout_dict = {}

    # target shape: [batch_size, object_size_channels, number_classes]
    for batch_index, batch in enumerate(train_loader):
        
        input_data = batch[0]
        target = batch[1]
        paths = batch[2]

        file_print(args.logging_file, f'\nBatch: [{batch_index}/{len(train_loader)}]')

        # torch.max returns (values, indices), additionally squeezes along the dimension dim
        target = target.max(dim=1)[0]
        im = input_data
        target = target.cpu().numpy()

        # Determine the first round masks which result in the worst loss
        fr_mask_optima, _ = mask_set_optimum_loss(model, im, target, mask_list_fr, criterion, args)
        file_print(args.logging_file, f'first round masks found...')

        # For each single mask, find the subset of images from the batch which have this as their 
        # optimum mask; then apply the second round of masking to find the double mask pair which
        # results in the worst loss
        file_print(args.logging_file, f'starting search for second round masks...')
        cumulative_indices = []
        sr_mask_optima = np.zeros(len(im),)
        losses = np.zeros(len(im),)
        for i, mask in enumerate(mask_list_fr):
            if (i % 10 == 0): file_print(args.logging_file, f'computing images associated with first round mask {i} out of {len(mask_list_fr)}')
            subset_indices = np.nonzero(fr_mask_optima == i)[0]
            if len(subset_indices) == 0: continue   # If this mask is not the optimum for any image in the batch, we can skip it
            cumulative_indices.extend(subset_indices)

            subset_images = im[subset_indices]
            subset_targets = target[subset_indices]

            # Apply the optimum first round masks to the image subset
            mask = mask.reshape(1, 1, *mask.shape).to(args.rank)            
            masked_subset_images = torch.where(mask, subset_images.to(args.rank), torch.tensor(0.0).to(args.rank))

            # Determine second round masks which result in the worst loss for this image subset
            subset_sr_mask_optima, subset_losses = mask_set_optimum_loss(model, masked_subset_images, subset_targets, mask_list_fr, criterion, args)

            # Update the global arrays corresponding to second round mask optima
            sr_mask_optima[subset_indices] = subset_sr_mask_optima
            losses[subset_indices] = subset_losses

        cumulative_indices = sorted(cumulative_indices)
        assert len(cumulative_indices) == len(im)
        for i in range(len(cumulative_indices)):
            assert cumulative_indices[i] == i

        # Update the global dictionary
        file_print(args.logging_file, f'second round masks found...')
        greedy_cutout_dict = greedy_cutout_dict | {paths[i]: {"fr_mask": fr_mask_optima[i], "sr_mask": sr_mask_optima[i], "loss": losses[i]} for i in range (len(im))}

        # Save the current state of the global dictionary every batch
        with open(args.save_dir + f"greedy_cutout_dict.json", "w") as f:
            json.dump(greedy_cutout_dict, f, indent=2)

    # Save the final state of the global dictionary along with metadata
    metadata = {"image_size": args.image_size, "patch_size": args.patch_size, "mask_number_fr": args.mask_number_fr, "mask_number_sr": args.mask_number_sr}
    greedy_cutout_dict = {"metadata": metadata, "data": greedy_cutout_dict}
    with open(args.save_dir + f"greedy_cutout_dict.json", "w") as f:
        json.dump(greedy_cutout_dict, f, indent=2)

    return

if __name__ == '__main__':
    main()
