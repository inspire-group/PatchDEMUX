# Adopted from: 
# - https://github.com/Alibaba-MIIL/ASL/blob/main/validate.py
# - https://github.com/SlongLiu/query2labels/blob/main/q2l_infer.py 

# NOTE: Currently altered to try and check inference time. The val dataset is altered to be a random subset

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
from utils.datasets import CocoDetection, VOCDetection, split_dataset_gpu, TransformWrapper

import sys
sys.path.append("packages/ASL/")
from packages.ASL.src.models import create_model

sys.path.append("packages/query2labels/lib")
from packages.query2labels.lib.models.query2label import build_q2l

parser = argparse.ArgumentParser(description='Multi-Label PatchCleanser Certification')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', choices=["mscoco", "nuswide", "pascalvoc"], default="mscoco")
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
parser.add_argument('--attacker-type', choices=["worst_case", "FN_attacker", "FP_attacker"], default="worst_case")

# * Transformer
# note that if we have a config file, it might not be eeded to have all these parser args as well...
# looking at the config file, looks like the args there are the exact same...delete these prolly
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
parser.add_argument('--print-freq', '-p', default=64, type=int, help='print frequency (default: 64)')

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
    model = build_q2l(args) if is_ViT else create_model(args)

    # Setup depends on whether architecture is based on transformer or ResNet
    file_print(args.logging_file, f"setting up the model...{'ViT' if is_ViT else 'resnet'}")
    state = torch.load(args.model_path, map_location='cpu')
    if is_ViT:
        state_dict = clean_state_dict(state['state_dict'])
        classes_list = np.ones((args.num_classes, 1))
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

    return model.cuda(), args, classes_list

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
    foldername = f"/scratch/gpfs/djacob/multi-label-patchcleanser/cached_outputs/{args.dataset_name}/{'ViT' if is_ViT else 'resnet'}/patch_{args.patch_size}_masknumfr_{args.mask_number_fr}_masknumsr_{args.mask_number_sr}/{todaystring}/trial_{args.trial}_{args.trial_type}/gpu_world_id_{args.world_gpu_id}/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging.txt"
    
    # Setup model
    model, args, classes_list = load_model(args, is_ViT)

    # Data loading code
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])

    if args.dataset_name == "mscoco":
        instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
        data_path = os.path.join(args.data, 'images/val2014')
        val_dataset = CocoDetection(data_path,
                                instances_path,
                                transforms.Compose([
                                    TransformWrapper(transforms.Resize((args.image_size, args.image_size))),
                                    TransformWrapper(transforms.ToTensor()),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    elif args.dataset_name == "pascalvoc":
        data_path_val = os.path.join(args.data, 'test')
        val_dataset = VOCDetection(root = data_path_val,
                                    year = "2007",
                                    image_set = "test",
                                    transform = transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        # normalize, # no need, toTensor does normalization
                                    ]))


    # Create GPU specific dataset
    gpu_val_dataset, start_idx, end_idx = split_dataset_gpu(val_dataset, args.batch_size, args.total_num_gpu, world_gpu_id)
    file_print(args.logging_file, "listing out info about this GPU process...")
    file_print(args.logging_file, f"length of gpu_val_dataset: {len(gpu_val_dataset)}\nbatch is currently at: {(int)(start_idx / args.batch_size)}\nstart_idx: {start_idx}\nend_idx: {end_idx}")
    # rng = np.random.default_rng(123)
    # gpu_val_dataset = torch.utils.data.Subset(val_dataset, rng.choice(40137, 100))

    val_loader = torch.utils.data.DataLoader(
        gpu_val_dataset, batch_size=args.batch_size, shuffle=False,
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

    validate_multi(model, val_loader, classes_list, mask_list_fr, mask_list_sr, mask_round_equal, args)

def validate_multi(model, val_loader, classes_list, mask_list_fr, mask_list_sr, mask_round_equal, args):
    file_print(args.logging_file, "starting output generation...")

    Sig = torch.nn.Sigmoid()

    preds = []
    targets = []
    num_masks_fr, num_masks_sr = len(mask_list_fr), len(mask_list_sr)
    num_classes = len(classes_list)

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

        # compute clean output with no masks
        with torch.no_grad():
            clean_output = Sig(model(im.to(args.rank)).to(args.rank)).cpu().numpy()
        output_dict["clean_output"] = clean_output

        # Allow double counting of mask pairs to facilitate histogram analysis later 
        all_preds = np.zeros([im.shape[0], num_masks_fr * num_masks_sr, num_classes]) - 1
        for i, mask1 in enumerate(mask_list_fr):
            mask1 = mask1.reshape(1, 1, *mask1.shape).to(args.rank)            

            file_print(args.logging_file, f"Certification is {(i / num_masks_fr) * 100:0.2f}% complete!")
            for j, mask2 in enumerate(mask_list_sr):
                mask2 = mask2.reshape(1, 1, *mask2.shape).to(args.rank)
                masked_im = torch.where(torch.logical_and(mask1, mask2), im.to(args.rank), torch.tensor(0.0).to(args.rank))

                # compute output
                with torch.no_grad():
                    output = Sig(model(masked_im).to(args.rank)).cpu()

                all_preds[:, i * num_masks_sr + j] = output.cpu().numpy()

        output_dict["masked_output"] = all_preds

        # Save outputs for this batch as numpy arrays
        np.savez(args.save_dir + f"gpu_{args.world_gpu_id}_batch_{batch_index}_outputs", **output_dict)

        # For testing purposes
        #np.savez(f"gpu_{args.world_gpu_id}_batch_{batch_index}_outputs", **output_dict)

        # could have something here where a file is shared among all processes to track progress...
        # use a lock for this file -> https://www.geeksforgeeks.org/file-locking-in-python/
        # After obtaining the lock, check if a running total is equal to num of GPUs. If so, then create a bash script for cleaning up

        # Writing bash script: https://stackoverflow.com/questions/49516592/easily-creating-a-bash-script-in-python
        # for i in {0..7}
        # do
        #   mv gpu_world_id_$i/*.npz cached_outputs/
        # done
        # then remove all the created folders....
    return

if __name__ == '__main__':
    main()
