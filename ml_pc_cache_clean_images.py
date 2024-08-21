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
import glob
from natsort import natsorted, ns
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

from utils.defense import gen_mask_set, double_masking_cache, ModelConfig
from utils.metrics import PerformanceMetrics

import sys
sys.path.append("packages/ASL/")
from packages.ASL.src.loss_functions.losses import AsymmetricLoss

parser = argparse.ArgumentParser(description='Multi-Label PatchCleanser Certification')

# Dataset specifics
parser.add_argument('--cache-location', help='path to cached output values')
parser.add_argument('--dataset-name', choices=["mscoco", "nuswide", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80)
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')

# Model specifics
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')

# Mask set specifics
parser.add_argument('--patchcleanser', action='store_true', help='enable PatchCleanser algorithm for inference; to disable, run --no-patchcleanser as the arg')
parser.add_argument('--no-patchcleanser', dest='patchcleanser', action='store_false', help='disable PatchCleanser algorithm for inference; to enable, run --patchcleanser as the arg')
parser.set_defaults(patchcleanser=True)
parser.add_argument('--patch-size', default=64, type=int, help='patch size (default: 64)')
parser.add_argument('--mask-number-fr', default=6, type=int, help='mask number first round (default: 6)')
parser.add_argument('--mask-number-sr', default=6, type=int, help='mask number second round (default: 6)')

# Miscellaneous
parser.add_argument('--trial-type', default="vanilla", type=str, help='type of checkpoints used with the trial (default: vanilla/unmodified)')

def file_print(file_path, msg):
    with open(file_path, "a") as f:
        print(msg, flush=True, file=f) 

def main():
    args = parser.parse_args()
    args.rank = 0

    # Create directory for logging
    args.save_dir = str(Path(args.cache_location).parent / f"{'defended' if args.patchcleanser else 'undefended'}" / f"{args.trial_type}_thre_{(args.thre * 100):g}percent") + "/"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    args.logging_file = args.save_dir + "logging.txt"

    # Create R-covering set of masks for both the first and second rounds
    im_size = [args.image_size, args.image_size]
    patch_size = [args.patch_size, args.patch_size]
    mask_number_fr = [args.mask_number_fr, args.mask_number_fr]
    mask_list_fr, mask_size_fr, mask_stride_fr = gen_mask_set(im_size, patch_size, mask_number_fr) if args.patchcleanser else (None, None, None)

    validate_cache(mask_list_fr, args)

def predict_cache(clean_output, target, criterion, model_config):
    rank = model_config.rank
    thre = model_config.thre

    # sigmoid will be done in loss, therefore apply the logit function here to undo the sigmoid from caching
    logit_output = torch.special.logit(torch.Tensor(clean_output), eps=None)

    # Compute loss and predictions
    loss = criterion(torch.Tensor(logit_output), torch.Tensor(target))
    pred = (clean_output > thre).astype(int)

    return torch.Tensor(pred), loss.item()

def validate_cache(mask_list_fr, args):
    file_print(args.logging_file, "starting validation...")

    # Find all .npz files corresponding to the cached outputs
    cached_list = natsorted(glob.glob(f'{args.cache_location}/*.npz'), key=lambda y: y.lower())

    # Initialize variables for validation
    preds = []
    targets = []
    num_classes = args.num_classes

    metrics = PerformanceMetrics(num_classes)
    model_config = ModelConfig(num_classes, args.rank, args.thre)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    total_loss = 0.0
    dataset_len = 0

    # target shape: [batch_size, object_size_channels, number_classes]
    for batch_index, cached_file in enumerate(cached_list):
        
        # Load in the .npz file
        with np.load(cached_file) as output_dict:
            target = output_dict["target"]
            clean_output = output_dict["clean_output"]
            masked_output = output_dict["masked_output"]

        all_preds = (masked_output > args.thre).astype(int)

        # Compute output
        pred, loss = (double_masking_cache(all_preds, mask_list_fr, num_classes, model_config), np.nan) if mask_list_fr else predict_cache(clean_output, target, criterion, model_config)

        # The ASL loss in each batch is NOT the average of losses from each image - rather, it is the sum
        total_loss += loss
        dataset_len += clean_output.shape[0]

        # Compute TP, TN, FN, FP
        tp = (pred + target).eq(2).cpu().numpy().astype(int)
        tn = (pred + target).eq(0).cpu().numpy().astype(int)
        fn = (pred - target).eq(-1).cpu().numpy().astype(int)
        fp = (pred - target).eq(1).cpu().numpy().astype(int)

        # Compute precision and recall
        metrics.updateMetrics(TP=tp, TN=tn, FN=fn, FP=fp)
        precision_o, recall_o = metrics.overallPrecision(), metrics.overallRecall()
        precision_c, recall_c = metrics.averageClassPrecision(), metrics.averageClassRecall()

        # if (batch_index % 100 == 0):
        file_print(args.logging_file, f'Batch: [{batch_index}/{len(cached_list)}]')
        file_print(args.logging_file, f'P_O {precision_o:.2f} R_O {recall_o:.2f} P_C {precision_c:.2f} R_C {recall_c:.2f} loss {f"{loss:.2f}" if not np.isnan(loss) else "Not Applicable"}\n')

    # Compute average loss per image sample
    average_loss = total_loss / dataset_len

    # Save the certified TP, TN, FN, FP
    np.savez(args.save_dir + f"certified_metrics", TP=metrics.TP, TN=metrics.TN, FN=metrics.FN, FP=metrics.FP)

    # Save the precision and recall metrics
    with open(args.save_dir + f"performance_metrics.txt", "w") as f:
        f.write(f"P_O: {precision_o:.2f} \t R_O: {recall_o:.2f}\n")
        f.write(f"P_C: {precision_c:.2f} \t R_C: {recall_c:.2f}\n")
        f.write(f'average loss: {f"{average_loss:.4f}" if not np.isnan(average_loss) else "Not Applicable"}')


if __name__ == '__main__':
    main()
