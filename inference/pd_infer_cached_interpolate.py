# Adopted from: 
# - https://github.com/Alibaba-MIIL/ASL/blob/main/validate.py
# - https://github.com/SlongLiu/query2labels/blob/main/q2l_infer.py 
# Assumes that recall monotonically decreases as threshold increases

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
from dataclasses import dataclass

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
parser.add_argument('--dataset-name', choices=["mscoco", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')

# Interpolation parameters
parser.add_argument('--first-thre', type=float, help='first threshold value')
parser.add_argument('--first-recall', type=float, help='first recall value')
parser.add_argument('--second-thre', type=float, help='second threshold value')
parser.add_argument('--second-recall', type=float, help='second recall value')

parser.add_argument('--target-recall', type=float, help='target recall value')
parser.add_argument('--tolerance', default=0.5, type=float, help='specify how close final output is to the target')
parser.add_argument('--max-interpolation-steps', default=15, type=int, help='number of interpolation steps until termination')

# Mask set specifics
parser.add_argument('--patchcleanser', action='store_true', help='enable PatchCleanser algorithm for inference; to disable, run --no-patchcleanser as the arg')
parser.add_argument('--no-patchcleanser', dest='patchcleanser', action='store_false', help='disable PatchCleanser algorithm for inference; to enable, run --patchcleanser as the arg')
parser.set_defaults(patchcleanser=True)
parser.add_argument('--patch-size', default=64, type=int, help='patch size (default: 64)')
parser.add_argument('--mask-number-fr', default=6, type=int, help='mask number first round (default: 6)')
parser.add_argument('--mask-number-sr', default=6, type=int, help='mask number second round (default: 6)')

# Miscellaneous
parser.add_argument('--trial', default=1, type=int, help='trial (default: 1)')

def file_print(file_path, msg):
    with open(file_path, "a") as f:
        print(msg, flush=True, file=f) 

def main():
    args = parser.parse_args()
    args.rank = 0

    # Create directory for logging
    args.save_dir = str(Path(args.cache_location).parent/"interpolated_vals"/f"{'defended' if args.patchcleanser else 'undefended'}"/f"target_recall_{(args.target_recall):g}percent"/f"trial_{args.trial}") + "/"
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

    @dataclass
    class ThresholdResults:
        thre: float
        recall: float

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)

    # Begin interpolation search
    first_point = ThresholdResults(args.first_thre, args.first_recall)
    second_point = ThresholdResults(args.second_thre, args.second_recall)
    interpolation_array = np.array([first_point, second_point])

    overestimated = False
    underestimated = False

    def saveMetrics(save_dir, metrics, average_loss):
        # Save the certified TP, TN, FN, FP
        np.savez(save_dir + f"certified_metrics", TP=metrics.TP, TN=metrics.TN, FN=metrics.FN, FP=metrics.FP)

        # Save the precision and recall metrics
        precision_o, recall_o = metrics.overallPrecision(), metrics.overallRecall()
        precision_c, recall_c = metrics.averageClassPrecision(), metrics.averageClassRecall()
        with open(save_dir + f"performance_metrics.txt", "w") as f:
            f.write(f"P_O: {precision_o:.2f} \t R_O: {recall_o:.2f}\n")
            f.write(f"P_C: {precision_c:.2f} \t R_C: {recall_c:.2f}\n\n")
            f.write(f'average loss: {f"{average_loss:.4f}" if not np.isnan(average_loss) else "Not Applicable"}')

    for i in range(args.max_interpolation_steps):
        # Update interpolation values
        interpolation_array = np.array(sorted(interpolation_array, key=lambda x: x.recall))
        recall_array = np.array([x.recall for x in interpolation_array])
        insert_idx = np.searchsorted(recall_array, args.target_recall)
        test_thre = np.mean([interpolation_array[insert_idx - 1].thre, interpolation_array[insert_idx].thre])
        test_thre = np.round(test_thre, decimals=10)

        file_print(args.logging_file, f'At interpolation step: [{i}/{args.max_interpolation_steps}]')
        file_print(args.logging_file, f'Testing threshold: {test_thre:0.10f}')

        # Initialize variables for validation
        preds = []
        targets = []
        num_classes = args.num_classes

        metrics = PerformanceMetrics(num_classes)
        model_config = ModelConfig(num_classes, args.rank, test_thre)
        
        total_loss = 0.0
        dataset_len = 0

        # Run cache evaluation for the given threshold
        # target shape: [batch_size, object_size_channels, number_classes]
        for batch_index, cached_file in enumerate(cached_list):
            
            if batch_index % 200 == 0:
                file_print(args.logging_file, f'Batch: [{batch_index}/{len(cached_list)}]')

            # Load in the .npz file
            with np.load(cached_file) as output_dict:
                target = output_dict["target"]
                clean_output = output_dict["clean_output"]
                masked_output = output_dict["masked_output"]

            all_preds = (masked_output > test_thre).astype(int)

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

        # Compute average loss per image sample
        average_loss = total_loss / dataset_len

        # Log the metrics
        file_print(args.logging_file, f'R_O {recall_o:.2f}\n')

        # Add to the interpolation array
        interpolation_array = np.insert(interpolation_array, insert_idx, ThresholdResults(test_thre, recall_o))

        # Check if the desired tolerance has been met
        if (not overestimated and ((recall_o < args.target_recall + args.tolerance) and (recall_o >= args.target_recall))):
            overest_path = args.save_dir + f"overestimated_thre_{test_thre * 100:0.10g}/"
            Path(overest_path).mkdir(parents=True, exist_ok=True)
            saveMetrics(overest_path, metrics, average_loss)
            file_print(args.logging_file, f'FOUND OVERESTIMATED VALUE!!!\n')

            overestimated = True

        if (not underestimated and ((recall_o > args.target_recall - args.tolerance) and (recall_o <= args.target_recall))):
            underest_path = args.save_dir + f"underestimated_thre_{test_thre * 100:0.10g}/"
            Path(underest_path).mkdir(parents=True, exist_ok=True)
            saveMetrics(underest_path, metrics, average_loss)
            file_print(args.logging_file, f'FOUND UNDERESTIMATED VALUE!!!\n')

            underestimated = True

        if (overestimated and underestimated): break

if __name__ == '__main__':
    main()
