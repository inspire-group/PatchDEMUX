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

from utils.defense import gen_mask_set
from utils.metrics import PerformanceMetrics

parser = argparse.ArgumentParser(description='Multi-Label PatchCleanser Certification')

# Dataset specifics
parser.add_argument('--cache-location', help='path to cached output values')
parser.add_argument('--dataset-name', choices=["mscoco", "nuswide", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')

# Model specifics
parser.add_argument('--attacker-type', choices=["none", "worst_case", "FN_attacker", "FP_attacker"], default="none")

# Interpolation parameters
parser.add_argument('--first-thre', type=float, help='first threshold value')
parser.add_argument('--first-recall', type=float, help='first recall value')
parser.add_argument('--second-thre', type=float, help='second threshold value')
parser.add_argument('--second-recall', type=float, help='second recall value')

parser.add_argument('--target-recall', type=float, help='target recall value')
parser.add_argument('--tolerance', default=0.5, type=float, help='specify how close final output is to the target')
parser.add_argument('--max-interpolation-steps', default=15, type=int, help='number of interpolation steps until termination')

# Mask set specifics
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

    # Create directory for logging
    args.save_dir = str(Path(args.cache_location).parent/"interpolated_vals"/f"certification{f'_hist_{args.attacker_type}' if args.attacker_type != 'none' else ''}"/f"target_recall_{(args.target_recall):g}percent"/f"trial_{args.trial}") + "/"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    args.logging_file = args.save_dir + "logging.txt"

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

    certify_cache(mask_list_fr, mask_list_sr, mask_round_equal, args)

def certify_cache(mask_list_fr, mask_list_sr, mask_round_equal, args):
    file_print(args.logging_file, "starting certification...")

    # Find all .npz files corresponding to the cached outputs
    cached_list = natsorted(glob.glob(f'{args.cache_location}/*.npz'), key=lambda y: y.lower())

    # Initialize security parameters
    Sig = torch.nn.Sigmoid()

    @dataclass
    class ThresholdResults:
        thre: float
        recall: float

    num_masks_fr, num_masks_sr = len(mask_list_fr), len(mask_list_sr)
    num_classes = args.num_classes

    # Compute a mask histogram in order to derive tighter bounds during certification -> put this into defense.py and remove "generator" from the name...
    def mask_histogram_generator(all_preds, metrics, metric_type, num_masks):
        discordant_pred = 0 if metric_type == "FN" else 1

        # Create a mask "histogram" corresponding to classes which have failed certification
        metric_class_idx = np.where(metrics == 1)[1]
        mask_histogram = np.zeros((1, num_masks, len(metric_class_idx)))
        
        # Loop over all sets of double masks to find where certification fails
        for i in range (num_masks * num_masks):

            discordant_preds_bool = (all_preds[:, i, :][:, metric_class_idx]) == discordant_pred

            for idx in range(len(metric_class_idx)):
                # If for a given mask combination the relevant class has failed certification, we mark this in the histogram
                if discordant_preds_bool[:, idx]:
                    first_mask = i // num_masks
                    second_mask = i % num_masks

                    mask_histogram[:, first_mask, idx] = True
                    mask_histogram[:, second_mask, idx] = True
                else:
                    continue

        return mask_histogram

    # Begin interpolation search
    first_point = ThresholdResults(args.first_thre, args.first_recall)
    second_point = ThresholdResults(args.second_thre, args.second_recall)
    interpolation_array = np.array([first_point, second_point])

    overestimated = False
    underestimated = False

    def saveMetrics(save_dir, metrics, tight_metrics):
        # Save the certified TP, TN, FN, FP
        np.savez(save_dir + f"certified_metrics", TP=metrics.TP, TN=metrics.TN, FN=metrics.FN, FP=metrics.FP)
        np.savez(save_dir + f"certified_metrics_tight", TP=tight_metrics.TP, TN=tight_metrics.TN, FN=tight_metrics.FN, FP=tight_metrics.FP)

        # Save the precision and recall metrics
        precision_o, recall_o = metrics.overallPrecision(), metrics.overallRecall()
        precision_c, recall_c = metrics.averageClassPrecision(), metrics.averageClassRecall()
        precision_tight_o, recall_tight_o = tight_metrics.overallPrecision(), tight_metrics.overallRecall()
        with open(save_dir + f"performance_metrics.txt", "w") as f:
            f.write(f"P_O: {precision_o:.2f} \t R_O: {recall_o:.2f}\n")
            f.write(f"P_C: {precision_c:.2f} \t R_C: {recall_c:.2f}\n\n")
            f.write(f"P_TO: {precision_tight_o:.2f} \t R_TO: {recall_tight_o:.2f}")

    for i in range(args.max_interpolation_steps):
        # Update interpolation values
        interpolation_array = np.array(sorted(interpolation_array, key=lambda x: x.recall))
        recall_array = np.array([x.recall for x in interpolation_array])
        insert_idx = np.searchsorted(recall_array, args.target_recall)
        test_thre = np.mean([interpolation_array[insert_idx - 1].thre, interpolation_array[insert_idx].thre])
        test_thre = np.round(test_thre, decimals=10)

        file_print(args.logging_file, f'At interpolation step: [{i}/{args.max_interpolation_steps}]')
        file_print(args.logging_file, f'Testing threshold: {test_thre:0.10f}')

        # Initialize variables for certifiation
        preds = []
        targets = []

        metrics = PerformanceMetrics(num_classes)
        tight_metrics = PerformanceMetrics(1)

        # Run cache evaluation for the given threshold
        # target shape: [batch_size, object_size_channels, number_classes]
        for batch_index, cached_file in enumerate(cached_list):
            
            if batch_index % 200 == 0:
                file_print(args.logging_file, f'Batch: [{batch_index}/{len(cached_list)}]')

            # Load in the .npz file
            with np.load(cached_file) as output_dict:
                target = output_dict["target"]
                masked_output = output_dict["masked_output"]

            all_preds = (masked_output > test_thre).astype(int)

            # Find which classes had consensus in masked predictions
            all_preds_ones = np.all(all_preds, axis=1)
            all_preds_zeros = np.all(np.logical_not(all_preds), axis=1)
            
            # Compute certified TP, TN, FN, FP
            confirmed_tp = np.logical_and(all_preds_ones, target).astype(int)
            confirmed_tn = np.logical_and(all_preds_zeros, np.logical_not(target)).astype(int)
            worst_case_fn = np.logical_and(np.logical_not(all_preds_ones), target).astype(int)
            worst_case_fp = np.logical_and(np.logical_not(all_preds_zeros), np.logical_not(target)).astype(int)

            # Compute certified precision and recall
            metrics.updateMetrics(TP=confirmed_tp, TN=confirmed_tn, FN=worst_case_fn, FP=worst_case_fp)
            precision_o, recall_o = metrics.overallPrecision(), metrics.overallRecall()
            precision_c, recall_c = metrics.averageClassPrecision(), metrics.averageClassRecall()

            # Compute tighter bounds on the worst case FN and worst case FP -> something that can help with readability is setting some
            # of these array indices to an actual variable name
            fn_bound_batch = np.zeros((all_preds.shape[0], 1))
            fp_bound_batch = np.zeros((all_preds.shape[0], 1))
            for im_idx in range(all_preds.shape[0]):
                total_fn_classes = np.sum(worst_case_fn[im_idx, :])
                total_fp_classes = np.sum(worst_case_fp[im_idx, :])
                fn_bound = total_fn_classes
                fp_bound = total_fp_classes

                ##### TEST THIS WITH THE FIRST BATCH OF 64 IN ORDER TO CONFIRM THAT THE LOGIC IS STILL CONSISTENT, THEN TRY THE OTHER TWO ATTACKERS

                if fn_bound:
                    fn_histogram = mask_histogram_generator(all_preds[None, im_idx, :, :], worst_case_fn[None, im_idx, :], "FN", num_masks_fr)
                    fn_histogram_sum = np.sum(fn_histogram, axis=2, keepdims=True)
                    fn_bound = np.max(fn_histogram_sum)
                    fn_bound_idx = np.where(fn_histogram_sum == fn_bound)[1]

                if fp_bound:
                    fp_histogram = mask_histogram_generator(all_preds[None, im_idx, :, :], worst_case_fp[None, im_idx, :], "FP", num_masks_fr)
                    fp_histogram_sum = np.sum(fp_histogram, axis=2, keepdims=True)
                    fp_bound = np.max(fp_histogram_sum)
                    fp_bound_idx = np.where(fp_histogram_sum == fp_bound)[1]

                # If the attacker prefers one metric over the other (i.e., for instance FN) then the other metric should be bounded
                # by the subset of masks which are optimal for the preferred metric
                if (args.attacker_type == "FN_attacker" and fn_bound and fp_bound): 
                    fp_bound = np.max(fp_histogram_sum[:, fn_bound_idx, :])
                if (args.attacker_type == "FP_attacker" and fn_bound and fp_bound):
                    fn_bound = np.max(fn_histogram_sum[:, fp_bound_idx, :])

                fn_bound_batch[im_idx, :] = fn_bound
                fp_bound_batch[im_idx, :] = fp_bound

            # Using tighter bounds leads to an inability to determine performance at the class level
            fn_difference = np.sum(worst_case_fn, axis=1)[:, None] - fn_bound_batch
            fp_difference = np.sum(worst_case_fp, axis=1)[:, None] - fp_bound_batch

            tight_metrics.updateMetrics(TP=np.sum(confirmed_tp, axis=1)[:, None] + fn_difference, TN=np.sum(confirmed_tn, axis=1)[:, None] + fp_difference, FN=fn_bound_batch, FP=fp_bound_batch)
            precision_tight_o, recall_tight_o = tight_metrics.overallPrecision(), tight_metrics.overallRecall()

        # Log the metrics
        file_print(args.logging_file, f'R_O {recall_o:.2f} R_TO {recall_tight_o:.2f}\n')

        # Add to the interpolation array
        recall_to_check = (recall_o if args.attacker_type == "none" else recall_tight_o)
        interpolation_array = np.insert(interpolation_array, insert_idx, ThresholdResults(test_thre, recall_to_check))

        # Check if the desired tolerance has been met
        if (not overestimated and ((recall_to_check < args.target_recall + args.tolerance) and (recall_to_check >= args.target_recall))):
            overest_path = args.save_dir + f"overestimated_thre_{test_thre * 100:0.10g}/"
            Path(overest_path).mkdir(parents=True, exist_ok=True)
            saveMetrics(overest_path, metrics, tight_metrics)
            file_print(args.logging_file, f'FOUND OVERESTIMATED VALUE!!!\n')

            overestimated = True

        if (not underestimated and ((recall_to_check > args.target_recall - args.tolerance) and (recall_to_check <= args.target_recall))):
            underest_path = args.save_dir + f"underestimated_thre_{test_thre * 100:0.10g}/"
            Path(underest_path).mkdir(parents=True, exist_ok=True)
            saveMetrics(underest_path, metrics, tight_metrics)
            file_print(args.logging_file, f'FOUND UNDERESTIMATED VALUE!!!\n')

            underestimated = True

        if (overestimated and underestimated): break

    return

if __name__ == '__main__':
    main()
