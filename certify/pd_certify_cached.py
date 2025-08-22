# Adopted from: 
# - https://github.com/Alibaba-MIIL/ASL/blob/main/validate.py
# - https://github.com/SlongLiu/query2labels/blob/main/q2l_infer.py 

import argparse
import numpy as np
import torch

from pathlib import Path
import glob
from natsort import natsorted, ns
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

from defenses.patchcleanser.pc_utils import gen_mask_set
from defenses.patchcleanser.pc_certify import pc_certify_cached
from utils.metrics import PerformanceMetrics

parser = argparse.ArgumentParser(description='PatchDEMUX certification with cached outputs')

# Dataset specifics
parser.add_argument('--cache-location', help='path to cached output values')
parser.add_argument('--dataset-name', choices=["mscoco", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')

# Model specifics
parser.add_argument('--attacker-type', choices=["worst_case", "FN_attacker", "FP_attacker"], default="worst_case")
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')

# Mask set specifics
parser.add_argument('--patch-size', default=64, type=int, help='patch size (default: 64)')
parser.add_argument('--mask-number', default=6, type=int, help='mask number (default: 6)')

# Miscellaneous
parser.add_argument('--trial-type', default="vanilla", type=str, help='type of checkpoints used with the trial (default: vanilla/unmodified)')

def file_print(file_path, msg):
    with open(file_path, "a") as f:
        print(msg, flush=True, file=f) 

def main():
    args = parser.parse_args()

    # Create directory for logging
    args.save_dir = str(Path(args.cache_location).parent / f"certification_hist_{args.attacker_type}" / f"{args.trial_type}_thre_{(args.thre * 100):g}percent") + "/"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    args.logging_file = args.save_dir + "logging.txt"

    # Create R-covering set of masks, which are security params for the single-label CDPA PatchCleanser
    im_size = [args.image_size, args.image_size]
    patch_size = [args.patch_size, args.patch_size]
    mask_number = [args.mask_number, args.mask_number]
    mask_list, mask_size, mask_stride = gen_mask_set(im_size, patch_size, mask_number)

    pd_certify_cached(mask_list, args)

def pd_certify_cached(mask_list, args):
    file_print(args.logging_file, "starting certification...")

    # Find all .npz files corresponding to the cached outputs
    cached_list = natsorted(glob.glob(f'{args.cache_location}/*.npz'), key=lambda y: y.lower())

    # Initialize variables for certifiation
    num_classes = args.num_classes

    metrics = PerformanceMetrics(num_classes)
    tight_metrics = PerformanceMetrics(1)

    # target shape: [batch_size, object_size_channels, number_classes]
    for batch_index, cached_file in enumerate(cached_list):
        
        file_print(args.logging_file, f'Batch: [{batch_index}/{len(cached_list)}]')

        # Load in the .npz file
        with np.load(cached_file) as output_dict:
            target = output_dict["target"]
            masked_output = output_dict["masked_output"]
    
        # Call single-label CDPA PatchCleanser certification
        confirmed_tp, confirmed_tn, worst_case_fn, worst_case_fp, inv_vul_arrays_fn, inv_vul_arrays_fp = pc_certify_cached(masked_output, mask_list, target, args)

        # Compute certified precision and recall
        metrics.updateMetrics(TP=confirmed_tp, TN=confirmed_tn, FN=worst_case_fn, FP=worst_case_fp)
        precision_o, recall_o = metrics.overallPrecision(), metrics.overallRecall()
        precision_c, recall_c = metrics.averageClassPrecision(), metrics.averageClassRecall()

        file_print(args.logging_file, f'P_O {precision_o:.2f} R_O {recall_o:.2f} P_C {precision_c:.2f} R_C {recall_c:.2f}\n')

        # Compute tighter bounds on the worst case FN and worst case FP by using location-aware certification
        fn_bounds_batch = np.zeros((masked_output.shape[0], 1))
        fp_bounds_batch = np.zeros((masked_output.shape[0], 1))
        for im_idx in range(masked_output.shape[0]):
            fn_bound = np.sum(worst_case_fn[im_idx, :])
            fp_bound = np.sum(worst_case_fp[im_idx, :])

            if fn_bound and inv_vul_arrays_fn[im_idx] is not None:
                inv_vul_arrays_fn_total = np.sum(inv_vul_arrays_fn[im_idx], axis=1, keepdims=True)
                fn_bound = np.max(inv_vul_arrays_fn_total)
                fn_bound_vul_loc = np.where(inv_vul_arrays_fn_total == fn_bound)[0]

            if fp_bound and inv_vul_arrays_fp[im_idx] is not None:
                inv_vul_arrays_fp_total = np.sum(inv_vul_arrays_fp[im_idx], axis=1, keepdims=True)
                fp_bound = np.max(inv_vul_arrays_fp_total)
                fp_bound_vul_loc = np.where(inv_vul_arrays_fp_total == fp_bound)[0]

            # If the attacker prefers one metric over the other (i.e., for instance FN) then the other metric should be bounded
            # based on the vulnerable locations that are optimal for the preferred metric
            # NOTE: If the attacker's preferred metric is not available, then the other metric will be used
            if (args.attacker_type == "FN_attacker" and fn_bound and fp_bound): 
                fp_bound = np.max(inv_vul_arrays_fp_total[fn_bound_vul_loc, :])
            if (args.attacker_type == "FP_attacker" and fn_bound and fp_bound):
                fn_bound = np.max(inv_vul_arrays_fn_total[fp_bound_vul_loc, :])

            fn_bounds_batch[im_idx, :] = fn_bound
            fp_bounds_batch[im_idx, :] = fp_bound

        # Using tighter bounds leads to an inability to determine performance at the class level
        fn_difference = np.sum(worst_case_fn, axis=1)[:, None] - fn_bounds_batch
        fp_difference = np.sum(worst_case_fp, axis=1)[:, None] - fp_bounds_batch

        tight_metrics.updateMetrics(TP=np.sum(confirmed_tp, axis=1)[:, None] + fn_difference, TN=np.sum(confirmed_tn, axis=1)[:, None] + fp_difference, FN=fn_bounds_batch, FP=fp_bounds_batch)
        precision_tight_o, recall_tight_o = tight_metrics.overallPrecision(), tight_metrics.overallRecall()

        file_print(args.logging_file, f'P_TO {precision_tight_o:.2f} R_TO {recall_tight_o:.2f}\n')

    # Save the certified TP, TN, FN, FP
    np.savez(args.save_dir + f"certified_metrics", TP=metrics.TP, TN=metrics.TN, FN=metrics.FN, FP=metrics.FP)
    np.savez(args.save_dir + f"certified_metrics_tight", TP=tight_metrics.TP, TN=tight_metrics.TN, FN=tight_metrics.FN, FP=tight_metrics.FP)

    # Save the precision and recall metrics
    with open(args.save_dir + f"performance_metrics.txt", "w") as f:
        f.write(f"P_O: {precision_o:.2f} \t R_O: {recall_o:.2f}\n")
        f.write(f"P_C: {precision_c:.2f} \t R_C: {recall_c:.2f}\n\n")
        f.write(f"P_TO: {precision_tight_o:.2f} \t R_TO: {recall_tight_o:.2f}")
    return

if __name__ == '__main__':
    main()
