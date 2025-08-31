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

from defenses.patchcleanser.pc_utils import gen_mask_set
from defenses.patchcleanser.pc_certify import pc_certify
from utils.metrics import PerformanceMetrics
from utils.datasets import split_dataset_gpu
from utils.common import file_print, load_model, load_eval_dataset

parser = argparse.ArgumentParser(description='PatchDEMUX certification')

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
parser.add_argument('--model-path', type=str)
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model. default is False. ')
parser.add_argument('--attacker-type', choices=["worst_case", "FN_attacker", "FP_attacker"], default="worst_case")

# * Transformer config file (optional, required for ViT models)
parser.add_argument('--config', type=str, default=None, help='config file containing all ViT parameters')

# Mask set specifics
parser.add_argument('--patch-size', default=64, type=int, help='patch size (default: 64)')
parser.add_argument('--mask-number', default=6, type=int, help='mask number (default: 6)')

# GPU info
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
    foldername = os.path.join(parent_dir, f"dump/outputs/{args.dataset_name}/{'ViT' if is_ViT else 'resnet'}/patch_{args.patch_size}_masknum_{args.mask_number}/{todaystring}/trial_{args.trial}_{args.trial_type}_thre_{(int)(args.thre * 100)}percent/certification_hist_{args.attacker_type}/gpu_world_id_{args.world_gpu_id}/")
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

    # Create R-covering set of masks, which are security params for the single-label CDPA PatchCleanser
    im_size = [args.image_size, args.image_size]
    patch_size = [args.patch_size, args.patch_size]
    mask_number = [args.mask_number, args.mask_number]
    mask_list, mask_size, mask_stride = gen_mask_set(im_size, patch_size, mask_number)

    pd_certify(model, val_loader, mask_list, args)

def pd_certify(model, val_loader, mask_list, args):
    file_print(args.logging_file, "starting certification...")

    num_masks = len(mask_list)
    num_classes = args.num_classes

    metrics = PerformanceMetrics(num_classes)
    tight_metrics = PerformanceMetrics(1)

    # target shape: [batch_size, object_size_channels, number_classes]
    for batch_index, batch in enumerate(val_loader):
        input_data = batch[0]
        target = batch[1]

        file_print(args.logging_file, f'Batch: [{batch_index}/{len(val_loader)}]')

        # torch.max returns (values, indices), additionally squeezes along the dimension dim
        if args.dataset_name == "mscoco":
            target = target.max(dim=1)[0]
    
        im = input_data
        target = target.cpu().numpy()

        # Call single-label CDPA PatchCleanser certification
        confirmed_tp, confirmed_tn, worst_case_fn, worst_case_fp, inv_vul_arrays_fn, inv_vul_arrays_fp = pc_certify(model, mask_list, im, target, args, lambda msg: file_print(args.logging_file, msg))

        # Compute certified precision and recall
        metrics.updateMetrics(TP=confirmed_tp, TN=confirmed_tn, FN=worst_case_fn, FP=worst_case_fp)
        precision_o, recall_o = metrics.overallPrecision(), metrics.overallRecall()
        precision_c, recall_c = metrics.averageClassPrecision(), metrics.averageClassRecall()

        file_print(args.logging_file, f'P_O {precision_o:.2f} R_O {recall_o:.2f} P_C {precision_c:.2f} R_C {recall_c:.2f}\n')

        # Compute tighter bounds on the worst case FN and worst case FP by using location-aware certification
        fn_bounds_batch = np.zeros((im.shape[0], 1))
        fp_bounds_batch = np.zeros((im.shape[0], 1))

        for im_idx in range(im.shape[0]):
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
