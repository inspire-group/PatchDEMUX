# Adopted from: https://github.com/Alibaba-MIIL/ASL/blob/main/validate.py

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

from pathlib import Path
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

from utils.defense import gen_mask_set
from utils.metrics import PerformanceMetrics
from utils.datasets import CocoDetection, split_dataset_gpu

import sys
sys.path.append("packages/ASL/")
from packages.ASL.src.models import create_model

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
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='./TRresNet_L_448_86.6.pth', type=str)
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')
parser.add_argument('--attacker-type', choices=["worst_case", "FN_attacker", "FP_attacker"], default="worst_case")

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

def main():
    args = parser.parse_args()
    args.batch_size = args.batch_size   

    # Get GPU id
    world_gpu_id = args.world_gpu_id

    # Construct file path for saving metrics
    foldername = f"dump/certification/{args.dataset_name}/patch_{args.patch_size}_masknumfr_{args.mask_number_fr}_masknumsr_{args.mask_number_sr}/{todaystring}/trial_{args.trial}_{args.trial_type}_thre_{(int)(args.thre * 100)}percent/gpu_world_id_{args.world_gpu_id}/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging.txt"

    # Setup model
    file_print(args.logging_file, 'creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    args.do_bottleneck_head = False
    args.rank = 0

    # Create model
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    classes_list = np.array(list(state['idx_to_class'].values()))
    file_print(args.logging_file, 'done\n')

    # Data loading code
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])

    instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
    data_path = os.path.join(args.data, 'images/val2014')
    val_dataset = CocoDetection(data_path,
                                instances_path,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))


    # Create GPU specific dataset
    gpu_val_dataset, start_idx, end_idx = split_dataset_gpu(val_dataset, args.batch_size, args.total_num_gpu, world_gpu_id)
    file_print(args.logging_file, "listing out info about this GPU process...")
    file_print(args.logging_file, f"length of gpu_val_dataset: {len(gpu_val_dataset)}\nbatch is currently at: {(int)(start_idx / args.batch_size)}\nstart_idx: {start_idx}\nend_idx: {end_idx}")

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
    file_print(args.logging_file, "starting actual validation...")

    Sig = torch.nn.Sigmoid()

    preds = []
    targets = []
    num_masks_fr, num_masks_sr = len(mask_list_fr), len(mask_list_sr)
    num_classes = len(classes_list)

    metrics = PerformanceMetrics(num_classes)
    tight_metrics = PerformanceMetrics(1)

# DO CHECKING ON INDIVIDUAL IMAGE ROBUSTNESS WITH IMAGE 3 (IE, THE DOG AND BICYCLE PICTURE; need to verify that the histogram aligns with the actual double masks listed in the analysis file
# THEN IMPLEMENT A BATCH VERSION OF THIS
# Batch version strategy: make metric class_idx be a boolean but of length 80 (i.e., every class) x BATCH_SIZE where a True element signifies that the class 
# is relevent to the metric for the given image in the batch. Mask_histogram should be BATCH_SIZE in depth. 
# Then, discordant_preds_bool should just be all_preds for a given mask == discordant_pred. During the line 160 loop
# loop over EVERY one of the 80 classes, get the subset of images where each class failed, and then slice mask_histogram according to this subset of images from the batch
# alternative: just loop this manually over every image in the batch in the validate func below

    # Compute a mask histogram in order to derive tighter bounds during certification
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

    # target shape: [batch_size, object_size_channels, number_classes]
    for batch_index, (input, target) in enumerate(val_loader):
        
        file_print(args.logging_file, f'Batch: [{batch_index}/{len(val_loader)}]')

        # torch.max returns (values, indices), additionally squeezes along the dimension dim
        target = target.max(dim=1)[0]
        im = input
        target = target.cpu().numpy()

        # Initialize all_preds to -1 in order to filter out unused mask combinations at the end
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

                pred = output.data.gt(args.thre).long()
                all_preds[:, i * num_masks_sr + j] = pred.cpu().numpy()

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

        file_print(args.logging_file, f'P_O {precision_o:.2f} R_O {recall_o:.2f} P_C {precision_c:.2f} R_C {recall_c:.2f}\n')

        # Compute tighter bounds on the worst case FN and worst case FP
        fn_bound_batch = np.zeros((im.shape[0], 1))
        fp_bound_batch = np.zeros((im.shape[0], 1))
        for im_idx in range(im.shape[0]):
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

        file_print(args.logging_file, f'P_TO {precision_tight_o:.2f} R_TO {recall_tight_o:.2f}\n')

    # Save the certified TP, TN, FN, FP
    np.savez(args.save_dir + f"certified_metrics", TP=metrics.TP, TN=metrics.TN, FN=metrics.FN, FP=metrics.FP)
    np.savez(args.save_dir + f"certified_metrics_tight", TP=tight_metrics.TP, TN=tight_metrics.TN, FN=tight_metrics.FN, FP=tight_metrics.FP)

    # Save the precision and recall metrics
    with open(args.save_dir + f"performance_metrics.txt", "w") as f:
        f.write(f"P_O: {precision_o:.2f} \t R_O: {recall_o:.2f}\n")
        f.write(f"P_C: {precision_c:.2f} \t R_C: {recall_c:.2f}\n\n")
        f.write(f"P_TO: {precision_tight_o:.2f} \t R_TO: {recall_tight_o:.2f}")

# Worst case
# array([[2.],
#        [1.],
#        [0.],
#        [2.],
#        [0.],
#        [2.],
#        [0.],
#        [5.],
#        [1.],
#        [2.],
#        [1.],
#        [0.],
#        [4.],
#        [1.],
#        [0.],
#        [4.],
#        [1.],
#        [3.],
#        [1.],
#        [0.],
#        [0.],
#        [0.],
#        [2.],
#        [1.],
#        [0.],
#        [0.],
#        [1.],
#        [0.],
#        [1.],
#        [1.],
#        [4.],
#        [0.],
#        [0.],
#        [1.],
#        [2.],
#        [0.],
#        [1.],
#        [4.],
#        [0.],
#        [2.],
#        [1.],
#        [0.],
#        [0.],
#        [5.],
#        [0.],
#        [0.],
#        [2.],
#        [2.],
#        [4.],
#        [2.],
#        [1.],
#        [3.],
#        [1.],
#        [0.],
#        [3.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [1.],
#        [0.],
#        [1.],
#        [1.],
#        [2.]])

#FP
# array([[2.],
#        [1.],
#        [0.],
#        [2.],
#        [0.],
#        [2.],
#        [0.],
#        [5.],
#        [1.],
#        [2.],
#        [1.],
#        [0.],
#        [4.],
#        [1.],
#        [0.],
#        [4.],
#        [1.],
#        [3.],
#        [1.],
#        [0.],
#        [0.],
#        [0.],
#        [2.],
#        [1.],
#        [0.],
#        [0.],
#        [1.],
#        [0.],
#        [1.],
#        [1.],
#        [4.],
#        [0.],
#        [0.],
#        [1.],
#        [2.],
#        [0.],
#        [1.],
#        [4.],
#        [0.],
#        [2.],
#        [1.],
#        [0.],
#        [0.],
#        [5.],
#        [0.],
#        [0.],
#        [2.],
#        [2.],
#        [4.],
#        [2.],
#        [1.],
#        [3.],
#        [1.],
#        [0.],
#        [3.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [1.],
#        [0.],
#        [1.],
#        [1.],
#        [2.]])

    return

if __name__ == '__main__':
    main()
