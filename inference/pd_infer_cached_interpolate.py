# Adopted from: 
# - https://github.com/Alibaba-MIIL/ASL/blob/main/validate.py
# - https://github.com/SlongLiu/query2labels/blob/main/q2l_infer.py 
# Assumes that recall monotonically decreases as threshold increases

import argparse
import numpy as np
import torch
from dataclasses import dataclass

from pathlib import Path
import glob
from natsort import natsorted, ns
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

# Add parent directory to path so we can import from utils
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from defenses.patchcleanser.pc_utils import gen_mask_set
from defenses.patchcleanser.pc_infer import pc_infer_doublemasking_cached
from utils.metrics import PerformanceMetrics
from utils.common import file_print, ModelConfig

sys.path.append(os.path.join(parent_dir, "packages/ASL/"))
from packages.ASL.src.loss_functions.losses import AsymmetricLoss

parser = argparse.ArgumentParser(description='PatchDEMUX inference with cached outputs; interpolation')

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
parser.add_argument('--defense', action='store_true', help='enable PatchDEMUX algorithm for inference; to disable, run --no-defense as the arg')
parser.add_argument('--no-defense', dest='defense', action='store_false', help='run inference on an undefended model; to enable, run --defense as the arg')
parser.set_defaults(defense=True)
parser.add_argument('--patch-size', default=64, type=int, help='patch size (default: 64)')
parser.add_argument('--mask-number', default=6, type=int, help='mask number (default: 6)')

# Miscellaneous
parser.add_argument('--trial', default=1, type=int, help='trial (default: 1)')

def main():
    args = parser.parse_args()
    args.rank = 0

    # Create directory for logging
    args.save_dir = str(Path(args.cache_location).parent/"interpolated_vals"/f"{'defended' if args.defense else 'undefended'}"/f"target_recall_{(args.target_recall):g}percent"/f"trial_{args.trial}") + "/"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    args.logging_file = args.save_dir + "logging.txt"

    # Create R-covering set of masks, which are security params for the single-label CDPA PatchCleanser
    im_size = [args.image_size, args.image_size]
    patch_size = [args.patch_size, args.patch_size]
    mask_number = [args.mask_number, args.mask_number]
    mask_list, mask_size, mask_stride = gen_mask_set(im_size, patch_size, mask_number) if args.defense else (None, None, None)

    pd_infer_cached_interpolate(mask_list, args)

def predict_cached(clean_output, target, criterion, model_config):
    thre = model_config.thre

    # sigmoid will be done in loss, therefore apply the logit function here to undo the sigmoid from caching
    logit_output = torch.special.logit(torch.Tensor(clean_output), eps=None)

    # Compute loss and predictions
    loss = criterion(torch.Tensor(logit_output), torch.Tensor(target))
    pred = (clean_output > thre).astype(int)

    return torch.Tensor(pred), loss.item()

def pd_infer_cached_interpolate(mask_list, args):
    file_print(args.logging_file, "starting validation...")

    # Find all .npz files corresponding to the cached outputs
    cached_list = natsorted(glob.glob(f'{args.cache_location}/*.npz'), key=lambda y: y.lower())

    @dataclass
    class ThresholdResults:
        thre: float
        recall: float

    num_classes = args.num_classes

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
        metrics = PerformanceMetrics(num_classes)
        model_config = ModelConfig(num_classes, test_thre)
        
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

            # Call single-label CDPA PatchCleanser inference if the flag is enabled
            if mask_list is not None:
                pred = pc_infer_doublemasking_cached(masked_output, mask_list, num_classes, model_config)
                loss = np.nan  # Loss is not defined in the defended setting
            else:
                pred, loss = predict_cached(clean_output, target, criterion, model_config)

            # The ASL loss in each batch is NOT the average of losses from each image - rather, it is the sum
            total_loss += loss
            dataset_len += clean_output.shape[0]

            # Compute TP, TN, FN, FP
            tp = (pred + target).eq(2).cpu().numpy().astype(int)
            tn = (pred + target).eq(0).cpu().numpy().astype(int)
            fn = (pred - target).eq(-1).cpu().numpy().astype(int)
            fp = (pred - target).eq(1).cpu().numpy().astype(int)

            # Compute recall
            metrics.updateMetrics(TP=tp, TN=tn, FN=fn, FP=fp)
            recall_o = metrics.overallRecall()

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
            file_print(args.logging_file, f'FOUND OVERESTIMATED VALUE!\n')

            overestimated = True

        if (not underestimated and ((recall_o > args.target_recall - args.tolerance) and (recall_o <= args.target_recall))):
            underest_path = args.save_dir + f"underestimated_thre_{test_thre * 100:0.10g}/"
            Path(underest_path).mkdir(parents=True, exist_ok=True)
            saveMetrics(underest_path, metrics, average_loss)
            file_print(args.logging_file, f'FOUND UNDERESTIMATED VALUE!\n')

            underestimated = True

        if (overestimated and underestimated): break

if __name__ == '__main__':
    main()
