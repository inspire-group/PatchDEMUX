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
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from defenses.patchcleanser.pc_utils import gen_mask_set
from defenses.patchcleanser.pc_infer import pc_infer_doublemasking
from utils.metrics import PerformanceMetrics
from utils.datasets import split_dataset_gpu
from utils.common import file_print, load_model, load_eval_dataset, ModelConfig, predict

sys.path.append(os.path.join(parent_dir, "packages/ASL/"))
from packages.ASL.src.loss_functions.losses import AsymmetricLoss

parser = argparse.ArgumentParser(description='PatchDEMUX inference')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', choices=["mscoco", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80, type=int, help='number of classes (default: 80)')
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=64
, type=int, help='mini-batch size (default: 64)')

# Model specifics
available_models = ['tresnet_l', 'Q2L-CvT_w24-384']
parser.add_argument('--model-name', choices=available_models, default='tresnet_l')
parser.add_argument('--model-path', type=str)
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model. default is False. ')

# * Transformer config file (optional, required for ViT models)
parser.add_argument('--config', type=str, default=None, help='config file containing all ViT parameters')

# Defense specifics
parser.add_argument('--defense', action='store_true', help='enable PatchDEMUX algorithm for inference; to disable, run --no-defense as the arg')
parser.add_argument('--no-defense', dest='defense', action='store_false', help='run inference on an undefended model; to enable, run --defense as the arg')
parser.set_defaults(defense=True)
parser.add_argument('--patch-size', default=64, type=int, help='patch size (default: 64)')
parser.add_argument('--mask-number', default=6, type=int, help='mask number (default: 6)')

# GPU info for parallelism
parser.add_argument('--world-gpu-id', default=0, type=int, help='overall GPU id (default: 0)')
parser.add_argument('--total-num-gpu', default=1, type=int, help='total number of GPUs (default: 1)')

# Miscellaneous
parser.add_argument('--trial', default=1, type=int, help='trial (default: 1)')
parser.add_argument('--trial-type', default="vanilla", type=str, help='type of checkpoints used with the trial (default: vanilla/unmodified)')

def main():
    args = parser.parse_args()

    if args.model_name == "Q2L-CvT_w24-384":
        is_ViT = True
        if args.config is None:
            raise ValueError("--config parameter is required when using ViT models (Q2L-CvT_w24-384)")
    else:
        is_ViT = False

    # Get GPU id
    world_gpu_id = args.world_gpu_id

    # Construct file path for saving metrics
    foldername = os.path.join(parent_dir, f"dump/outputs/{args.dataset_name}/{'ViT' if is_ViT else 'resnet'}/patch_{args.patch_size}_masknum_{args.mask_number}/{todaystring}/trial_{args.trial}_{args.trial_type}_thre_{(int)(args.thre * 100)}percent/{'defended' if args.defense else 'undefended'}/gpu_world_id_{args.world_gpu_id}/")
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging_val.txt"

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
    mask_list, mask_size, mask_stride = gen_mask_set(im_size, patch_size, mask_number) if args.defense else (None, None, None)

    pd_infer(model, val_loader, args, mask_list)

def pd_infer(model, val_loader, args, mask_list=None):
    file_print(args.logging_file, "starting inference...")

    num_classes = args.num_classes
    metrics = PerformanceMetrics(num_classes)
    model_config = ModelConfig(num_classes, args.thre)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    total_loss = 0.0

    # target shape: [batch_size, object_size_channels, number_classes]
    for batch_index, batch in enumerate(val_loader):
        input_data = batch[0]
        target = batch[1]

        file_print(args.logging_file, f'Batch: [{batch_index}/{len(val_loader)}]')

        # torch.max returns (values, indices), additionally squeezes along the dimension dim
        if args.dataset_name == "mscoco":
            target = target.max(dim=1)[0]
        
        im = input_data.cuda()
        target = target.cpu().numpy()

       # Call single-label CDPA PatchCleanser inference if the flag is enabled
        if mask_list is not None:
            pred = pc_infer_doublemasking(im, mask_list, num_classes, model, model_config)
            loss = np.nan  # Loss is not defined in the defended setting
        else:
            pred, loss = predict(model, im, target, criterion, model_config)

        # The ASL loss in each batch is NOT the average of losses from each image - rather, it is the sum
        total_loss += loss
        
        # Compute TP, TN, FN, FP
        tp = (pred + target).eq(2).cpu().numpy().astype(int)
        tn = (pred + target).eq(0).cpu().numpy().astype(int)
        fn = (pred - target).eq(-1).cpu().numpy().astype(int)
        fp = (pred - target).eq(1).cpu().numpy().astype(int)

        # Compute precision and recall
        metrics.updateMetrics(TP=tp, TN=tn, FN=fn, FP=fp)
        precision_o, recall_o = metrics.overallPrecision(), metrics.overallRecall()
        precision_c, recall_c = metrics.averageClassPrecision(), metrics.averageClassRecall()

        file_print(args.logging_file, f'P_O {precision_o:.2f} R_O {recall_o:.2f} P_C {precision_c:.2f} R_C {recall_c:.2f} loss {f"{loss:.2f}" if not np.isnan(loss) else "Not Applicable"}\n')

    # Compute average loss per image sample
    average_loss = total_loss / len(val_loader.dataset)

    # Save the inference metrics
    np.savez(args.save_dir + f"inference_metrics", TP=metrics.TP, TN=metrics.TN, FN=metrics.FN, FP=metrics.FP)

    # Save the precision and recall metrics
    with open(args.save_dir + f"performance_metrics.txt", "w") as f:
        f.write(f"P_O: {precision_o:.2f} \t R_O: {recall_o:.2f}\n")
        f.write(f"P_C: {precision_c:.2f} \t R_C: {recall_c:.2f}\n")
        f.write(f'average loss: {f"{average_loss:.4f}" if not np.isnan(average_loss) else "Not Applicable"}')

    return

if __name__ == '__main__':
    main()
