# Adopted from: 
# - https://github.com/Alibaba-MIIL/ASL/blob/main/validate.py
# - https://github.com/SlongLiu/query2labels/blob/main/q2l_infer.py 

import argparse
import time
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
from defenses.patchcleanser.pc_infer import pc_infer_doublemasking, pc_infer_firstclass
from utils.metrics import PerformanceMetrics
from utils.datasets import split_dataset_gpu
from utils.common import file_print, load_model, load_eval_dataset, ModelConfig, predict

sys.path.append(os.path.join(parent_dir, "packages/ASL/"))
from packages.ASL.src.loss_functions.losses import AsymmetricLoss

parser = argparse.ArgumentParser(description='PatchDEMUX inference runtime')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', choices=["mscoco", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80, type=int, help='number of classes')
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')

# Model specifics
available_models = ['tresnet_l', 'Q2L-CvT_w24-384']
parser.add_argument('--model-name', choices=available_models, default='Q2L-CvT_w24-384')
parser.add_argument('--model-path', default='./TRresNet_L_448_86.6.pth', type=str)
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
    val_status = f"defended/{args.dataset_name}/patch_{args.patch_size}_masknum_{args.mask_number}" if args.defense else f"undefended/{args.dataset_name}"
    foldername = os.path.join(parent_dir, f"runtime/{val_status}/{'ViT' if is_ViT else 'resnet'}/{todaystring}/trial_{args.trial}_{args.trial_type}_thre_{(int)(args.thre * 100)}percent/gpu_world_id_{args.world_gpu_id}/")
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging_val.txt"

    # Setup model
    model, args = load_model(args, is_ViT)

    # Data loading code
    val_dataset = load_eval_dataset(args)

    # Use a subset of 2000 random indices here (make sure it corresponds to the correct dataset)
    loaded_dict = dict(np.load("/home/djacob/multi-label-patchcleanser/scripts/runtime_exps/mscoco_rand_idx.npz"))
    random_subset = torch.utils.data.Subset(val_dataset, loaded_dict["rand_idx"])

    # Create GPU specific dataset - use a batch size of 1 in order to have the most straightforward per-sample time measurement
    args.batch_size = 1
    gpu_val_dataset, start_idx, end_idx = split_dataset_gpu(random_subset, args.batch_size, args.total_num_gpu, world_gpu_id)
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

    pd_infer_runtime(model, val_loader, args, mask_list)

def pd_infer_runtime(model, val_loader, args, mask_list = None):
    file_print(args.logging_file, "starting inference timing...")

    num_classes = args.num_classes

    metrics = PerformanceMetrics(num_classes)
    model_config = ModelConfig(num_classes, args.thre)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    total_loss = 0.0

    # target shape: [batch_size, object_size_channels, number_classes]
    indiv_class_runtime_arr = []
    runtime_arr = []
    for batch_index, batch in enumerate(val_loader):
        input_data = batch[0]
        target = batch[1]

        # torch.max returns (values, indices), additionally squeezes along the dimension dim
        if args.dataset_name == "mscoco":
            target = target.max(dim=1)[0]
        im = input_data.cuda()

        # Compute output corresponding to the first class (i.e., vanilla PatchCleanser)
        t1_first_class = 0
        t2_first_class = 0
        if mask_list:
            t1_first_class = time.process_time()
            first_class_pred = pc_infer_firstclass(im, mask_list, num_classes, model, model_config)
            t2_first_class = time.process_time()

        # Use PatchDEMUX to compute the output for all classes - use PatchCleanser as a backbone if the flag is enabled
        t1 = time.process_time()
        if mask_list is not None:
            pred = pc_infer_doublemasking(im, mask_list, num_classes, model, model_config)
            loss = np.nan  # Loss is not defined in the defended setting
        else:
            pred, loss = predict(model, im, target, criterion, model_config)
        t2 = time.process_time()

        # The ASL loss in each batch is NOT the average of losses from each image - rather, it is the sum
        total_loss += loss

        if mask_list:
            assert(first_class_pred.item() == pred[0][0].item())
        
        # Compute TP, TN, FN, FP
        tp = (pred + target).eq(2).cpu().numpy().astype(int)
        tn = (pred + target).eq(0).cpu().numpy().astype(int)
        fn = (pred - target).eq(-1).cpu().numpy().astype(int)
        fp = (pred - target).eq(1).cpu().numpy().astype(int)

        # Compute precision and recall
        metrics.updateMetrics(TP=tp, TN=tn, FN=fn, FP=fp)
        precision_o, recall_o = metrics.overallPrecision(), metrics.overallRecall()
        precision_c, recall_c = metrics.averageClassPrecision(), metrics.averageClassRecall()

        file_print(args.logging_file, f'Batch: [{batch_index}/{len(val_loader)}]')
        file_print(args.logging_file, f'P_O {precision_o:.2f} R_O {recall_o:.2f} P_C {precision_c:.2f} R_C {recall_c:.2f} loss {f"{loss:.2f}" if not np.isnan(loss) else "Not Applicable"}\n')

        # Runtime averaging
        indiv_class_runtime_arr.append(t2_first_class - t1_first_class)
        runtime_arr.append(t2 - t1)
        
    # Save timing information
    with open(args.save_dir + f"runtime.txt", "w") as f:
        f.write("Time elapsed\n")
        for i in range(len(runtime_arr)):
            f.write(f"image idx {i}: single label defense -> {indiv_class_runtime_arr[i]} seconds, multi label defense -> {runtime_arr[i]} seconds\n")
        f.write(f"===Average out of {len(runtime_arr)} samples: single label -> {np.mean(indiv_class_runtime_arr)} seconds, multi label -> {np.mean(runtime_arr)} seconds===")

    np.savez(args.save_dir + f"runtime_array", indiv_class_runtime_arr=indiv_class_runtime_arr, runtime_arr=runtime_arr)

    return

if __name__ == '__main__':
    main()
