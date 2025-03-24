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
from datetime import date
import time
todaystring = date.today().strftime("%m-%d-%Y")

from utils.metrics import PerformanceMetrics
from utils.datasets import CocoDetection, split_dataset_gpu, TransformWrapper
from utils.defense import gen_mask_set, double_masking, ModelConfig

import sys
sys.path.append("packages/ASL/")
from packages.ASL.src.models import create_model
from packages.ASL.src.loss_functions.losses import AsymmetricLoss

sys.path.append("packages/query2labels/lib")
from packages.query2labels.lib.models.query2label import build_q2l

parser = argparse.ArgumentParser(description='Multi-Label ASL Model Validation')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', choices=["mscoco", "nuswide", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80)
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=64
, type=int, help='mini-batch size (default: 64)')

# Model specifics
available_models = ['tresnet_l', 'Q2L-CvT_w24-384']
parser.add_argument('--model-name', choices=available_models, default='Q2L-CvT_w24-384')
parser.add_argument('--model-path', default='./TRresNet_L_448_86.6.pth', type=str)
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model. default is False. ')

# * Transformer
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
parser.add_argument('--patchcleanser', action='store_true', help='enable PatchCleanser algorithm for inference; to disable, run --no-patchcleanser as the arg')
parser.add_argument('--no-patchcleanser', dest='patchcleanser', action='store_false', help='disable PatchCleanser algorithm for inference; to enable, run --patchcleanser as the arg')
parser.set_defaults(patchcleanser=True)
parser.add_argument('--patch-size', default=64, type=int, help='patch size (default: 64)')
parser.add_argument('--mask-number-fr', default=6, type=int, help='mask number first round (default: 6)')
parser.add_argument('--mask-number-sr', default=6, type=int, help='mask number second round (default: 6)')

# GPU info for parallelism
parser.add_argument('--world-gpu-id', default=0, type=int, help='overall GPU id (default: 0)')
parser.add_argument('--total-num-gpu', default=1, type=int, help='total number of GPUs (default: 1)')

# Miscellaneous
parser.add_argument('--trial', default=1, type=int, help='trial (default: 1)')
parser.add_argument('--trial-type', default="baseline", type=str, help='type of checkpoints used with the trial (default: baseline/unmodified)')

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

    # Setup model
    # file_print(args.logging_file, 'creating and loading the model...')
    # state = torch.load(args.model_path, map_location='cpu')
    # args.num_classes = state['num_classes']
    args.do_bottleneck_head = False
    # args.rank = 0

    # Create model
    model = build_q2l(args).cuda() if is_ViT else create_model(args).cuda()

    # Setup depends on whether architecture is based on transformer or ResNet
    file_print(args.logging_file, f"setting up the model...{'ViT' if is_ViT else 'resnet'}")
    state = torch.load(args.model_path, map_location='cpu')
    if is_ViT:
        state_dict = clean_state_dict(state['state_dict'])
        classes_list = np.ones((80, 1))
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

    return model, args, classes_list

def main():
    args = parser.parse_args()

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
    val_status = f"defended/{args.dataset_name}/patch_{args.patch_size}_masknumfr_{args.mask_number_fr}_masknumsr_{args.mask_number_sr}" if args.patchcleanser else f"undefended/{args.dataset_name}"
    foldername = f"dump/{val_status}/{'ViT' if is_ViT else 'resnet'}/{todaystring}/trial_{args.trial}_{args.trial_type}_thre_{(int)(args.thre * 100)}percent/gpu_world_id_{args.world_gpu_id}/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging_val.txt"

    # build model
    model, args, classes_list = load_model(args, is_ViT)

    # Data loading code
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])

    instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
    data_path = os.path.join(args.data, 'images/val2014')
    val_dataset = CocoDetection(data_path,
                                instances_path,
                                transforms.Compose([
                                    TransformWrapper(transforms.Resize((args.image_size, args.image_size))),
                                    TransformWrapper(transforms.ToTensor()),
                                    # normalize, # no need, toTensor does normalization
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
    mask_list_fr, mask_size_fr, mask_stride_fr = gen_mask_set(im_size, patch_size, mask_number_fr) if args.patchcleanser else (None, None, None)

    validate_multi(model, val_loader, classes_list, args, mask_list_fr)

def predict(model, im, target, criterion, model_config):
    rank = model_config.rank
    thre = model_config.thre
    Sig = torch.nn.Sigmoid()

    # Compute output
    with torch.no_grad():
        output = model(im)
        output_regular = Sig(output).cpu()

    # Compute loss and predictions
    loss = criterion(output.to(rank), target.to(rank))  # sigmoid will be done in loss !
    pred = output_regular.detach().gt(thre).long()

    return pred, loss.item()

def validate_multi(model, val_loader, classes_list, args, mask_list_fr = None):
    file_print(args.logging_file, "starting actual validation...")

    preds = []
    targets = []
    num_classes = len(classes_list)

    metrics = PerformanceMetrics(num_classes)
    model_config = ModelConfig(num_classes, args.rank, args.thre)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    total_loss = 0.0

    # target shape: [batch_size, object_size_channels, number_classes]
    for batch_index, batch in enumerate(val_loader):
        input_data = batch[0]
        target = batch[1]

        # torch.max returns (values, indices), additionally squeezes along the dimension dim
        target = target.max(dim=1)[0]
        im = input_data.to(args.rank)

        # Compute output
        pred, loss = (double_masking(im, mask_list_fr, num_classes, model, model_config), np.nan) if mask_list_fr else predict(model, im, target, criterion, model_config)

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

        if (batch_index % 100 == 0):
            file_print(args.logging_file, f'Batch: [{batch_index}/{len(val_loader)}]')
            file_print(args.logging_file, f'P_O {precision_o:.2f} R_O {recall_o:.2f} P_C {precision_c:.2f} R_C {recall_c:.2f} loss {f"{loss:.2f}" if not np.isnan(loss) else "Not Applicable"}\n')

    # Compute average loss per image sample
    average_loss = total_loss / len(val_loader.dataset)

    # Save the certified TP, TN, FN, FP
    np.savez(args.save_dir + f"certified_metrics", TP=metrics.TP, TN=metrics.TN, FN=metrics.FN, FP=metrics.FP)

    # Save the precision and recall metrics
    with open(args.save_dir + f"performance_metrics.txt", "w") as f:
        f.write(f"P_O: {precision_o:.2f} \t R_O: {recall_o:.2f}\n")
        f.write(f"P_C: {precision_c:.2f} \t R_C: {recall_c:.2f}\n")
        f.write(f'average loss: {f"{average_loss:.4f}" if not np.isnan(average_loss) else "Not Applicable"}')

    return

if __name__ == '__main__':
    main()
