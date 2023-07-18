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


from torch.cuda.amp import GradScaler, autocast


from pathlib import Path
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

from utils.metrics import PerformanceMetrics
from utils.datasets import CocoDetection

import sys
sys.path.append("ASL/")
from ASL.src.models import create_model
from ASL.src.loss_functions.losses import AsymmetricLoss

parser = argparse.ArgumentParser(description='Multi-Label ASL Model Validation')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', choices=["mscoco", "nuswide", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80)
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size (default: 64)')

# Model specifics
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='./TRresNet_L_448_86.6.pth', type=str)
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')

# Miscellaneous
parser.add_argument('--trial', default=1, type=int, help='trial (default: 1)')

def file_print(file_path, msg):
    with open(file_path, "a") as f:
        print(msg, flush=True, file=f) 

def main():
    args = parser.parse_args()
    args.batch_size = args.batch_size   

    # Construct file path for saving metrics
    foldername = f"dump/undefended/{args.dataset_name}/{todaystring}/trial_{args.trial}/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging_val.txt"

    # Setup model
    file_print(args.logging_file, 'creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    #args.num_classes = state['num_classes']  # use value provided by args for now
    args.do_bottleneck_head = False
    args.rank = 0

    # Create model
    model = create_model(args).cuda()
    # model.load_state_dict(state['model'], strict=True) # FIX THE SAVING DICTIONARY FROM TRAIN DATA!!!!!!
    model.load_state_dict(state, strict=True)
    model.eval()
    #classes_list = np.array(list(state['idx_to_class'].values()))  # FIX THE SAVING DICTIONARY FROM TRAIN DATA!!!!!!
    classes_list = np.ones(args.num_classes)
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

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate_multi(model, val_loader, classes_list, args)

def validate_multi(model, val_loader, classes_list, args):
    file_print(args.logging_file, "starting actual validation...")

    Sig = torch.nn.Sigmoid()

    preds = []
    targets = []
    num_classes = len(classes_list)

    metrics = PerformanceMetrics(num_classes)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    total_loss = 0.0
    # target shape: [batch_size, object_size_channels, number_classes]
    for batch_index, (input, target) in enumerate(val_loader):

        # torch.max returns (values, indices), additionally squeezes along the dimension dim
        target = target.max(dim=1)[0]
        im = input.to(args.rank)

        # compute output
        with torch.no_grad():
            with autocast():  # mixed precision
                output = model(im)
                output_regular = Sig(output).cpu()

        # The ASL loss in each batch is NOT the average of losses from each image - rather, it is the sum
        loss = criterion(output.to(args.rank), target.to(args.rank))
        total_loss += loss.item()

        pred = output_regular.detach().gt(args.thre).long()
        
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
            file_print(args.logging_file, f'P_O {precision_o:.2f} R_O {recall_o:.2f} P_C {precision_c:.2f} R_C {recall_c:.2f} loss {loss.item():.2f}\n')

    # Compute average loss per image sample
    average_loss = total_loss / len(val_loader.dataset)

    # Save the certified TP, TN, FN, FP
    np.savez(args.save_dir + f"certified_metrics", TP=metrics.TP, TN=metrics.TN, FN=metrics.FN, FP=metrics.FP)

    # Save the precision and recall metrics
    with open(args.save_dir + f"performance_metrics.txt", "w") as f:
        f.write(f"P_O: {precision_o:.2f} \t R_O: {recall_o:.2f}\n")
        f.write(f"P_C: {precision_c:.2f} \t R_C: {recall_c:.2f}\n")
        f.write(f"average loss: {average_loss:.4f}")

    return

if __name__ == '__main__':
    main()
