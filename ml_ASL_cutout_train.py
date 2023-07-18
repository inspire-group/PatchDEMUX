# Adopted from: https://github.com/Alibaba-MIIL/ASL/blob/main/train.py

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
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
import os

from pathlib import Path
from contextlib import nullcontext
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

from utils.metrics import PerformanceMetrics
from utils.datasets import CocoDetection
from utils.training_helpers import Cutout, ModelEma

import sys
sys.path.append("ASL/")
from ASL.src.models import create_model
from ASL.src.loss_functions.losses import AsymmetricLoss

parser = argparse.ArgumentParser(description='Multi-Label ASL Model Cutout Training')

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

# Training specifics
parser.add_argument('--lr', default=1e-4, type=float, help='maximum learning rate (default: 1e-4)')
parser.add_argument('--cutout-size', default=224, type=int, help='size of cutout masks (default: 224)')
parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision (AMP); to disable, run --no-amp as the arg')
parser.add_argument('--no-amp', dest='amp', action='store_false', help='disable automatic mixed precision (AMP); to enable, run --amp as the arg')
parser.set_defaults(amp=True)
parser.add_argument('--lr-scheduler', choices=["onecyclelr", "linear", "none"], default="onecyclelr", help='learning rate scheduler (default: onecyclelr)')
parser.add_argument('--ema-decay-rate', default=0.9997, type=float, help='expected moving average (EMA) decay rate (default: 0.9997)')

# Miscellaneous
parser.add_argument('--trial', default=1, type=int, help='trial (default: 1)')

def file_print(file_path, msg):
    with open(file_path, "a") as f:
        print(msg, flush=True, file=f) 

def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False

    cutout_len = args.cutout_size  

    # Construct file path for saving metrics
    training_specifics = (f"cutout_{cutout_len}" + 
                        f"{f'_{args.lr_scheduler}' if args.lr_scheduler != 'none' else ''}" +
                        f"{f'_mixedprec' if args.amp else ''}" +
                        f"{f'_ema' if args.ema_decay_rate > 0 else ''}")

    foldername = f"/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/{args.dataset_name}/{training_specifics}/{todaystring}/trial_{args.trial}/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging.txt"

    # Setup model - assume weights are from a model pretrained on MSCOCO
    file_print(args.logging_file, 'creating and loading the model...')

    #breakpoint()
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    args.classes_dict = state['idx_to_class']
    args.do_bottleneck_head = False
    args.rank = 0

    # Create model
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.train()
    file_print(args.logging_file, 'done\n')

    # COCO Data loading
    instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')

    data_path_val   = os.path.join(args.data, 'images/val2014')    # args.data
    data_path_train = os.path.join(args.data, 'images/train2014')  # args.data
    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    train_dataset = CocoDetection(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      transforms.ToTensor(),
                                      Cutout(n_holes=2, length=cutout_len),
                                      # normalize,
                                  ]))
    file_print(args.logging_file, f"len(val_dataset): {len(val_dataset)}")
    file_print(args.logging_file, f"len(train_dataset): {len(train_dataset)}")

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actual Training
    train_multi_label_coco(model, train_loader, val_loader, args.lr, args)

def train_multi_label_coco(model, train_loader, val_loader, lr, args):

    # Initialize EMA
    ema = ModelEma(model, args.ema_decay_rate) if (args.ema_decay_rate > 0) else None

    # set optimizer
    epochs = 10
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scaler = GradScaler() if args.amp else None

    # Initialize scheduler
    scheduler = None
    if (args.lr_scheduler == "onecyclelr"):
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs, pct_start=0.2)
    elif (args.lr_scheduler == "linear"):
        scheduler = lr_scheduler.LinearLR(optimizer, total_iters = 5 * steps_per_epoch)
    current_lr = lr

    lowest_val_loss = np.inf
    for epoch in range(epochs):
        file_print(args.logging_file, "starting training...")

        model.train()
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()    # (batch,3,num_classes)
            target = target.max(dim=1)[0]
            current_lr = scheduler.get_last_lr()[0] if scheduler else current_lr

            # Run forward pass
            with autocast() if scaler else nullcontext() as context:    # mixed precision
                output = model(inputData).float()    # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            # Perform backprop
            scaler.scale(loss).backward() if scaler else loss.backward()
            scaler.step(optimizer) if scaler else optimizer.step()
            if scaler: scaler.update()
 
            if scheduler: scheduler.step()    # Apply learning rate scheduler
            if ema: ema.update(model)    # Update EMA checkpoints

            # display information
            if i % 100 == 0:
                file_print(args.logging_file, f'Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], LR {current_lr:.1e}, Loss: {loss.item():.1f}')
        
        save_model = ema.module if ema else model
        save_model.eval()
        average_loss = validate_multi(val_loader, save_model, args)

        # Save the checkpoints associated with current epoch
        checkpoints = {"model":save_model.state_dict(), "epoch":epoch, "num_classes":args.num_classes, "idx_to_class":args.classes_dict}
        try:

            save_model_dir = args.save_dir + f'epoch_{epoch}/'
            Path(save_model_dir).mkdir(parents=True, exist_ok=True)
            torch.save(checkpoints, save_model_dir + f"{'ema-' if ema else ''}model-epoch-{epoch}.pth")
        except:
            pass

        # Save the checkpoints associated with the best model
        if (average_loss < lowest_val_loss):
            lowest_val_loss = average_loss
            try:
                torch.save(checkpoints, args.save_dir + f"{'ema-' if ema else ''}model-best-epoch-{epoch}.pth")
            except:
                pass

def validate_multi(val_loader, model, args):
    file_print(args.logging_file, "starting validation...")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    targets = []
    metrics = PerformanceMetrics(args.num_classes)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    total_loss = 0.0
    for i, (input, target) in enumerate(val_loader):

        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast() if args.amp else nullcontext():    # mixed precision
                output = model(input.cuda())
                output_regular = Sig(output).cpu()

        # The ASL loss in each batch is NOT the average of losses from each image - rather, it is the sum
        loss = criterion(output.cuda(), target.cuda())
        total_loss += loss.item()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        targets.append(target.cpu().detach())

        # for other metrics
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

        if (i % 100 == 0):
            file_print(args.logging_file, f'Batch: [{i}/{len(val_loader)}]')
            file_print(args.logging_file, f'P_O {precision_o:.2f} R_O {recall_o:.2f} P_C {precision_c:.2f} R_C {recall_c:.2f} loss {loss.item():.2f}\n')
    
    average_loss = total_loss / len(val_loader.dataset)
    file_print(args.logging_file, f'Final Results:')
    file_print(args.logging_file, f'P_O {precision_o:.2f} R_O {recall_o:.2f} P_C {precision_c:.2f} R_C {recall_c:.2f} average loss {average_loss:.4f}\n')

    return average_loss


if __name__ == '__main__':
    main()
