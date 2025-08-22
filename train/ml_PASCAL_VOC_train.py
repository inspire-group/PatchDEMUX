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
import json
from collections import OrderedDict

from pathlib import Path
from contextlib import nullcontext
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

from utils.defense import gen_mask_set
from utils.metrics import PerformanceMetrics
from utils.datasets import CocoDetection, VOCDetection, TransformWrapper
from utils.cutout_augmentations import Cutout, GreedyCutout
from utils.model_ema import ModelEma

import sys
sys.path.append("packages/ASL/")
from packages.ASL.src.models import create_model
from packages.ASL.src.loss_functions.losses import AsymmetricLoss

sys.path.append("packages/query2labels/lib")
from packages.query2labels.lib.models.query2label import build_q2l
from packages.query2labels.lib.models.query2label import GroupWiseLinear

parser = argparse.ArgumentParser(description='Multi-Label Pascal VOC finetuning')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', choices=["mscoco", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size (default: 64)')

# Model specifics
available_models = ['tresnet_l', 'Q2L-CvT_w24-384']
parser.add_argument('--model-name', choices=available_models, default='tresnet_l')
parser.add_argument('--model-path', default='./TRresNet_L_448_86.6.pth', type=str)
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model. default is False. ')
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')

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

# Training specifics
parser.add_argument('--lr', default=1e-4, type=float, help='maximum learning rate (default: 1e-4)')
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
    args.do_bottleneck_head = False

    # Create model - note that the config file should have num_class = 80 in order to properly load the MSCOCO weights
    model = build_q2l(args) if is_ViT else create_model(args)

    # Setup depends on whether architecture is based on transformer or ResNet
    file_print(args.logging_file, f"setting up the model...{'ViT' if is_ViT else 'resnet'}")
    state = torch.load(args.model_path, map_location='cpu')
    if is_ViT:
        state_dict = clean_state_dict(state['state_dict'])
        classes_list = np.ones((args.num_classes, 1))
    else:
        state_dict = state['model']
        classes_list = np.array(list(state['idx_to_class'].values()))
        args.classes_dict = state['idx_to_class']
        args.do_bottleneck_head = False
    
    # Load model
    model.load_state_dict(state_dict, strict=True)
    args.rank = 0

    # Adjust model for PASCAL-VOC dataset
    model.query_embed = torch.nn.Embedding(20, 1024)
    model.fc = GroupWiseLinear(20, 1024)

    # Freeze the weights of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Make the new query_embed and GroupWiseLinear layers trainable
    for param in model.query_embed.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # Cleanup intermediate variables
    del state
    del state_dict
    torch.cuda.empty_cache()
    file_print(args.logging_file, 'done\n')

    return model.cuda(), args, classes_list

def main():
    args = parser.parse_args()

    # Check if ViT is being used
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
        
    # Construct file path for saving metrics
    training_specifics = (f"training" + 
                        f"{f'_{args.lr_scheduler}' if args.lr_scheduler != 'none' else ''}" +
                        f"{f'_mixedprec' if args.amp else ''}" +
                        f"{f'_ema' if args.ema_decay_rate > 0 else ''}")

    foldername = f"/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/{args.dataset_name}/{'ViT' if is_ViT else 'resnet'}_trained/{training_specifics}/{todaystring}/trial_{args.trial}/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging.txt"
    
    # Setup model
    model, args, classes_list = load_model(args, is_ViT)
    model.train()
    file_print(args.logging_file, 'done\n')

    # PASCAL-VOC Data loading
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                        std=[1, 1, 1])

    data_path_train = os.path.join(args.data, 'train')

    val_dataset = VOCDetection(root = data_path_train,
                            year = "2007",
                            image_set = "val",
                            transform = transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
                                transforms.ToTensor(),
                                # normalize, # no need, toTensor does normalization
                            ]))

    train_dataset = VOCDetection(root = data_path_train,
                                year = "2007",
                                image_set = "train",
                                transform = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
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
    train_multi_label_pascal_voc(model, train_loader, val_loader, args.lr, args)

def train_multi_label_pascal_voc(model, train_loader, val_loader, lr, args):
    # Initialize EMA
    ema = ModelEma(model, args.ema_decay_rate) if (args.ema_decay_rate > 0) else None

    # set optimizer
    epochs = 15
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
        file_print(args.logging_file, f'\nEpoch [{epoch}/{epochs}]')
        model.train()
        file_print(args.logging_file, "starting training...")

        for batch_index, batch in enumerate(train_loader):
            input_data = batch[0]
            target = batch[1]

            input_data = input_data.cuda()
            target = target.cuda()    # (batch,3,num_classes)
            #target = target.max(dim=1)[0]  # ONLY DO THIS FOR MSCOCO!
            current_lr = scheduler.get_last_lr()[0] if scheduler else current_lr

            # Run forward pass
            with autocast() if scaler else nullcontext() as context:    # mixed precision
                output = model(input_data).float()    # sigmoid will be done in loss !
            
            loss = criterion(output, target)
            model.zero_grad()

            # Perform backprop
            scaler.scale(loss).backward() if scaler else loss.backward()
            scaler.step(optimizer) if scaler else optimizer.step()
            if scaler: scaler.update()
 
            if scheduler: scheduler.step()    # Apply learning rate scheduler
            if ema: ema.update(model)    # Update EMA checkpoints

            # display information
            if batch_index % 100 == 0:
                batch_info = f"Batch: [{batch_index}/{len(train_loader)}]"
                file_print(args.logging_file, f"{batch_info:<24}{'-->':<8}LR: {current_lr:.1e}, Loss: {loss.item():.1f}")
        
        save_model = ema.module if ema else model
        save_model.eval()
        average_loss = validate_multi(val_loader, save_model, args)

        # Save the checkpoints associated with current epoch
        if args.model_name == "Q2L-CvT_w24-384":
            checkpoints = {"state_dict":save_model.state_dict(), "epoch":epoch}
        else:
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
    file_print(args.logging_file, "\nstarting validation...")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    targets = []
    metrics = PerformanceMetrics(args.num_classes)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    total_loss = 0.0
    for batch_index, batch in enumerate(val_loader):
        input_data = batch[0]
        target = batch[1]

        #target = target.max(dim=1)[0] # ONLY DO THIS FOR MSCOCO!
        
        # compute output
        with torch.no_grad():
            with autocast() if args.amp else nullcontext():    # mixed precision
                output = model(input_data.cuda())
                output_regular = Sig(output).cpu()

        # The ASL loss in each batch is NOT the average of losses from each image - rather, it is the sum
        loss = criterion(output.cuda(), target.cuda())  # sigmoid will be done in loss !
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

        if (batch_index % 100 == 0):
            batch_info = f"Batch: [{batch_index}/{len(val_loader)}]"
            file_print(args.logging_file, f"{batch_info:<24}{'-->':<8}P_O: {precision_o:.2f}, R_O: {recall_o:.2f}, P_C: {precision_c:.2f}, R_C: {recall_c:.2f}, Loss: {loss.item():.2f}")
    
    average_loss = total_loss / len(val_loader.dataset)
    file_print(args.logging_file, f"{'[===Final Results===]':<24}{'-->':<8}P_O: {precision_o:.2f}, R_O: {recall_o:.2f}, P_C: {precision_c:.2f}, R_C: {recall_c:.2f}, Average Loss: {average_loss:.4f}")

    return average_loss


if __name__ == '__main__':
    main()
