# Adopted from: https://github.com/Alibaba-MIIL/ASL/blob/main/train.py

import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast
import os
import json

from pathlib import Path
from contextlib import nullcontext
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

# Add parent directory to path so we can import from utils
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from defenses.patchcleanser.pc_utils import gen_mask_set
from utils.datasets import TransformWrapper
from utils.cutout_augmentations import Cutout, GreedyCutout
from utils.common import file_print, load_model, load_eval_dataset, load_train_dataset
from train_utils import validate_multi_label, setup_training_components, save_model_checkpoints

sys.path.append(os.path.join(parent_dir, "packages/ASL/"))
from packages.ASL.src.models import create_model
from packages.ASL.src.loss_functions.losses import AsymmetricLoss

sys.path.append(os.path.join(parent_dir, "packages/query2labels/lib"))
from packages.query2labels.lib.models.query2label import build_q2l

parser = argparse.ArgumentParser(description='Defense fine-tuning for multi-label classifiers via cutout augmentations')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
# NOTE: This script in its current form only supports MSCOCO. To support PASCALVOC, the dataset loaders in
# utils/datasets.py and utils/common.py will need to be modified to operate with the TransformWrapper class.
parser.add_argument('--dataset-name', choices=["mscoco"], default="mscoco")
parser.add_argument('--num-classes', default=80, type=int, help='number of classes (default: 80)')
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size (default: 64)')

# Model specifics
available_models = ['tresnet_l', 'Q2L-CvT_w24-384']
parser.add_argument('--model-name', choices=available_models, default='tresnet_l')
parser.add_argument('--model-path', type=str)
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model. default is False. ')
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')

# * Transformer config file (optional, required for ViT models)
parser.add_argument('--config', type=str, default=None, help='config file containing all ViT parameters')

# Cutout specifics
parser.add_argument('--cutout-type', choices=["randomcutout", "greedycutout"], default="greedycutout")
parser.add_argument('--cutout-size', default=224, type=int, help='size of random cutout masks (default: 224)')
parser.add_argument('--greedy-cutout-path', default='./greedy_cutout_dict.json', type=str)

# Training specifics
parser.add_argument('--lr', default=1e-4, type=float, help='maximum learning rate (default: 1e-4)')
parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision (AMP); to disable, run --no-amp as the arg')
parser.add_argument('--no-amp', dest='amp', action='store_false', help='disable automatic mixed precision (AMP); to enable, run --amp as the arg')
parser.set_defaults(amp=True)
parser.add_argument('--lr-scheduler', choices=["onecyclelr", "linear", "none"], default="onecyclelr", help='learning rate scheduler (default: onecyclelr)')
parser.add_argument('--ema-decay-rate', default=0.9997, type=float, help='expected moving average (EMA) decay rate (default: 0.9997)')

# Miscellaneous
parser.add_argument('--trial', default=1, type=int, help='trial (default: 1)')
parser.add_argument('--cache-dir', type=str, help='directory for storing checkpoints')

def main():
    args = parser.parse_args()

    if args.model_name == "Q2L-CvT_w24-384":
        is_ViT = True
        if args.config is None:
            raise ValueError("--config parameter is required when using ViT models (Q2L-CvT_w24-384)")
    else:
        is_ViT = False

    # Choose between different types of cutout
    if args.cutout_type == "randomcutout":
        cutout_str = f"randomcutout/size_{args.cutout_size}"
        cutout_transform = TransformWrapper(Cutout(n_holes=2, length=args.cutout_size))
    elif args.cutout_type == "greedycutout":
        with open(args.greedy_cutout_path) as f:
            greedy_cutout_file = json.load(f)

        # Extract metadata from the greedy cutout dict
        greedy_cutout_metadata = greedy_cutout_file["metadata"]
        greedy_cutout_data = greedy_cutout_file["data"]
        
        # Create R-covering mask set
        mask_list, mask_size, mask_stride = gen_mask_set([greedy_cutout_metadata["image_size"], greedy_cutout_metadata["image_size"]], 
                                                        [greedy_cutout_metadata["patch_size"], greedy_cutout_metadata["patch_size"]], 
                                                        [greedy_cutout_metadata["mask_number_fr"], greedy_cutout_metadata["mask_number_fr"]])

        # Generate string-based identifier
        cutout_str = f"greedycutout/patch_{greedy_cutout_metadata['patch_size']}_masknum_{greedy_cutout_metadata['mask_number_fr']}"

        # Initialize greedy cutout transform
        cutout_transform = GreedyCutout(mask_list=mask_list, greedy_cutout_data=greedy_cutout_data)
        
    # Construct file path for saving metrics
    training_specifics = (f"training" + 
                        f"{f'_{args.lr_scheduler}' if args.lr_scheduler != 'none' else ''}" +
                        f"{f'_mixedprec' if args.amp else ''}" +
                        f"{f'_ema' if args.ema_decay_rate > 0 else ''}")

    foldername = os.path.join(args.cache_dir, f"checkpoints/{args.dataset_name}/{'ViT' if is_ViT else 'resnet'}_trained/{cutout_str}/{training_specifics}/{todaystring}/trial_{args.trial}/")
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging.txt"
    
    # Setup model
    model, args = load_model(args, is_ViT)
    model.train()
    file_print(args.logging_file, 'done\n')

    # Data loading
    val_dataset = load_eval_dataset(args)
    train_dataset = load_train_dataset(args, cutout_transform)
    
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
    cutout_defense_finetune(model, train_loader, val_loader, args.lr, args)

def cutout_defense_finetune(model, train_loader, val_loader, lr, args):
    epochs = 10
    steps_per_epoch = len(train_loader)
    
    # Setup training components
    optimizer, scheduler, ema, scaler = setup_training_components(
        model, args, steps_per_epoch, epochs, lr)
    
    # Set criterion
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    current_lr = lr

    lowest_val_loss = np.inf
    for epoch in range(epochs):
        file_print(args.logging_file, f'\nEpoch [{epoch}/{epochs}]')
        model.train()
        file_print(args.logging_file, "starting training...")

        for batch_index, batch in enumerate(train_loader):
            input_data = batch[0]
            target = batch[1]

            if args.dataset_name == "mscoco":
                target = target.max(dim=1)[0]

            input_data = input_data.cuda()
            target = target.cuda()    # (batch,3,num_classes)
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
        average_loss = validate_multi_label(val_loader, save_model, args, args.dataset_name)

        # Save model checkpoints
        lowest_val_loss = save_model_checkpoints(save_model, args.save_dir, args.model_name, epoch, average_loss, lowest_val_loss, ema)

if __name__ == '__main__':
    main()
