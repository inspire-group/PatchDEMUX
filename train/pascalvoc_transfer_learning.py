# Adopted from: https://github.com/Alibaba-MIIL/ASL/blob/main/train.py

import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast
import os

from pathlib import Path
from contextlib import nullcontext
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

# Add parent directory to path so we can import from utils
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.common import file_print, load_model, load_val_dataset, load_train_dataset
from train_utils import validate_multi_label, setup_training_components, save_model_checkpoints

sys.path.append(os.path.join(parent_dir, "packages/ASL/"))
from packages.ASL.src.loss_functions.losses import AsymmetricLoss

sys.path.append(os.path.join(parent_dir, "packages/query2labels/lib"))
from packages.query2labels.lib.models.query2label import GroupWiseLinear

parser = argparse.ArgumentParser(description='Transfer learning on Pascal VOC dataset')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size (default: 64)')

# Model specifics
# NOTE: This script only supports transfer learning for ViT-based models. PASCALVOC transfer learning can be done
# for Resnet models by adjusting the relevant layers in lines 87 through 102.
available_models = ['Q2L-CvT_w24-384']
parser.add_argument('--model-name', choices=available_models, default='Q2L-CvT_w24-384')
parser.add_argument('--model-path', type=str)
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model. default is False. ')
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')

# * Transformer config file (required for ViT models)
parser.add_argument('--config', type=str, help='config file containing all ViT parameters')

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

    # This script only supports ViT models for transfer learning
    if args.model_name == "Q2L-CvT_w24-384":
        is_ViT = True
        if args.config is None:
            raise ValueError("--config parameter is required when using ViT models (Q2L-CvT_w24-384)")
    else:
        raise ValueError("This transfer learning script only supports ViT models (Q2L-CvT_w24-384)")

    # Construct file path for saving metrics
    training_specifics = (f"training" + 
                        f"{f'_{args.lr_scheduler}' if args.lr_scheduler != 'none' else ''}" +
                        f"{f'_mixedprec' if args.amp else ''}" +
                        f"{f'_ema' if args.ema_decay_rate > 0 else ''}")

    foldername = os.path.join(args.cache_dir, f"checkpoints/pascalvoc/ViT_trained/{training_specifics}/{todaystring}/trial_{args.trial}/")
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername
    args.logging_file = foldername + "logging.txt"

    # Setup model - note that the config file should have num_class = 80 in order to properly load the MSCOCO weights
    # (if the base model was trained on another dataset, then num_class should be set to the relevant number of classes)
    model, args = load_model(args, is_ViT)
    
    # Adjust model for PASCAL-VOC dataset (transfer learning specific modifications for ViT models)
    args.dataset_name = "pascalvoc"
    args.num_classes = 20

    model.query_embed = torch.nn.Embedding(args.num_classes, 1024)
    model.fc = GroupWiseLinear(args.num_classes, 1024)
    
    # Freeze the weights of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False
    
    # Make the new query_embed and GroupWiseLinear layers trainable
    for param in model.query_embed.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    
    model.train()
    model = model.cuda()
    file_print(args.logging_file, 'done\n')

    # Data loading
    val_dataset = load_val_dataset(args)
    train_dataset = load_train_dataset(args)

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
    pascalvoc_transfer_learn(model, train_loader, val_loader, args.lr, args)

def pascalvoc_transfer_learn(model, train_loader, val_loader, lr, args):
    epochs = 15
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
