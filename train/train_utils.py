# Shared training and validation utilities for multi-label classification scripts

import torch
from torch.cuda.amp import autocast
from contextlib import nullcontext

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.metrics import PerformanceMetrics
from utils.common import file_print

sys.path.append(os.path.join(parent_dir, "packages/ASL/"))
from packages.ASL.src.loss_functions.losses import AsymmetricLoss

def validate_multi_label(val_loader, model, args, dataset_name=None):
    """
    Run validation for multi-label classification models.
    
    Args:
        val_loader: DataLoader for validation data
        model: Multi-label classification model
        args: Arguments containing num_classes, thre, amp, logging_file
        dataset_name: Name of dataset for target processing ("mscoco", "pascalvoc", etc.)
    
    Returns:
        float: Average validation loss
    """
    file_print(args.logging_file, "\nstarting validation...")
    Sig = torch.nn.Sigmoid()
    metrics = PerformanceMetrics(args.num_classes)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    total_loss = 0.0
    
    for batch_index, batch in enumerate(val_loader):
        input_data = batch[0]
        target = batch[1]

        # Dataset-specific target processing
        if dataset_name == "mscoco":
            target = target.max(dim=1)[0]
        
        # compute output
        with torch.no_grad():
            with autocast() if args.amp else nullcontext():    # mixed precision
                output = model(input_data.cuda())
                output_regular = Sig(output).cpu()

        # The ASL loss in each batch is NOT the average of losses from each image - rather, it is the sum
        loss = criterion(output.cuda(), target.cuda())  # sigmoid will be done in loss !
        total_loss += loss.item()

        # for metrics
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

def setup_training_components(model, args, steps_per_epoch, epochs, lr):
    """
    Setup training components (optimizer, scheduler, EMA, scaler).
    
    Args:
        model: Multi-label classification model
        args: Arguments containing lr_scheduler, amp, ema_decay_rate
        steps_per_epoch: Number of steps per epoch
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        tuple: (optimizer, scheduler, ema, scaler)
    """
    from utils.model_ema import ModelEma
    from torch.cuda.amp import GradScaler
    from torch.optim import lr_scheduler
    
    # Initialize EMA
    ema = ModelEma(model, args.ema_decay_rate) if (args.ema_decay_rate > 0) else None

    # Set optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0)
    scaler = GradScaler() if args.amp else None

    # Initialize scheduler
    scheduler = None
    if (args.lr_scheduler == "onecyclelr"):
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs, pct_start=0.2)
    elif (args.lr_scheduler == "linear"):
        scheduler = lr_scheduler.LinearLR(optimizer, total_iters = 5 * steps_per_epoch)
    
    return optimizer, scheduler, ema, scaler

def save_model_checkpoints(save_model, save_dir, model_name, epoch, average_loss, lowest_val_loss, ema=None):
    """
    Save model checkpoint for current epoch and best model if improved.
    
    Args:
        save_model: Multi-label classification model
        save_dir: Path to save model checkpoints
        model_name: Name of the model architecture
        epoch: Current epoch number
        average_loss: Current validation loss
        lowest_val_loss: Best validation loss so far
        ema: EMA model (optional)
    
    Returns:
        float: Updated lowest validation loss
    """
    from pathlib import Path
    
    # Create checkpoint dictionary
    if model_name == "Q2L-CvT_w24-384":
        checkpoints = {"state_dict":save_model.state_dict(), "epoch":epoch}
    else:
        checkpoints = {"model":save_model.state_dict(), "epoch":epoch}
    
    # Save epoch checkpoint
    try:
        save_model_dir = save_dir + f'epoch_{epoch}/'
        Path(save_model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(checkpoints, save_model_dir + f"{'ema-' if ema else ''}model-epoch-{epoch}.pth")
    except:
        pass

    # Save best model if improved
    if (average_loss < lowest_val_loss):
        lowest_val_loss = average_loss
        try:
            torch.save(checkpoints, save_dir + f"{'ema-' if ema else ''}model-best-epoch-{epoch}.pth")
        except:
            pass
    
    return lowest_val_loss