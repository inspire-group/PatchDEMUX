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

from pathlib import Path
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

from utils.defense import gen_mask_set
from utils.metrics import PerformanceMetrics
from utils.datasets import CocoDetection

import sys
sys.path.append("ASL/")
from ASL.src.models import create_model

parser = argparse.ArgumentParser(description='Multi-Label PatchCleanser Certification')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', choices=["mscoco", "nuswide", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80)
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size (default: 32)')

# Model specifics
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='./TRresNet_L_448_86.6.pth', type=str)
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')

# Mask set specifics
parser.add_argument('--patch-size', default=64, type=int, help='patch size (default: 64)')
parser.add_argument('--mask-number', default=6, type=int, help='mask number (default: 6)')

# Miscellaneous
parser.add_argument('--trial', default=1, type=int, help='trial (default: 1)')
parser.add_argument('--print-freq', '-p', default=64, type=int, help='print frequency (default: 64)')

def main(rank, world_size):

    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

    # Create process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    args = parser.parse_args()
    args.batch_size = args.batch_size

    # Construct file path for saving metrics
    foldername = f"dump/certification/{args.dataset_name}/patch_{args.patch_size}_masknum_{args.mask_number}/{todaystring}/trial_{args.trial}/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername

    # setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    args.do_bottleneck_head = False
    args.rank = rank
    # model = create_model(args).cuda()

    print(f"Rank before create model is: {rank}")

    model = create_model(args).to(rank)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    classes_list = np.array(list(state['idx_to_class'].values()))
    print('done\n')

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

    print("len(val_dataset)): ", len(val_dataset))

    gpu_val_dataset = torch.utils.data.Subset(val_dataset, list(range(50 * rank, 50 * (rank + 1))))

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
        gpu_val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Create R-covering set of masks
    im_size = [args.image_size, args.image_size]
    patch_size = [args.patch_size, args.patch_size]
    mask_number = [args.mask_number, args.mask_number]
    mask_list, mask_size, mask_stride = gen_mask_set(im_size, patch_size, mask_number)

    validate_multi(model, val_loader, classes_list, mask_list, args)

    dist.destroy_process_group()


def validate_multi(model, val_loader, classes_list, mask_list, args):
    print("starting actual validation")

    Sig = torch.nn.Sigmoid()

    preds = []
    targets = []
    num_masks = len(mask_list)
    num_classes = len(classes_list)

    metrics = PerformanceMetrics(num_classes)

    # target shape: [batch_size, object_size_channels, number_classes]
    for batch_index, (input, target) in enumerate(val_loader):

        if args.rank == 0:
            print(f'Batch: [{batch_index}/{len(val_loader)}]')
    
        if (batch_index == 1):
            break

        # torch.max returns (values, indices), additionally squeezes along the dimension dim
        target = target.max(dim=1)[0]
        
        im = input
        target = target.cpu().numpy()

        all_preds = np.zeros([args.batch_size, num_masks * num_masks, num_classes])

        for i, mask1 in enumerate(mask_list):
            mask1 = mask1.reshape(1, 1, *mask1.shape).to(args.rank)

            if (args.rank == 0):
                print(f"the rank 0 mask1 is: {i}")
            
            if (args.rank == 1):
                print(f"the rank 1 mask1 is: {i}")

            for j, mask2 in enumerate(mask_list):
                mask2 = mask2.reshape(1, 1, *mask2.shape).to(args.rank)

                # masked_im = torch.where(torch.logical_and(mask1, mask2), im.cuda(), torch.tensor(0.0).cuda())
                masked_im = torch.where(torch.logical_and(mask1, mask2), im.to(args.rank), torch.tensor(0.0).to(args.rank))

                # compute output
                with torch.no_grad():
                    # output = Sig(model(masked_im).cuda()).cpu()
                    output = Sig(model(masked_im).to(args.rank)).cpu()

                pred = output.data.gt(args.thre).long()
                all_preds[:, i * num_masks + j] = pred.cpu().numpy()
        
        # Find which classes had consensus in masked predictions
        all_preds_ones = np.all(all_preds, axis=1)
        all_preds_zeros = np.all(np.logical_not(all_preds), axis=1)
        
        # Compute certified TP, TN, FN, FP
        confirmed_tp = np.logical_and(all_preds_ones, target).astype(int)
        confirmed_tn = np.logical_and(all_preds_zeros, np.logical_not(target)).astype(int)
        worst_case_fn = np.logical_and(np.logical_not(all_preds_ones), target).astype(int)
        worst_case_fp = np.logical_and(np.logical_not(all_preds_zeros), np.logical_not(target)).astype(int)

        # Compute certified precision and recall
        metrics.updateMetrics(TP=confirmed_tp, TN=confirmed_tn, FN=worst_case_fn, FP=worst_case_fp)
        precision_o, recall_o = metrics.overallPrecision(), metrics.overallRecall()
        precision_c, recall_c = metrics.averageClassPrecision(), metrics.averageClassRecall()

        if (args.rank == 0): 
            print(f'P_O {precision_o:.2f} R_O {recall_o:.2f} P_C {precision_c:.2f} R_C {recall_c:.2f}\n')

    # Save the certified TP, TN, FN, FP
    if (args.rank == 0):
        np.savez(args.save_dir + "certified_metrics", TP=metrics.TP, TN=metrics.TN, FN=metrics.FN, FP=metrics.FP)

        # Save the precision and recall metrics
        with open(args.save_dir + "performance_metrics.txt", "w") as f:
            f.write(f"P_O: {precision_o:.2f} \t R_O: {recall_o:.2f}\n")
            f.write(f"P_C: {precision_c:.2f} \t R_C: {recall_c:.2f}")

    if (args.rank == 1):
        np.savez(args.save_dir + "certified_metrics1", TP=metrics.TP, TN=metrics.TN, FN=metrics.FN, FP=metrics.FP)

        # Save the precision and recall metrics
        with open(args.save_dir + "performance_metrics1.txt", "w") as f:
            f.write(f"P_O: {precision_o:.2f} \t R_O: {recall_o:.2f}\n")
            f.write(f"P_C: {precision_c:.2f} \t R_C: {recall_c:.2f}")

# CHECK WHY THE ORDER OF IMAGES IS WRONG!!!
# CHECK HOW METRICS DO WITH LARGER BATCH!!!
# ENSURE THAT THE MASKED PREDICTIONS ARE CORRECT - USE VISUALIZATION SCRIPT TO CHECK
#########################################################

        # tp += (pred + target).eq(2).sum(dim=0)
        # fp += (pred - target).eq(1).sum(dim=0)
        # fn += (pred - target).eq(-1).sum(dim=0)
        # tn += (pred + target).eq(0).sum(dim=0)
        # count += input.size(0)

        # this_tp = (pred + target).eq(2).sum()
        # this_fp = (pred - target).eq(1).sum()
        # this_fn = (pred - target).eq(-1).sum()
        # this_tn = (pred + target).eq(0).sum()

        # this_prec = this_tp.float() / (
        #     this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        # this_rec = this_tp.float() / (
        #     this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        # prec.update(float(this_prec), input.size(0))
        # rec.update(float(this_rec), input.size(0))

        # # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
        #                                                                      i] > 0 else 0.0
        #        for i in range(len(tp))]
        # r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
        #                                                                      i] > 0 else 0.0
        #        for i in range(len(tp))]
        # f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
        #        i in range(len(tp))]

        # mean_p_c = sum(p_c) / len(p_c)
        # mean_r_c = sum(r_c) / len(r_c)
        # mean_f_c = sum(f_c) / len(f_c)

        # p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        # r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        # f_o = 2 * p_o * r_o / (p_o + r_o)

        # if i % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
        #           'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
        #         i, len(val_loader), batch_time=batch_time,
        #         prec=prec, rec=rec))
        #     print(
        #         'P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
        #             .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    # print(
    #     '--------------------------------------------------------------------')
    # print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
    #       .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    # mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    # print("mAP score:", mAP_score)

    return

if __name__ == '__main__':
    world_size = 2  # number of gpus to parallize over
    mp.spawn(main, args=(world_size, ), nprocs=world_size, join=True)
