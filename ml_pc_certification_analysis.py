# TODO:
# - SHOULD GO ALONG WITH ML_PC_VISUALIZATION AS AN OBSERVATION LEVEL SCRIPT...

# Adopted from: https://github.com/Alibaba-MIIL/ASL/blob/main/validate.py

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import os

from pathlib import Path
from datetime import date
todaystring = date.today().strftime("%m-%d-%Y")

from utils.defense import gen_mask_set
from utils.datasets import CocoDetection

import sys
sys.path.append("packages/ASL/")
from packages.ASL.src.models import create_model

# Sample execution is as follows:
# DATA_DIR="/scratch/gpfs/djacob/multi-label-patchcleanser/coco/"
# MODEL_PATH="/scratch/gpfs/djacob/multi-label-patchcleanser/checkpoints/mscoco/MS_COCO_TRresNet_L_448_86.6.pth"
# python ml_pc_certification_analysis.py $DATA_DIR --model-path $MODEL_PATH --thre 0.8 --patch-size 64 --mask-number 6 --image-index 0
parser = argparse.ArgumentParser(description='Multi-Label PatchCleanser Certification Analysis')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', choices=["mscoco", "nuswide", "pascalvoc"], default="mscoco")
parser.add_argument('--num-classes', default=80)
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')

# Model specifics
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='./TRresNet_L_448_86.6.pth', type=str)
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')

# Mask set specifics
parser.add_argument('--patch-size', default=64, type=int, help='patch size (default: 64)')
parser.add_argument('--mask-number', default=6, type=int, help='mask number (default: 6)')

# Analysis specifics
parser.add_argument('--image-index', default=0, type=int, help='index in the dataset to visualize (default: 0)')

def file_print(file_path, msg):
    with open(file_path, "a") as f:
        print(msg, flush=True, file=f) 

def main():
    args = parser.parse_args()

    # plan: take in the desired image and look at the target. We need to consider both true positives and true negatives when we certify
    #  - consider true positives. for the classifier consider all pairs of masks from the mask set, and evaluate. if certified, ignore the class.
    #    otherwise, find all pairs of masks which lead to a false negative and list them out in a txt file called "im_id_analysis.txt"
    #  - consider true negatives. apply a similar approach as above

    # maybe one thing is also consdiering going over every image in the dataset and collecting a dataset-wide statistic on the number of times we have shared mask combinations screwing up certification

    # Construct file path for saving metrics
    foldername = f"dump/certification_analysis/{args.dataset_name}/patch_{args.patch_size}_masknum_{args.mask_number}_thre_{(int)(args.thre * 100)}/{todaystring}/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    args.save_dir = foldername

    # Setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    args.do_bottleneck_head = False
    args.rank = 0

    # Create model
    model = create_model(args).cuda()
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
                                    # normalize, # no need, toTensor does normalization
                                ]))

    args.img_id = val_dataset.ids[args.image_index]
    args.logging_file = foldername + f"imgid_{args.img_id}_logging.txt"
    print("len(val_dataset)): ", len(val_dataset))
    
    # Create R-covering set of masks
    im_size = [args.image_size, args.image_size]
    patch_size = [args.patch_size, args.patch_size]
    mask_number = [args.mask_number, args.mask_number]
    mask_list, mask_size, mask_stride = gen_mask_set(im_size, patch_size, mask_number)

    # Visualize the image at selected index
    analyze_certification(model, val_dataset, classes_list, mask_list, args)

def log_forgotten_classes_analysis(args, all_preds, forgotten_class, class_type, num_masks):
    failure_num = 0 if class_type == "tp" else 1

    # Find list of double masks which cause failure in certification for the forgotten class
    mask_idx, class_idx = (all_preds[:, forgotten_class[0]] == failure_num).nonzero()
    combined_idx = np.concatenate((mask_idx[np.newaxis, ...], class_idx[np.newaxis, ...]), axis = 0)

    sorted_forgotten_class_idx = np.argsort(combined_idx[1], kind="stable")
    sorted_combined_idx = combined_idx[:, sorted_forgotten_class_idx]

    # Iterate over all failed mask combinations
    curr_idx = -1
    for idx, forgotten_idx in enumerate(sorted_combined_idx[1]):
        if(forgotten_idx != curr_idx):
            # Initialize a dictionary to assist with skipping over any duplicate mask combinations which appear
            duplicate_dict = {}

            file_print(args.logging_file, f"Class {forgotten_class[0][forgotten_idx]}")
            curr_idx = forgotten_idx

        first_mask = sorted_combined_idx[0, idx] // num_masks
        second_mask = sorted_combined_idx[0, idx] % num_masks

        mask_comb = f"({first_mask}, {second_mask})"
        mask_comb_flipped = f"({second_mask}, {first_mask})"
        if mask_comb_flipped in duplicate_dict:
            continue
        else:
            duplicate_dict[mask_comb] = 1

            # Print the mask combination which causes failure in certification
            file_print(args.logging_file, f"\tMask combination: ({sorted_combined_idx[0, idx] // num_masks}, {sorted_combined_idx[0, idx] % num_masks})")

def analyze_certification(model, val_dataset, classes_list, mask_list, args):
    print("starting analysis...")
    Sig = torch.nn.Sigmoid()

    preds = []
    targets = []
    num_masks = len(mask_list)
    num_classes = len(classes_list)

    # target shape: [batch_size, object_size_channels, number_classes]
    im, target = val_dataset[args.image_index]

    # torch.max returns (values, indices), additionally squeezes along the dimension dim
    target = target.max(dim=0)[0]
    target = target.cpu().numpy()
 
    # Initialize all_preds to -1 in order to filter out unused mask combinations at the end
    all_preds = np.zeros([num_masks * num_masks, num_classes]) - 1
    for i, mask1 in enumerate(mask_list):
        mask1 = mask1.reshape(1, 1, *mask1.shape).to(args.rank)            

        print(f"Certification is {(i / num_masks) * 100:0.2f}% complete!")
        for j, mask2 in enumerate(mask_list):
            mask2 = mask2.reshape(1, 1, *mask2.shape).to(args.rank)
            masked_im = torch.where(torch.logical_and(mask1, mask2), im.to(args.rank), torch.tensor(0.0).to(args.rank))

            # compute output
            with torch.no_grad():
                output = Sig(model(masked_im).to(args.rank)).cpu()

            pred = output.data.gt(args.thre).long()
            all_preds[i * num_masks + j] = pred.cpu().numpy()
  
    # Find which classes had consensus in masked predictions
    all_preds_ones = np.all(all_preds, axis=0)
    all_preds_zeros = np.all(np.logical_not(all_preds), axis=0)

    # Compute certified TP, TN, FN, FP
    confirmed_tp = np.logical_and(all_preds_ones, target).astype(int)
    confirmed_tn = np.logical_and(all_preds_zeros, np.logical_not(target)).astype(int)

    # Determine the indices where certification failed to recover a TP
    forgotten_tp = (target - confirmed_tp == 1).nonzero()

    # Determine the indices where certification failed to recover a TN
    forgotten_tn = (target - confirmed_tn == 0).nonzero()

    #####################################################################
    #                     Start of logging file...                      #
    #####################################################################

    file_print(args.logging_file, f"Logging file for MSCOCO image {args.img_id}...")
    file_print(args.logging_file, f"Target vector is: {target}\n")
    
    # Log each of the true positives in the target label
    target_tp_classes = (target == 1).nonzero()[0]
    file_print(args.logging_file, "TRUE POSITIVES ANALYSIS")
    file_print(args.logging_file, f"List of true positive classes:")
    for class_idx in target_tp_classes:
        file_print(args.logging_file, f"Class {class_idx}")

    # Log each of the forgotten true positives, along with the set double masks which result in a FN
    file_print(args.logging_file, "\nFORGOTTEN TRUE POSITIVES ANALYSIS")
    log_forgotten_classes_analysis(args, all_preds, forgotten_tp, "tp", num_masks)

    # Log each of the true negatives in the target label
    target_tn_classes = (target == 0).nonzero()[0]
    file_print(args.logging_file, "\nTRUE NEGATIVES ANALYSIS")
    file_print(args.logging_file, f"List of true negative classes:")
    for class_idx in target_tn_classes:
        file_print(args.logging_file, f"Class {class_idx}")

    # Log each of the forgotten true negatives, along with the set double masks which result in a FP
    file_print(args.logging_file, "\nFORGOTTEN TRUE NEGATIVES ANALYSIS")
    log_forgotten_classes_analysis(args, all_preds, forgotten_tn, "tn", num_masks)

    return


if __name__ == '__main__':
    main()
