# Adopted from: https://github.com/Alibaba-MIIL/ASL/blob/main/validate.py

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import os

from utils.defense import gen_mask_set
from utils.datasets import CocoDetection

import sys
sys.path.append("ASL/")
from ASL.src.models import create_model

parser = argparse.ArgumentParser(description='Multi-Label PatchCleanser Image Visualization')

# Dataset specifics
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--num-classes', default=80)
parser.add_argument('--image-size', default=448, type=int, help='input image size (default: 448)')

# Model specifics
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='./TRresNet_L_448_86.6.pth', type=str)
parser.add_argument('--thre', default=0.8, type=float, help='threshold value')

# Mask set specifics
parser.add_argument('--patch-size', default=64, type=int, help='patch size (default: 64)')
parser.add_argument('--mask-number', default=6, type=int, help='mask number (default: 6)')

# Visualization specifics
parser.add_argument('--image-index', default=0, type=int, help='index in the dataset to visualize (default: 0)')
parser.add_argument('--first-mask-index', default=0, type=int, help='first mask index to cover image (default: 0)')
parser.add_argument('--second-mask-index', default=0, type=int, help='second mask index to cover image (default: 0)')
parser.add_argument('--clean-im', action='store_true', help='perform inference on the clean image; to run on masked image, use --no-clean-im as the arg')
parser.add_argument('--no-clean-im', dest='clean_im', action='store_false', help='perform inference on the masked image; to run on the clean image, use --clean-im as the arg')
parser.set_defaults(clean_im=False)

def main():
    args = parser.parse_args()

    # setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    args.do_bottleneck_head = False
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
                                    normalize,
                                ]))

    print("len(val_dataset)): ", len(val_dataset))
    
    # Create R-covering set of masks
    im_size = [args.image_size, args.image_size]
    patch_size = [args.patch_size, args.patch_size]
    mask_number = [args.mask_number, args.mask_number]
    mask_list, mask_size, mask_stride = gen_mask_set(im_size, patch_size, mask_number)

    # Visualize the image at selected index
    visualize_image(model, val_dataset, classes_list, mask_list, args)


def visualize_image(model, val_dataset, classes_list, mask_list, args):
    print("starting visualization...")
    Sig = torch.nn.Sigmoid()

    # target shape: [batch_size, object_size_channels, number_classes]
    im, target = val_dataset[args.image_index]

    # torch.max returns (values, indices), additionally squeezes along the dimension dim
    target = target.max(dim=0)[0]
 
    # Obtain specified masks
    mask1 = mask_list[args.first_mask_index].unsqueeze(0)
    mask2 = mask_list[args.second_mask_index].unsqueeze(0)

    masked_im = torch.where(torch.logical_and(mask1, mask2), im.cuda(), torch.tensor(0.0).cuda())

    if (args.clean_im):
        masked_im = im.cuda()

    # Compute output
    with torch.no_grad():
        output = Sig(model((masked_im.unsqueeze(0)).cuda())).cpu()

    pred = output.data.gt(args.thre).long()

    # Better way to get detected classes????
    detected_classes = []
    for i in range(80):
        if (pred[0][i]):
            detected_classes.append(classes_list[i])

    # Displaying image
    mask_im = torch.logical_not(torch.logical_and(mask1, mask2)).cpu().numpy().transpose((1, 2, 0)).astype(float)
    #plt.imsave("COCO_val2014_000000000143.png", im.cpu().numpy().transpose((1, 2, 0)))
    
    # Use this to save an image of the masks themselves
    #plt.imsave("testmask.png", np.repeat(mask_im, 3, axis=2))

    print('showing image on screen...')
    fig = plt.figure()
    plt.imshow(masked_im.cpu().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.axis('tight')
    plt.title(f"detected classes: {detected_classes}")    
    plt.show()

    return


if __name__ == '__main__':
    main()
