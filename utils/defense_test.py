import numpy as np 
import torch 
from dataclasses import dataclass

def gen_1D_mask_set(input_len, patch_len, mask_number):
    # generate a R-covering mask set in 1D
    '''
    INPUT:
    im_len          length of input
    patch_len       estimated patch length
    mask_number     computation budget

    OUTPUT:
    mask_list       the generated R-covering mask set
    mask_size       the mask size
    masks_stride     the mask stride
    '''

    # Determine mask stride and mask size
    mask_stride = int(np.ceil((input_len - patch_len + 1) / mask_number))
    mask_size = patch_len + mask_stride - 1    

    # Generate mask set (maybe rewrite using concatenate...)
    # masks_list = [0] * mask_number
    # base_mask = np.concatenate((mask_size * [0], (input_size - mask_size) * [1]))[..., np.newaxis].astype(bool)
    # for k in np.arange(mask_number - 1):
    #     masks_list[k] = np.roll(base_mask, k * mask_stride)
      
    masks_list = [0] * mask_number
    for index, start in enumerate(np.arange(mask_number - 1) * mask_stride):
        mask = np.ones((input_len, 1))
        mask[start : start + mask_size] = 0

        masks_list[index] = mask.astype(bool)
    
    # Account for final mask
    # masks_list[-1] = np.concatenate(((input_size - mask_size) * [1], mask_size * [0]))[..., np.newaxis].astype(bool)
    masks_list[-1] = np.concatenate((np.ones((input_len - mask_size, 1)), np.zeros((mask_size, 1)))).astype(bool)

    return masks_list, mask_size, mask_stride

def gen_mask_set(im_size, patch_size, mask_number):
    # generate a R-covering mask set for a 2D image
    '''
    INPUT:
    im_size         size of the input image along both axes, (width, height)
    patch_size      estimated patch size along both axes, (width, height)
    mask_number     computation budget along both axes, (width, height)

    OUTPUT:
    mask_list       list of torch.tensor, the generated R-covering mask set, the binary masks are moved to CUDA
    mask_size       the mask size along both axes, (width, height)
    mask_stride     the mask stride along both axes, (width, height)
    '''
    # Initialize variables
    dim = len(im_size)
    mask_size = np.zeros(dim,)
    mask_stride = np.zeros(dim,)

    # Create 1D masks for each axis
    width_masks, mask_size[0], mask_stride[0] = gen_1D_mask_set(im_size[0], patch_size[0], mask_number[0])
    height_masks, mask_size[1], mask_stride[1] = gen_1D_mask_set(im_size[1], patch_size[1], mask_number[1])

    # Combine 1D masks to create 2D image masks
    mask_list = [torch.from_numpy(w_mask | h_mask.T).cuda() for w_mask in width_masks for h_mask in height_masks]

    return mask_list, mask_size, mask_stride

def gen_mask_set_OLD(args,ds_config):
    # generate a R-covering mask set
    '''
    INPUT:
    args            argparse.Namespace, the set of argumements/hyperparamters for mask set generation
    ds_config       dict, data preprocessing dict 

    OUTPUT:
    mask_list       list of torch.tensor, the generation R-covering mask set, the binary masks are moved to CUDA
    MASK_SIZE       tuple (int,int), the mask size along two axes
    MASK_STRIDE     tuple (int,int), the mask stride along two axes
    '''
    # generate mask set
    assert args.mask_stride * args.num_mask < 0 #can only set either mask_stride or num_mask

    IMG_SIZE = (ds_config['input_size'][1],ds_config['input_size'][2])

    if args.pa>0 and args.pb>0: #rectangle patch
        PATCH_SIZE = (args.pa,args.pb)
    else: #square patch
        PATCH_SIZE = (args.patch_size,args.patch_size)

    if args.mask_stride>0: #use specified mask stride
        MASK_STRIDE = (args.mask_stride,args.mask_stride)
    else: #calculate mask stride based on the computation budget
        MASK_STRIDE = (int(np.ceil((IMG_SIZE[0] - PATCH_SIZE[0] + 1)/args.num_mask)),int(np.ceil((IMG_SIZE[1] - PATCH_SIZE[1] + 1)/args.num_mask)))

    # calculate mask size
    MASK_SIZE = (min(PATCH_SIZE[0]+MASK_STRIDE[0]-1,IMG_SIZE[0]),min(PATCH_SIZE[1]+MASK_STRIDE[1]-1,IMG_SIZE[1]))

    mask_list = []
    idx_list1 = list(range(0,IMG_SIZE[0] - MASK_SIZE[0] + 1,MASK_STRIDE[0]))
    if (IMG_SIZE[0] - MASK_SIZE[0])%MASK_STRIDE[0]!=0:
        idx_list1.append(IMG_SIZE[0] - MASK_SIZE[0])
    idx_list2 = list(range(0,IMG_SIZE[1] - MASK_SIZE[1] + 1,MASK_STRIDE[1]))
    if (IMG_SIZE[1] - MASK_SIZE[1])%MASK_STRIDE[1]!=0:
        idx_list2.append(IMG_SIZE[1] - MASK_SIZE[1])

    for x in idx_list1:
        for y in idx_list2:
            mask = torch.ones([1,1,IMG_SIZE[0],IMG_SIZE[1]],dtype=bool).cuda()
            mask[...,x:x+MASK_SIZE[0],y:y+MASK_SIZE[1]] = False
            mask_list.append(mask)
    return mask_list,MASK_SIZE,MASK_STRIDE

# New code
im_size = [224, 224]
patch_size = [32, 32]
mask_number = [6, 6]

# import time

# t0 = time.time()
mask_list, mask_size, mask_stride = gen_mask_set(im_size, patch_size, mask_number)
# t1 = time.time()
# print(f"my method: {t1 - t0}")

import time
# Old code
@dataclass
class Args:
    patch_size: int
    num_mask: int
    mask_stride: int = -1
    pa: int = 0
    pb: int = 0

args = Args(32, 6)
ds_config = {"input_size":[3, 224, 224]}

# t0 = time.time()
mask_list_OLD, mask_size_OLD, mask_stride_OLD = gen_mask_set_OLD(args,ds_config)
# t1 = time.time()
# print(f"old method: {t1 - t0}")

total_sum = np.sum([np.sum((mask_list_OLD[i][0, 0] ^ mask_list[i]).cpu().numpy().astype(int)) for i in range(mask_number[0] * mask_number[1])])
print(f"total sum of errors: {total_sum}")