import numpy as np 
import torch 

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
    masks_list = [0] * mask_number
    for index, start in enumerate(np.arange(mask_number - 1) * mask_stride):
        mask = np.zeros((input_len, 1))
        mask[start : start + mask_size] = 1

        masks_list[index] = mask
    
    # Account for final mask
    masks_list[-1] = np.concatenate((np.zeros((input_len - mask_size, 1)), np.ones((mask_size, 1))))

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
    mask_list = [torch.from_numpy((w_mask @ h_mask.T).astype(bool)).cuda() for w_mask in width_masks for h_mask in height_masks]

    return mask_list, mask_size, mask_stride
