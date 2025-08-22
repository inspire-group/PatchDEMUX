# Adopted from: 
# - https://github.com/inspire-group/PatchCleanser

import numpy as np 
import torch 

def gen_1D_mask_set(input_len, patch_len, mask_number):
    # generate a R-covering mask set in 1D
    '''
    INPUT:
    input_len       length of input
    patch_len       estimated patch length
    mask_number     computation budget

    OUTPUT:
    mask_list       the generated R-covering mask set
    mask_size       the mask size
    masks_stride    the mask stride
    '''

    # Determine mask stride and mask size
    mask_stride = int(np.ceil((input_len - patch_len + 1) / mask_number))
    mask_size = patch_len + mask_stride - 1    

    # Generate valid mask positions
    idx_list = list(range(0, input_len - mask_size + 1, mask_stride))
    if (input_len - mask_size) % mask_stride != 0:
        idx_list.append(input_len - mask_size)
    
    # Create masks at valid positions only
    masks_list = []
    for pos in idx_list:
        mask = np.ones(input_len, dtype=bool)
        mask[pos:pos + mask_size] = False
        masks_list.append(mask[..., np.newaxis])

    return masks_list, mask_size, mask_stride

def gen_mask_set(im_size, patch_size, mask_number):
    # generate a R-covering mask set for a 2D image
    '''
    INPUT:
    im_size         size of the input image along both axes, (width, height)
    patch_size      estimated patch size along both axes, (width, height)
    mask_number     computation budget along both axes, (width, height)

    OUTPUT:
    mask_list       list of torch.tensor, the generated R-covering mask set
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

    # Combine 1D masks with matrix OR operation to create 2D image masks
    mask_list = [torch.from_numpy(w_mask | h_mask.T) for w_mask in width_masks for h_mask in height_masks]

    return mask_list, mask_size, mask_stride

def cache_masked_outputs(model, mask_list, im, args, logger):
    """
    Generate the entire set of masked predictions for caching purposes.
    
    This function applies all mask pairs to the input image and generates 
    predictions for caching, similar to pc_certify() but focused on output
    generation rather than certification.
    
    Args:
        model: The multi-label classifier
        mask_list: List of masks to apply
        im: Input image tensor
        args: Arguments containing number of classes
        logger: Logging function for progress updates
    
    Returns:
        masked_output: Array of raw outputs for all mask pairs [batch_size, num_masks^2, num_classes]
    """
    num_masks = len(mask_list)
    num_classes = args.num_classes
    Sig = torch.nn.Sigmoid()
    
    # Allow double counting of mask pairs to facilitate location-aware analysis later 
    masked_output = np.zeros([im.shape[0], num_masks * num_masks, num_classes]) - 1
    for i, mask1 in enumerate(mask_list):
        mask1 = mask1.reshape(1, 1, *mask1.shape).cuda()            

        logger(f"Caching is {(i / num_masks) * 100:0.2f}% complete!")
        for j, mask2 in enumerate(mask_list):
            mask2 = mask2.reshape(1, 1, *mask2.shape).cuda()
            masked_im = torch.where(torch.logical_and(mask1, mask2), im.cuda(), torch.tensor(0.0).cuda())

            # compute output
            with torch.no_grad():
                output = Sig(model(masked_im).cuda()).cpu()

            masked_output[:, i * num_masks + j] = output.cpu().numpy()
    
    return masked_output