# Adopted from: https://github.com/inspire-group/PatchCleanser/blob/main/utils/defense.py

import numpy as np 
import torch 
from dataclasses import dataclass

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

    # Initialize a base mask (i.e., mask is located all the way to the left)
    masks_list = [0] * mask_number
    base_mask = np.concatenate((mask_size * [0], (input_len - mask_size) * [1]))[..., np.newaxis].astype(bool)

    # Generate mask set by sliding the mask across the 1D array
    for k in np.arange(mask_number - 1): masks_list[k] = np.roll(base_mask, k * mask_stride)
      
    # Account for final mask (i.e., mask is located all the way to the right)
    masks_list[-1] = np.concatenate(((input_len - mask_size) * [1], mask_size * [0]))[..., np.newaxis].astype(bool)

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

    # Combine 1D masks with matrix OR operation to create 2D image masks
    mask_list = [torch.from_numpy(w_mask | h_mask.T).cuda() for w_mask in width_masks for h_mask in height_masks]

    return mask_list, mask_size, mask_stride

def single_masking(data, mask_list, num_classes, model, model_config):
    # run inference on a set of data which is augmented by masks from mask_list
    '''
    INPUT:
    data            torch.Tensor [B,C,W,H], a batch of data
    mask_list       a list of torch.Tensor, R-covering mask set
    num_classes     number of classes in the dataset
    model           torch.nn.module, the vanilla undefended model
    model_config    ModelConfig dataclass, contains configuration for model

    OUTPUT:
    mask_all_preds  numpy.ndarray, the prediction labels across all masks from mask_list
    '''
    num_classes = model_config.num_classes
    rank = model_config.rank
    thre = model_config.thre
    Sig = torch.nn.Sigmoid()

    num_masks = len(mask_list)
    mask_all_preds = np.zeros([data.shape[0], num_masks, num_classes])
    for i, mask in enumerate(mask_list):
        mask = mask.reshape(1, 1, *mask.shape).to(rank)            
        masked_im = torch.where(mask, data.to(rank), torch.tensor(0.0).to(rank))

        # compute output
        with torch.no_grad():
            output = Sig(model(masked_im).to(rank)).cpu()

        pred = output.data.gt(thre).long()
        mask_all_preds[:, i] = pred.cpu().numpy()

    return mask_all_preds

def double_masking(data, mask_list_fr, num_classes, model, model_config, debug=False):
    # perform double masking inference with the input image, the mask set, and the undefended model
    '''
    INPUT:
    data            torch.Tensor [B,C,W,H], a batch of data
    mask_list       a list of torch.Tensor, R-covering mask set
    num_classes     number of classes in the dataset
    model           torch.nn.module, the vanilla undefended model
    model_config    ModelConfig dataclass, contains configuration for model

    OUTPUT:
    output_pred     numpy.ndarray, the prediction labels
    '''

    # Initialize output_pred to -1 in order to facilitate filtering at the end
    num_classes = model_config.num_classes
    rank = model_config.rank
    output_pred = np.zeros((data.shape[0], num_classes)) - 1

    # First-round masking
    if debug:
        fr_all_preds = np.ones((5, 10, 80))
        fr_all_preds[4, :, 0] = fr_all_preds[3, :, 1] = fr_all_preds[2, :, 2] = fr_all_preds[4, 1:, 2] = fr_all_preds[4, 8:, 1] = fr_all_preds[3, 9:, 78] = fr_all_preds[3, 7:, 79] = 0
    else:
        fr_all_preds = single_masking(data, mask_list_fr, num_classes, model, model_config)

    # Find which classes had consensus in first round masked predictions (relevant for case I)
    consensus_bool = np.all(fr_all_preds == fr_all_preds[:, 0:1, :], axis=1)
    output_pred[consensus_bool] = fr_all_preds[:, 0, :][consensus_bool]
    if np.all(consensus_bool): 
        return torch.tensor(output_pred)    # If every element has consensus, then skip cases II and III

    # Find the majority class associated with the masked predictions
    def axis_unique(arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]
    majority_pred = np.apply_along_axis(axis_unique, 1, fr_all_preds)
    minority_pred = np.transpose((fr_all_preds != np.expand_dims(majority_pred, 1)).nonzero())

    # Apply second round masking to all minority predictions
    filtered_masks_idx = np.unique(minority_pred[:, 1])
    for idx in filtered_masks_idx:

        # Ensure that the minority_pred array is not empty before slicing
        if not minority_pred.size: break
        minority_masks = minority_pred[minority_pred[:, 1] == idx]

        # Ensure that the minority_pred values associated with the first round mask at idx are not empty before slicing
        if not minority_masks.size: continue
        minority_image_idx = np.unique(minority_masks[:, 0])
        minority_images = data[minority_image_idx]
    
        # Apply the appropriate first round mask then perform second round masking
        if debug:
            import copy
            sr_all_preds = copy.deepcopy(fr_all_preds)
            sr_all_preds[4, :, 1] = sr_all_preds[3, :, 78] = 0
            sr_all_preds = sr_all_preds[minority_image_idx]
        else:
            mask_fr = mask_list_fr[idx]
            mask_fr = mask_fr.reshape(1, 1, *mask_fr.shape).to(rank)
            masked_minority_images = torch.where(mask_fr, minority_images.to(rank), torch.tensor(0.0).to(rank))
            sr_all_preds = single_masking(masked_minority_images, mask_list_fr, num_classes, model, model_config)

        # Find which classes had consensus in second round masked predictions (relevant for case II)
        sr_consensus_bool = np.all(sr_all_preds == sr_all_preds[:, 0:1, :], axis=1)
        for masked_im in minority_masks:
            masked_im_idx = np.where(minority_image_idx == masked_im[0])[0][0]    # Get reverse map of image index to minority_pred image index
            masked_im_class = masked_im[2]
            if sr_consensus_bool[masked_im_idx, masked_im_class]:
                output_pred[masked_im[0], masked_im_class] = sr_all_preds[masked_im_idx, 0, masked_im_class]
                minority_pred = minority_pred[np.logical_or(minority_pred[:, 0] != masked_im[0], minority_pred[:, 2] != masked_im_class)]    # Unwiring procedure to avoid unecessary second round masking
    
    # Set remaining outputs to be the first round majority value (relevant for case III)
    output_pred[output_pred == -1] = majority_pred[output_pred == -1]

    return torch.tensor(output_pred)
############################################ TEST CASE STARTING #######################################################
# test case (test represents fr_all_preds):
# test = np.ones((5, 10, 80))
# data = np.ones((5, 3, 448, 448)) + np.arange(5).reshape(-1, 1, 1, 1)
# test[4, :, 0] = test[3, :, 1] = test[2, :, 2] = test[4, 1:, 2] = test[4, 8:, 1] = test[3, 9:, 78] = test[3, 7:, 79] = 0
# output_pred = np.zeros((5, 80)) - 1

# consensus_bool = np.all(test == test[:, 0:1, :], axis=1)
# output_pred[consensus_bool] = test[:, 0 , :][consensus_bool]

# # then check if elements in fr_all_preds align with majority or not
# def axis_unique(arr):
#     values, counts = np.unique(arr, return_counts=True)
#     return values[np.argmax(counts)]
# majority_pred = np.apply_along_axis(axis_unique, 1, test)
# minority = np.transpose((test != np.expand_dims(majority_pred, 1)).nonzero())

# # say we want mask 9 (inside of first mask loop). also need an unwiring mechanism
# minority_masks = minority[minority[:, 1] == 9]
# minority_image_idx = np.unique(minority_masks[:, 0])
# minority_images = data[minority_image_idx]
# minority_classes = np.unique(minority_masks[:, 2])

# import copy
# minority_consensus_bool = copy.deepcopy(consensus_bool)
# minority_consensus_bool[4, :] = True

@dataclass
class ModelConfig:
    num_classes: int
    rank: int
    thre: float

if __name__ == '__main__':    
    # Set up model config
    num_classes = 80
    rank = 0
    thre = 0.8
    model_config = ModelConfig(num_classes, rank, thre)

    # Set up model datastructure
    model = None
    mask_list_fr = None
    
    # Set DEBUG mode on
    DEBUG = True

    # Perform test with sample data
    data = np.ones((5, 3, 448, 448)) + np.arange(5).reshape(-1, 1, 1, 1)
    output_pred = double_masking(data, mask_list_fr, num_classes, model, model_config, debug=DEBUG)


    # array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.],
    #    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.],
    #    [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.],
    #    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #     -1., -1.],
    #    [ 0., -1., -1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
    #      1.,  1.]])