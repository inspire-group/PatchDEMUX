# Adopted from: 
# - https://github.com/inspire-group/PatchCleanser

import numpy as np 
import torch
from utils.common import ModelConfig

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
    thre = model_config.thre
    Sig = torch.nn.Sigmoid()

    num_masks = len(mask_list)
    mask_all_preds = np.zeros([data.shape[0], num_masks, num_classes])
    for i, mask in enumerate(mask_list):
        mask = mask.reshape(1, 1, *mask.shape).cuda()            
        masked_im = torch.where(mask, data.cuda(), torch.tensor(0.0).cuda())

        # compute output
        with torch.no_grad():
            output = Sig(model(masked_im).cuda()).cpu()

        pred = output.data.gt(thre).long()
        mask_all_preds[:, i] = pred.cpu().numpy()

    return mask_all_preds

# perform double masking inference with the input image, the mask set, and the undefended model
def pc_infer_doublemasking(data, mask_list, num_classes, model, model_config):
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
    output_pred = np.zeros((data.shape[0], num_classes)) - 1

    # First-round masking
    fr_all_preds = single_masking(data, mask_list, num_classes, model, model_config)

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

    # numpy.nonzero returns a list of indices for each axis corresponding to elements with nonzero values - in this context the three 
    # axes are [batch_index, mask_id, class_id], so minority_pred_entries.shape = (num_nonzero_entries, 3)
    minority_pred_entries = np.transpose((fr_all_preds != np.expand_dims(majority_pred, 1)).nonzero())

    # Apply second round masking to all minority predictions - we loop over first round masks that have minority predictions
    # because the same mask might have been a minority for multiple classes. This allows us to re-use intermediate outputs and
    # speed up inference. 
    filtered_masks_idx = np.unique(minority_pred_entries[:, 1])
    for idx in filtered_masks_idx:

        # Ensure that the minority_pred_entries array is not empty before slicing
        if not minority_pred_entries.size: break
        minority_pred_subset = minority_pred_entries[minority_pred_entries[:, 1] == idx]

        # Ensure that the minority_pred_entries values associated with the first round mask at idx are not empty before slicing
        if not minority_pred_subset.size: continue
        minority_image_idx = np.unique(minority_pred_subset[:, 0])
        minority_images = data[minority_image_idx]
    
        # Apply the appropriate first round mask then perform second round masking
        mask_fr = mask_list[idx]
        mask_fr = mask_fr.reshape(1, 1, *mask_fr.shape).cuda()
        masked_minority_images = torch.where(mask_fr, minority_images.cuda(), torch.tensor(0.0).cuda())
        sr_all_preds = single_masking(masked_minority_images, mask_list, num_classes, model, model_config)

        # Find which classes had consensus in second round masked predictions (relevant for case II)
        sr_consensus_bool = np.all(sr_all_preds == sr_all_preds[:, 0:1, :], axis=1)
        for minority_pred_entry in minority_pred_subset:
            masked_im_idx = np.where(minority_image_idx == minority_pred_entry[0])[0][0]    # Get reverse map of image index to the location within the minority_image_idx array
            masked_im_class = minority_pred_entry[2]
            if sr_consensus_bool[masked_im_idx, masked_im_class]:
                output_pred[minority_pred_entry[0], masked_im_class] = sr_all_preds[masked_im_idx, 0, masked_im_class]
                minority_pred_entries = minority_pred_entries[np.logical_or(minority_pred_entries[:, 0] != minority_pred_entry[0], minority_pred_entries[:, 2] != masked_im_class)]    # Unwiring procedure to avoid unecessary second round masking
    
    # Set remaining outputs to be the first round majority value (relevant for case III)
    output_pred[output_pred == -1] = majority_pred[output_pred == -1]

    return torch.tensor(output_pred)

# perform double masking inference with the input image, the mask set, and the undefended model; only returns predictions for the first class
def pc_infer_firstclass(data, mask_list, num_classes, model, model_config):
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

    FIRST_CLASS = 0

    # Initialize output_pred to -1 in order to facilitate filtering at the end
    num_classes = model_config.num_classes
    output_pred = np.zeros((data.shape[0], 1)) - 1

    # First-round masking
    fr_all_preds = single_masking(data, mask_list, num_classes, model, model_config)
    first_class_preds = fr_all_preds[:, :, FIRST_CLASS]

    # Find whether there was consensus in first round masked predictions (relevant for case I)
    first_class_consensus_bool = np.all(first_class_preds == first_class_preds[:, 0:1], axis=1)[:, np.newaxis]

    output_pred[first_class_consensus_bool] = first_class_preds[:, 0:1][first_class_consensus_bool]
    if np.all(first_class_consensus_bool): 
        return torch.tensor(output_pred)    # If every element has consensus, then skip cases II and III

    # Find the majority class associated with the masked predictions
    def axis_unique(arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]
    majority_pred = np.apply_along_axis(axis_unique, 1, first_class_preds)

    # numpy.nonzero returns a list of indices for each axis corresponding to elements with nonzero values - in this context the two 
    # axes are [batch_index, mask_id], so minority_pred.shape = (num_nonzero_entries, 2)
    minority_pred_entries = np.transpose((first_class_preds != np.expand_dims(majority_pred, 1)).nonzero())

    # Apply second round masking to all minority predictions
    filtered_masks_idx = np.unique(minority_pred_entries[:, 1])
    for idx in filtered_masks_idx:

        # Ensure that the minority_pred_entries array is not empty before slicing
        if not minority_pred_entries.size: break
        minority_pred_subset = minority_pred_entries[minority_pred_entries[:, 1] == idx]

        # Ensure that the minority_pred_entries values associated with the first round mask at idx are not empty before slicing
        if not minority_pred_subset.size: continue
        minority_image_idx = np.unique(minority_pred_subset[:, 0])
        minority_images = data[minority_image_idx]
    
        # Apply the appropriate first round mask then perform second round masking
        mask_fr = mask_list[idx]
        mask_fr = mask_fr.reshape(1, 1, *mask_fr.shape).cuda()
        masked_minority_images = torch.where(mask_fr, minority_images.cuda(), torch.tensor(0.0).cuda())
        sr_all_preds = single_masking(masked_minority_images, mask_list, num_classes, model, model_config)
        first_class_sr_preds = sr_all_preds[:, :, FIRST_CLASS]

        # Find which classes had consensus in second round masked predictions (relevant for case II)
        first_class_sr_consensus_bool = np.all(first_class_sr_preds == first_class_sr_preds[:, 0:1], axis=1)[:, np.newaxis]
        for minority_pred_entry in minority_pred_subset:
            masked_im_idx = np.where(minority_image_idx == minority_pred_entry[0])[0][0]    # Get reverse map of image index to the location within the minority_image_idx array
            if first_class_sr_consensus_bool[masked_im_idx]:
                output_pred[minority_pred_entry[0]] = first_class_sr_preds[masked_im_idx, 0]
                minority_pred_entries = minority_pred_entries[minority_pred_entries[:, 0] != minority_pred_entry[0]]    # Unwiring procedure to avoid unecessary second round masking
    
    # Set remaining outputs to be the first round majority value (relevant for case III)
    majority_pred = majority_pred[:, np.newaxis]
    output_pred[output_pred == -1] = majority_pred[output_pred == -1]

    return torch.tensor(output_pred)

def single_masking_cache(cached_outputs, fixed_mask_idx, mask_list, num_classes, model_config):
    # run inference on a set of data which is augmented by masks from mask_list
    '''
    INPUT:
    cached_outputs  numpy.ndarray [B,len(mask_list) ** 2,len(num_classes)], a batch of cached outputs
    fixed_mask_idx  int, index for a primary mask which is present in addition to the mask_list (-1 implies absence)
    mask_list       a list of torch.Tensor, R-covering mask set
    num_classes     number of classes in the dataset
    model_config    ModelConfig dataclass, contains configuration for model

    OUTPUT:
    mask_all_preds  numpy.ndarray, the prediction labels across all masks from mask_list
    '''
    num_classes = model_config.num_classes

    num_masks = len(mask_list)
    primary_mask_absence = (fixed_mask_idx < 0)
    mask_all_preds = np.zeros([cached_outputs.shape[0], num_masks, num_classes])
    for i, mask in enumerate(mask_list):
        if primary_mask_absence: fixed_mask_idx = i

        # For cached_outputs, set of single mask predictions corresponding to mask i are at index (i * num_masks + i)
        mask_all_preds[:, i] = cached_outputs[:, (i * num_masks + fixed_mask_idx)]

    return mask_all_preds

# perform double masking inference with the cached masked outputs and the mask set
def pc_infer_doublemasking_cached(cached_outputs, mask_list, num_classes, model_config):
    '''
    INPUT:
    cached_outputs  numpy.ndarray [B,len(mask_list) ** 2,len(num_classes)], a batch of cached outputs
    mask_list       a list of torch.Tensor, R-covering mask set
    num_classes     number of classes in the dataset
    model_config    ModelConfig dataclass, contains configuration for model

    OUTPUT:
    output_pred     numpy.ndarray, the prediction labels
    '''

    cached_outputs = (cached_outputs > model_config.thre).astype(int)

    # Initialize output_pred to -1 in order to facilitate filtering at the end
    num_classes = model_config.num_classes
    output_pred = np.zeros((cached_outputs.shape[0], num_classes)) - 1

    # First-round masking
    fr_all_preds = single_masking_cache(cached_outputs, -1, mask_list, num_classes, model_config)

    # Find which classes had consensus in first round masked predictions (relevant for case I)
    consensus_bool = np.all(fr_all_preds == np.expand_dims(fr_all_preds[:, 0, :], 1), axis=1)
    output_pred[consensus_bool] = fr_all_preds[:, 0, :][consensus_bool]
    if np.all(consensus_bool): 
        return torch.tensor(output_pred)    # If every element has consensus, then skip cases II and III

    # Find the majority class associated with the masked predictions
    def axis_unique(arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]
    majority_pred = np.apply_along_axis(axis_unique, 1, fr_all_preds)

    # numpy.nonzero returns a list of indices for each axis corresponding to elements with nonzero values - in this context the three 
    # axes are [batch_index, mask_id, class_id], so minority_pred.shape = (num_nonzero_entries, 3)
    minority_pred_entries = np.transpose((fr_all_preds != np.expand_dims(majority_pred, 1)).nonzero())

    # Apply second round masking to all minority predictions
    filtered_masks_idx = np.unique(minority_pred_entries[:, 1])
    for idx in filtered_masks_idx:

        # Ensure that the minority_pred_entries array is not empty before slicing
        if not minority_pred_entries.size: break
        minority_pred_subset = minority_pred_entries[minority_pred_entries[:, 1] == idx]

        # Ensure that the minority_pred_entries values associated with the first round mask at idx are not empty before slicing
        if not minority_pred_subset.size: continue
        minority_image_idx = np.unique(minority_pred_subset[:, 0])
        minority_images_cache = cached_outputs[minority_image_idx]
    
        # Apply the appropriate first round mask then perform second round masking
        sr_all_preds = single_masking_cache(minority_images_cache, idx, mask_list, num_classes, model_config)

        # Find which classes had consensus in second round masked predictions (relevant for case II)
        sr_consensus_bool = np.all(sr_all_preds == np.expand_dims(sr_all_preds[:, 0, :], 1), axis=1)
        for minority_pred_entry in minority_pred_subset:
            masked_im_idx = np.where(minority_image_idx == minority_pred_entry[0])[0][0]    # Get reverse map of image index to the location within the minority_image_idx array
            masked_im_class = minority_pred_entry[2]
            if sr_consensus_bool[masked_im_idx, masked_im_class]:
                output_pred[minority_pred_entry[0], masked_im_class] = sr_all_preds[masked_im_idx, 0, masked_im_class]
                minority_pred_entries = minority_pred_entries[np.logical_or(minority_pred_entries[:, 0] != minority_pred_entry[0], minority_pred_entries[:, 2] != masked_im_class)]    # Unwiring procedure to avoid unecessary second round masking
    
    # Set remaining outputs to be the first round majority value (relevant for case III)
    output_pred[output_pred == -1] = majority_pred[output_pred == -1]

    return torch.tensor(output_pred)