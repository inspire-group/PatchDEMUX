# Adopted from: 
# - https://github.com/inspire-group/PatchCleanser

import numpy as np 
import torch

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

############## FOR INFERENCE TIMING PURPOSES ##################
def double_masking_indiv_class(data, mask_list_fr, num_classes, model, model_config):
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
    output_pred = np.zeros((data.shape[0], 1)) - 1

    # First-round masking
    fr_all_preds = single_masking(data, mask_list_fr, num_classes, model, model_config)
    first_class_preds = fr_all_preds[:, :, 0]

    # Find which classes had consensus in first round masked predictions (relevant for case I)
    first_class_consensus_bool = np.all(first_class_preds == first_class_preds[:, 0:1], axis=1)[:, np.newaxis]

    output_pred[first_class_consensus_bool] = first_class_preds[:, 0:1][first_class_consensus_bool]
    if np.all(first_class_consensus_bool): 
        return torch.tensor(output_pred)    # If every element has consensus, then skip cases II and III

    # Find the majority class associated with the masked predictions
    def axis_unique(arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]

    majority_pred = np.apply_along_axis(axis_unique, 1, first_class_preds)
    minority_pred_idx = np.transpose((first_class_preds != np.expand_dims(majority_pred, 1)).nonzero())

    # Apply second round masking to all minority predictions (this is the set of maniority predictions across all classes and all images)
    minority_pred_masks = np.unique(minority_pred_idx[:, 1])

    # Iterating a loop based on the mask idx makes more sense here, as it more closely imitates the actual inference process (only difference is we are dealing with a batch of images at a time)
    for idx in minority_pred_masks:

        # Ensure that the minority_pred array is not empty before slicing
        if not minority_pred_idx.size: break
        filtered_minority_pred_idx = minority_pred_idx[minority_pred_idx[:, 1] == idx]

        # Ensure that the minority_pred values associated with the first round mask at idx are not empty before slicing
        if not filtered_minority_pred_idx.size: continue
        minority_image_idx = np.unique(filtered_minority_pred_idx[:, 0])
        minority_image_batch = data[minority_image_idx]
    
        # Apply the appropriate first round mask then perform second round masking
        mask_fr = mask_list_fr[idx]
        mask_fr = mask_fr.reshape(1, 1, *mask_fr.shape).to(rank)
        masked_minority_images = torch.where(mask_fr, minority_image_batch.to(rank), torch.tensor(0.0).to(rank))
        sr_all_preds = single_masking(masked_minority_images, mask_list_fr, num_classes, model, model_config)
        first_class_sr_preds = sr_all_preds[:, :, 0]

        # Find which classes had consensus in second round masked predictions (relevant for case II)
        first_class_sr_consensus_bool = np.all(first_class_sr_preds == first_class_sr_preds[:, 0:1], axis=1)[:, np.newaxis]
    
        for filtered_im_idx in filtered_minority_pred_idx:
            masked_im_idx = np.where(minority_image_idx == filtered_im_idx[0])[0][0]    # Get reverse map of image index to minority_pred image index
            if first_class_sr_consensus_bool[masked_im_idx]:
                output_pred[filtered_im_idx[0]] = first_class_sr_preds[masked_im_idx, 0]
                minority_pred_idx = minority_pred_idx[minority_pred_idx[:, 0] != filtered_im_idx[0]]    # Unwiring procedure to avoid unecessary second round masking
    
    # Set remaining outputs to be the first round majority value (relevant for case III)
    majority_pred = majority_pred[:, np.newaxis]
    output_pred[output_pred == -1] = majority_pred[output_pred == -1]

    return torch.tensor(output_pred)

#######################################################
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
    rank = model_config.rank
    thre = model_config.thre

    num_masks = len(mask_list)
    primary_mask_absence = (fixed_mask_idx < 0)
    mask_all_preds = np.zeros([cached_outputs.shape[0], num_masks, num_classes])
    for i, mask in enumerate(mask_list):
        if primary_mask_absence: fixed_mask_idx = i

        # For cached_outputs, set of single mask predictions corresponding to mask i are at index (i * num_masks + i)
        mask_all_preds[:, i] = cached_outputs[:, (i * num_masks + fixed_mask_idx)]

    return mask_all_preds

def double_masking_cache(cached_outputs, mask_list_fr, num_classes, model_config):
    # perform double masking inference with the cached masked outputs and the mask set
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
    rank = model_config.rank
    output_pred = np.zeros((cached_outputs.shape[0], num_classes)) - 1

    # First-round masking
    fr_all_preds = single_masking_cache(cached_outputs, -1, mask_list_fr, num_classes, model_config)

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
        minority_images_cache = cached_outputs[minority_image_idx]
    
        # Apply the appropriate first round mask then perform second round masking
        sr_all_preds = single_masking_cache(minority_images_cache, idx, mask_list_fr, num_classes, model_config)

        # Find which classes had consensus in second round masked predictions (relevant for case II)
        sr_consensus_bool = np.all(sr_all_preds == np.expand_dims(sr_all_preds[:, 0, :], 1), axis=1)
        for masked_im in minority_masks:
            masked_im_idx = np.where(minority_image_idx == masked_im[0])[0][0]    # Get reverse map of image index to minority_pred image index; leverage the fact that minority_image_idx removes duplicate batch image indices
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