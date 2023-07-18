# Adopted from: https://github.com/inspire-group/PatchCleanser/blob/main/utils/defense.py

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

def double_masking(data,mask_list_fr,num_classes,model):
    # perform double masking inference with the input image, the mask set, and the undefended model
    '''
    INPUT:
    data            torch.Tensor [B,C,W,H], a batch of data
    mask_list       a list of torch.Tensor, R-covering mask set
    num_classes     number of classes in the dataset
    model           torch.nn.module, the vanilla undefended model

    OUTPUT:
    output_pred     numpy.ndarray, the prediction labels
    '''

    # Initialize output_pred to -1 in order to facilitate filtering at the end
    output_pred = np.zeros(data.shape[0], num_classes) - 1
    rank = 0    # Assume GPU id is 0

    # first-round masking
    num_masks_fr = len(mask_list)

    # Initialize all_preds
    fr_all_preds = np.zeros([data.shape[0], num_masks_fr, num_classes])
    for i, mask in enumerate(mask_list_fr):
        mask = mask.reshape(1, 1, *mask.shape).to(rank)            
        masked_im = torch.where(mask, data.to(rank), torch.tensor(0.0).to(rank))

        # compute output
        with torch.no_grad():
            output = Sig(model(masked_im).to(rank)).cpu()

        pred = output.data.gt(args.thre).long()
        fr_all_preds[:, i] = pred.cpu().numpy()

    # Find which classes had consensus in masked predictions (relevant for case I)
    consensus_bool = np.all(fr_all_preds == fr_all_preds[:, 0:1, :], axis=1)
    output_pred[consensus_bool] = fr_all_preds[:, 0, :][consensus_bool]

    # Find the majority class associated with the masked predictions (relevant for case III)
    def axis_unique(arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]
    majority_pred = np.apply_along_axis(axis_unique, 1, fr_all_preds)

    # Apply second round masking (relevant for case II)
    num_masks_fr = len(mask_list)

    for i, mask1 in enumerate(mask_list_fr):
        mask = mask.reshape(1, 1, *mask.shape).to(rank)

        for j, mask2 in enumerate(mask_list_fr):
            mask2 = mask2.reshape(1, 1, *mask2.shape).to(rank)
            masked_im = torch.where(torch.logical_and(mask1, mask2), im.to(args.rank), torch.tensor(0.0).to(args.rank))            

            # compute output
            with torch.no_grad():
                output = Sig(model(masked_im).to(rank)).cpu()

        pred = output.data.gt(args.thre).long()
        fr_all_preds[:, i] = pred.cpu().numpy()

    # filtered_masks_idx = np.unique(minority[:, 1])
    # for i in filtered_masks_idx:
    #     mask1 = mask_list_fr[i].reshape(1, 1, *mask.shape).to(rank)
    #
    #     if not minority.size: break
    #     minority_masks = minority[minority[:, 1] == i]
    #     if not minority_masks.size: continue
    #     minority_image_idx = np.unique(minority_masks[:, 0])
    #     minority_images = data[minority_image_idx]
    #
    #     RUN MODEL ON DOUBLE MASKED BATCH TO GET sr_all_preds (sr_all_preds is size num_filtered_im, num_second_rd_masks, num_classses)
    #
    #     minority_consensus_bool = np.all(sr_all_preds == sr_all_preds[:, 0:1, :], axis=1)
    #     for masked_im in minority_masks:
    #         masked_im_idx = np.where(minority_image_idx == masked_im[0])[0][0]    # Get reverse map of image index to minority image index
    #         masked_im_class = masked_im[2]
    #         if minority_consensus_bool[masked_im_idx, masked_im_class]
    #             output_pred[masked_im[0], masked_im_class] = sr_all_preds[masked_im_idx, 0, masked_im_class]
    #             minority = minority[np.logical_or(minority[:, 0] != masked_im[0], minority[:, 2] != masked_im_class)]
    #         
############################################ TEST CASE STARTING #######################################################
# test case (test represents fr_all_preds):
test = np.ones((5, 10, 80))
data = np.ones((5, 3, 448, 448)) + np.arange(5).reshape(-1, 1, 1, 1)
test[4, :, 0] = test[3, :, 1] = test[2, :, 2] = test[4, 1:, 2] = test[4, 8:, 1] = test[3, 9:, 78] = test[3, 7:, 79] = 0
output_pred = np.zeros((5, 80)) - 1

consensus_bool = np.all(test == test[:, 0:1, :], axis=1)
output_pred[consensus_bool] = test[:, 0 , :][consensus_bool]

# then check if elements in fr_all_preds align with majority or not
def axis_unique(arr):
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]
majority_pred = np.apply_along_axis(axis_unique, 1, test)
minority = np.transpose((test != np.expand_dims(majority_pred, 1)).nonzero())

# say we want mask 9 (inside of first mask loop). also need an unwiring mechanism
minority_masks = minority[minority[:, 1] == 9]
minority_image_idx = np.unique(minority_masks[:, 0])
minority_images = data[minority_image_idx]
minority_classes = np.unique(minority_masks[:, 2])

import copy
minority_consensus_bool = copy.deepcopy(consensus_bool)
minority_consensus_bool[4, :] = True

############################################ EXPLANATION OF THE CODE #######################################################
    # first check which of the images have consensus at the batch level; these are already set then (i.e., via case I)

    # then check the majority value of the remaining images at the batch level; if second round masking does not work out, these will be the values (i.e., via case III)

    # Hardest part: implementing second round masking (i.e., via case II)
    # take all the values that are -1 (these were not certified) and note the mask id (class id not needed as multiple classes in same mask may 
    # need to be cerified) also note the image id (this way you can take boolean slice of the batch and only run these into the maskings)
    # - should have a mapping for mask index to an array with all the image indices which had an uncertified class; then use this in the first for loop going over first round mask
    # - enumerate over all masks in the second round. then at the end need to check each class id individually when certifying; if there is consensus, then note this down so that 
    #   future second round masking does not have to include (image_id, class_id) pair


    # ACTUALLY INSTEAD: at first when you are taking values that are -1 find out image id, mask id, AND class id. Do the loops in the same way (i.e., the 
    # first loop goes over first round masks, which then fetches relevant image_id from a dictionary for teh batch and only looks at class_id for what to check for consensus)

    # dictionart takes mask_id as key and returns an array with first coumn as image_id and second as class _id (obv. can be repetas of image_id)
    # Then take np.unique of the image_ids to fill out the batch, and np.unique of the class_ids to slice the outputs. finally loop over the list of 
    # (image_id, class_id) pairs and check if consensus was achieved. if so, check a boolean array (dont need separate boolean array; )


####################################################################################################
    def double_masking(data,mask_list,model):
    # perform double masking inference with the input image, the mask set, and the undefended model
    '''
    INPUT:
    data            torch.Tensor [B,C,W,H], a batch of data
    mask_list       a list of torch.Tensor, R-covering mask set
    model           torch.nn.module, the vanilla undefended model

    OUTPUT:
    output_pred     numpy.ndarray, the prediction labels
    '''

    # first-round masking 
    num_img = len(data)
    num_mask = len(mask_list)
    pred_one_mask_batch = np.zeros([num_img,num_mask],dtype=int)
    # compute one-mask prediction in batch
    for i,mask in enumerate(mask_list):
        masked_output = model(torch.where(mask,data,torch.tensor(0.).cuda()))
        _, masked_pred = masked_output.max(1)
        masked_pred = masked_pred.detach().cpu().numpy()
        pred_one_mask_batch[:,i] = masked_pred

    # determine the prediction label for each image
    output_pred = np.zeros([num_img],dtype=int)
    for j in range(num_img):
        pred_one_mask = pred_one_mask_batch[j]
        pred,cnt = np.unique(pred_one_mask,return_counts=True)

        if len(pred)==1: # unanimous agreement in the first-round masking
            defense_pred = pred[0] # Case I: agreed prediction
        else:
            sorted_idx = np.argsort(cnt)
            # get majority prediction and disagreer prediction
            majority_pred = pred[sorted_idx][-1]
            disagreer_pred = pred[sorted_idx][:-1]

            # second-round masking
            # get index list of the disagreer mask            
            tmp = np.zeros_like(pred_one_mask,dtype=bool)
            for dis in disagreer_pred:
                tmp = np.logical_or(tmp,pred_one_mask==dis)
            disagreer_pred_mask_idx = np.where(tmp)[0]

            for i in disagreer_pred_mask_idx:
                dis = pred_one_mask[i]
                mask = mask_list[i]
                flg=True
                for mask2 in mask_list:
                    # evaluate two-mask predictions
                    masked_output = model(torch.where(torch.logical_and(mask,mask2),data[j],torch.tensor(0.).cuda()))
                    masked_conf, masked_pred = masked_output.max(1)
                    masked_pred = masked_pred.item()
                    if masked_pred!=dis: # disagreement in the second-round masking -> discard the disagreer
                        flg=False
                        break
                if flg:
                    defense_pred = dis # Case II: disagreer prediction
                    break
            if not flg:
                defense_pred = majority_pred # Case III: majority prediction
        output_pred[j] = defense_pred
    return output_pred