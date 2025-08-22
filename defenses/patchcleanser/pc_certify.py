# Adopted from: 
# - https://github.com/inspire-group/PatchCleanser

import numpy as np
import torch

# Helper function to compute certification metrics from predictions
def compute_certification_metrics(all_preds, target):
    # Find which classes had consensus in masked predictions
    all_preds_ones = np.all(all_preds, axis=1)
    all_preds_zeros = np.all(np.logical_not(all_preds), axis=1)
    
    # Compute certified TP, TN, FN, FP
    confirmed_tp = np.logical_and(all_preds_ones, target).astype(int)
    confirmed_tn = np.logical_and(all_preds_zeros, np.logical_not(target)).astype(int)
    worst_case_fn = np.logical_and(np.logical_not(all_preds_ones), target).astype(int)
    worst_case_fp = np.logical_and(np.logical_not(all_preds_zeros), np.logical_not(target)).astype(int)
    
    return confirmed_tp, confirmed_tn, worst_case_fn, worst_case_fp

# Helper function to generate vulnerability status arrays for all images in a batch; each
# array is inverted and has size (num_masks x num_classes)
def compute_batch_vulnerability_arrays(all_preds, worst_case_fn, worst_case_fp, num_masks):
    inv_vul_arrays_fn = {}
    inv_vul_arrays_fp = {}
    
    for im_idx in range(all_preds.shape[0]):
        fn_bound = np.sum(worst_case_fn[im_idx, :])
        fp_bound = np.sum(worst_case_fp[im_idx, :])

        if fn_bound:
            inv_vul_arrays_fn[im_idx] = generate_vul_status_array(
                all_preds[im_idx, :, :], worst_case_fn[im_idx, :], "FN", num_masks
            )
        else:
            inv_vul_arrays_fn[im_idx] = None
        
        if fp_bound:
            inv_vul_arrays_fp[im_idx] = generate_vul_status_array(
                all_preds[im_idx, :, :], worst_case_fp[im_idx, :], "FP", num_masks
            )
        else:
            inv_vul_arrays_fp[im_idx] = None
    
    return inv_vul_arrays_fn, inv_vul_arrays_fp

# Generate vulnerability status array for a given metric. The inverted array is returned with
# size (num_masks x num_classes).
def generate_vul_status_array(im_preds, metrics, metric_type, num_masks):
    discordant_pred = 0 if metric_type == "FN" else 1

    # Create a vulnerability status array for classes that have failed certification
    failed_cert_class_idx = np.where(metrics == 1)[0]  # Get the actual indices
    inverted_vul_status_array = np.zeros((num_masks, len(failed_cert_class_idx)))
    
    # Loop over all sets of double masks to find where certification fails
    for i in range (num_masks * num_masks):
        discordant_preds_bool = (im_preds[i, :][failed_cert_class_idx]) == discordant_pred

        for idx in range(len(failed_cert_class_idx)):
            # If for a given mask combination the relevant class has failed certification, we note this down
            if discordant_preds_bool[idx]:
                first_mask = i // num_masks
                second_mask = i % num_masks

                inverted_vul_status_array[first_mask, idx] = True
                inverted_vul_status_array[second_mask, idx] = True
            else:
                continue

    return inverted_vul_status_array

# PatchCleanser certification. For convenience, inputs a multi-label classifier and 
# accumulates results for each multi-label class before returning.
def pc_certify(model, mask_list, im, target, args, logger):
    num_masks = len(mask_list)
    num_classes = args.num_classes
    Sig = torch.nn.Sigmoid()

    # Allow double counting of mask pairs to facilitate location-aware analysis later 
    all_preds = np.zeros([im.shape[0], num_masks * num_masks, num_classes]) - 1
    for i, mask1 in enumerate(mask_list):
        mask1 = mask1.reshape(1, 1, *mask1.shape).cuda()            

        logger(f"Certification is {(i / num_masks) * 100:0.2f}% complete!")
        for j, mask2 in enumerate(mask_list):
            mask2 = mask2.reshape(1, 1, *mask2.shape).cuda()
            masked_im = torch.where(torch.logical_and(mask1, mask2), im.cuda(), torch.tensor(0.0).cuda())

            # compute output
            with torch.no_grad():
                output = Sig(model(masked_im)).cpu()

            pred = output.data.gt(args.thre).long()
            all_preds[:, i * num_masks + j] = pred.cpu().numpy()

    # Compute certification metrics using helper function
    confirmed_tp, confirmed_tn, worst_case_fn, worst_case_fp = compute_certification_metrics(all_preds, target)

    # Generate vulnerability status arrays using helper function
    inv_vul_arrays_fn, inv_vul_arrays_fp = compute_batch_vulnerability_arrays(
        all_preds, worst_case_fn, worst_case_fp, num_masks
    )

    return confirmed_tp, confirmed_tn, worst_case_fn, worst_case_fp, inv_vul_arrays_fn, inv_vul_arrays_fp

# PatchCleanser certification using cached outputs.
def pc_certify_cached(masked_output, mask_list, target, args):
    num_masks = len(mask_list)
    all_preds = (masked_output > args.thre).astype(int)

    # Compute certification metrics using helper function
    confirmed_tp, confirmed_tn, worst_case_fn, worst_case_fp = compute_certification_metrics(all_preds, target)

    # Generate vulnerability status arrays using helper function
    inv_vul_arrays_fn, inv_vul_arrays_fp = compute_batch_vulnerability_arrays(
        all_preds, worst_case_fn, worst_case_fp, num_masks
    )
    
    return confirmed_tp, confirmed_tn, worst_case_fn, worst_case_fp, inv_vul_arrays_fn, inv_vul_arrays_fp