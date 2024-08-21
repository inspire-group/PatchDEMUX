import numpy as np
import torch
from copy import deepcopy

# Adopted from: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length  (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w)).astype(int)

        for n in range(self.n_holes):
            # Initialize the center of the cutout at a random location
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - np.floor(self.length / 2).astype(int), 0, h)
            y2 = np.clip(y + np.ceil(self.length / 2).astype(int), 0, h)
            x1 = np.clip(x - np.floor(self.length / 2).astype(int), 0, w)
            x2 = np.clip(x + np.ceil(self.length / 2).astype(int), 0, w)

            mask[y1: y2, x1: x2] = 0

        mask = torch.from_numpy(mask.astype(bool))
        img_masked = torch.where(mask, img, torch.tensor(0.0))

        return img_masked

class GreedyCutout(object):
    """Applies a pair of masks which greedily induce high loss

    Args:
        mask_list          (list): Set of candidate masks
        greedy_cutout_data (dict): Specifies which mask pair is the greedy optimum for each image
    """
    def __init__(self, mask_list, greedy_cutout_data):
        self.mask_list = mask_list
        self.greedy_cutout_data = greedy_cutout_data

    # - apply transformerwrapper to the non greedy cutout transforms in cutout_trian.py
    # - in datasets, ensure that we are now pasing tuple (im, file_name) into the overall transform
    # - here, just read from the data and apply the two masks to the input image before returning (check manually that the loss is still correct, and save a sample image to disk!)
    def __call__(self, input_data):
        """
        Args:
            data (tuple): consists of an image and its filename
        Returns:
            Tensor: Image augmented with the mask pair specified in greedy_cutout_data
        """
        im, file_name = input_data

        # Get the mask indices
        fr_mask_idx = int(self.greedy_cutout_data[file_name]["fr_mask"])
        sr_mask_idx = int(self.greedy_cutout_data[file_name]["sr_mask"])

        # Obtain the masks
        fr_mask = self.mask_list[fr_mask_idx]
        sr_mask = self.mask_list[sr_mask_idx]

        # Construct the masked image
        combined_mask = torch.logical_and(fr_mask, sr_mask)
        combined_mask = combined_mask.reshape(1, *combined_mask.shape)         
        img_masked = torch.where(combined_mask, im, torch.tensor(0.0))

        return img_masked, file_name

# Sourced from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py
class ModelEma(torch.nn.Module):
    """ Model Exponential Moving Average

    Keep a moving average of everything in the model state_dict (parameters and buffers).

    Args:
        model (torch.nn.Module): Base model for EMA.
        decay (float): Proportion of previous model to keep.
    """
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))  # use torch.Tensor.copy_() method to modify in-place

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)