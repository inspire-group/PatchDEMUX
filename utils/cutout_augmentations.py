import numpy as np
import torch

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