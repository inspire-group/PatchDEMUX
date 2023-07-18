import numpy as np
import torch
from copy import deepcopy

# Adopted from: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
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