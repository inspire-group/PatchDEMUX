import torch
from copy import deepcopy

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