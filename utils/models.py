"""
Model loading utilities for multi-label classification models.

Provides separate classes for ResNet-based (TResNet) and Vision Transformer (ViT) models.
"""

import os
import sys
import torch
import numpy as np
import json
from collections import OrderedDict

# Add package paths for model imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(os.path.join(parent_dir, "packages/ASL/"))
from packages.ASL.src.models import create_model

sys.path.append(os.path.join(parent_dir, "packages/query2labels/lib"))
from packages.query2labels.lib.models.query2label import build_q2l

class ResNetModel:
    """ResNet-based multi-label classification model (TResNet)."""
    
    def __init__(self, args):
        self.args = args
        
    def load_model(self, model_path, logger=None):
        """
        Load ResNet model from checkpoint.
        
        Args:
            model_path (str): Path to model checkpoint
            logger (callable, optional): Logging function for status messages
            
        Returns:
            tuple: (model, args, classes_list)
        """
        if logger:
            logger("Setting up the model...resnet")
        
        # Setup arguments
        
        # Create model
        model = create_model(self.args).cuda()
        
        # Load checkpoint
        state = torch.load(model_path, map_location='cpu')
        state_dict = state['model']
        
        # Load weights into model
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        # Cleanup
        del state
        del state_dict
        torch.cuda.empty_cache()
        
        if logger:
            logger('Model loading complete')
        
        return model, self.args


class ViTModel:
    """Vision Transformer-based multi-label classification model."""
    
    def __init__(self, args):
        self.args = args
        
    def _load_config(self):
        """Load ViT parameters from config file (required for ViT models)."""
        if not self.args.config:
            raise ValueError("Config file is required for ViT models (Q2L-CvT_w24-384)")
        
        with open(self.args.config, 'r') as f:
            cfg_dict = json.load(f)
        for k, v in cfg_dict.items():
            setattr(self.args, k, v)
        
    def clean_state_dict(self, state_dict):
        """
        Clean the state dict by removing 'module.' prefix.
        
        Args:
            state_dict (dict): Raw state dictionary from checkpoint
            
        Returns:
            dict: Cleaned state dictionary
        """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:7] == 'module.':
                k = k[7:]  # remove `module.`
            new_state_dict[k] = v
        return new_state_dict
        
    def load_model(self, model_path, logger=None):
        """
        Load ViT model from checkpoint.
        
        Args:
            model_path (str): Path to model checkpoint
            logger (callable, optional): Logging function for status messages
            
        Returns:
            tuple: (model, args, classes_list)
        """
        if logger:
            logger("Setting up the model...ViT")
        
        # Load config file parameters for ViT models
        self._load_config()
        
        # Setup arguments
        
        # Create model
        model = build_q2l(self.args).cuda()
        
        # Load checkpoint
        state = torch.load(model_path, map_location='cpu')
        state_dict = self.clean_state_dict(state['state_dict'])
        
        # Load weights into model
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        # Cleanup
        del state
        del state_dict
        torch.cuda.empty_cache()
        
        if logger:
            logger('Model loading complete')
        
        return model, self.args