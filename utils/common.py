# Common utility functions shared across multiple scripts

import os
import torchvision.transforms as transforms
from dataclasses import dataclass
from .datasets import CocoDetection, VOCDetection, TransformWrapper
from .models import ResNetModel, ViTModel

@dataclass
class ModelConfig:
    num_classes: int
    thre: float

def file_print(file_path, msg):
    """Print a message to both stdout and a log file"""
    with open(file_path, "a") as f:
        print(msg, flush=True, file=f)

def load_model(args, is_ViT):
    """Load in the multi-label classifier model"""
    args.do_bottleneck_head = False

    # Create appropriate model instance
    if is_ViT:
        model_loader = ViTModel(args)
    else:
        model_loader = ResNetModel(args)
    
    # Load the model using the model class
    model, args = model_loader.load_model(args.model_path, lambda msg: file_print(args.logging_file, msg))
    
    return model, args

def load_eval_dataset(args):
    """Load evaluation dataset based on dataset name"""
    if args.dataset_name == "mscoco":
        instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
        data_path = os.path.join(args.data, 'images/val2014')
        eval_dataset = CocoDetection(data_path,
                                    instances_path,
                                    transforms.Compose([
                                        TransformWrapper(transforms.Resize((args.image_size, args.image_size))),
                                        TransformWrapper(transforms.ToTensor()),
                                    ]))
    elif args.dataset_name == "pascalvoc":
        data_path_val = os.path.join(args.data, 'test')
        eval_dataset = VOCDetection(root = data_path_val,
                                    year = "2007",
                                    image_set = "test",
                                    transform = transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                    ]))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    return eval_dataset