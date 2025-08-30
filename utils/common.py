# Common utility functions shared across multiple scripts

import os
import torch
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
        # NOTE: In multi-label classification, evaluations for MSCOCO are commonly done using the validation set.
        # As an example, see the Asymmetric Loss paper (https://arxiv.org/abs/2009.14119)
        instances_path_eval = os.path.join(args.data, 'annotations/instances_val2014.json')
        data_path_eval = os.path.join(args.data, 'images/val2014')
        eval_dataset = CocoDetection(data_path_eval,
                                    instances_path_eval,
                                    transforms.Compose([
                                        TransformWrapper(transforms.Resize((args.image_size, args.image_size))),
                                        TransformWrapper(transforms.ToTensor()),
                                    ]))
    elif args.dataset_name == "pascalvoc":
        data_path_eval = os.path.join(args.data, 'test')
        eval_dataset = VOCDetection(root = data_path_eval,
                                    year = "2007",
                                    image_set = "test",
                                    transform = transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                    ]))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    return eval_dataset

def load_val_dataset(args):
    """Load validation dataset based on dataset name"""
    if args.dataset_name == "pascalvoc":
        data_path_val = os.path.join(args.data, 'train')
        val_dataset = VOCDetection(root = data_path_val,
                                    year = "2007",
                                    image_set = "val",
                                    transform = transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                    ]))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    return val_dataset

def load_train_dataset(args, cutout_transform=None):
    """Load training dataset based on dataset name with optional cutout transform"""
    if args.dataset_name == "mscoco":
        instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
        data_path_train = os.path.join(args.data, 'images/train2014')
        
        # Build COCO transforms
        coco_transforms = [
            TransformWrapper(transforms.Resize((args.image_size, args.image_size))),
            TransformWrapper(transforms.ToTensor()),
        ]
        if cutout_transform is not None:
            coco_transforms.append(cutout_transform)
            
        train_dataset = CocoDetection(data_path_train,
                                      instances_path_train,
                                      transforms.Compose(coco_transforms))
    elif args.dataset_name == "pascalvoc":
        data_path_train = os.path.join(args.data, 'train')
        
        # Build VOC transforms (don't use TransformWrapper)
        voc_transforms = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        if cutout_transform is not None:
            voc_transforms.append(cutout_transform)
            
        train_dataset = VOCDetection(root = data_path_train,
                                    year = "2007", 
                                    image_set = "train",
                                    transform = transforms.Compose(voc_transforms))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    return train_dataset

def predict(model, im, target, criterion, model_config):
    """Perform inference on a batch of data with the given model"""
    thre = model_config.thre
    Sig = torch.nn.Sigmoid()

    # Compute output
    with torch.no_grad():
        output = model(im)
        output_regular = Sig(output).cpu()

    # Compute loss and predictions
    loss = criterion(output.cuda(), target.cuda())  # when using the ASL loss,sigmoid will be done in loss !
    pred = output_regular.detach().gt(thre).long()

    return pred, loss.item()
