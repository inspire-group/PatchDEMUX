import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from pycocotools.coco import COCO
import os

import xml.etree.ElementTree as ET

class VOCDetection(datasets.VOCDetection):
    def __init__(self, root, year = "2007", image_set = "train", transform=None):
        super(VOCDetection, self).__init__(root, year, image_set)
        self.transform = transform
        self.class_names = self.get_class_names()

    # Get list of class names by parsing each XML file
    def get_class_names(self):
        class_names = set()
        for xml_file in self.annotations:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_names.add(obj.find('name').text)
        return sorted(list(class_names))

    def __getitem__(self, index):
        img, img_info = super(VOCDetection, self).__getitem__(index)

        # List of object names 
        object_names = [obj["name"] for obj in img_info["annotation"]["object"]]

        # Create label
        target = torch.zeros(len(self.class_names))
        for name in object_names:
            target[self.class_names.index(name)] = 1
        
        if self.transform is not None:
            img = self.transform(img)
        return img, target

root_path = "/scratch/gpfs/djacob/multi-label-patchcleanser/pascal-voc/test"

import torchvision.transforms as transforms
im_size = 384
val_dataset = VOCDetection(root = root_path,
                            year = "2007",
                            image_set = "test",
                            transform = transforms.Compose([
                                transforms.Resize((im_size, im_size)),
                                transforms.ToTensor(),
                                # normalize, # no need, toTensor does normalization
                            ]))

breakpoint()