# Adopted from: https://github.com/Alibaba-MIIL/ASL/blob/main/src/helper_functions/helper_functions.py

import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from pycocotools.coco import COCO
import os

class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        # Obtain MSCOCO ids in the order which they appear in the directory
        self.ids = sorted(list(self.coco.imgToAnns.keys()))
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()

        # Map from official class id (i.e., the one with missing numbers which goes up to 90) to actual id (i.e., the one that goes up to 80)
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat) # Length adjusts every iteration

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        # target consists of the object annotations from the instances.json file which are associated with img_id
        target = coco.loadAnns(ann_ids)

        # Convert target for each object to a multi-label one-hot encoded representation, organized by object size
        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:                               # one-hot target for small objects
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:                             # one-hot target for medium objects
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1     # one-hot target for large objects
        target = output

        # Parse image path from instances.json file, then use PIL to import the image
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
