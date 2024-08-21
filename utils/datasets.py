import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from pycocotools.coco import COCO
import os

def split_dataset_gpu(dataset, batch_size, total_num_gpu, world_gpu_id):
    # split a PyTorch dataset into subsets for different GPUs
    '''
    INPUT:
    dataset          the dataset (torchvision.datasets)
    batch_size       number of images in a batch (int)
    total_num_gpu    total number of available GPUs to split across (int)
    world_gpu_id     the GPU for which to generate a dataset subset (int)

    OUTPUT:
    gpu_dataset      the generated dataset subset (torch.utils.data.Subset)
    start_idx        starting image index for gpu_dataset (int)
    end_idx          final image index for gpu_dataset (int)
    '''

    # Compute number of batches per GPU
    num_batches = np.ceil(len(dataset) / batch_size).astype(int)

    # By taking the floor of the quotient num_batches / total_num_gpu, we ensure all datapoints are accounted for; however this
    # means that the last GPU will always have extra batches to compute (i.e., these correspond to the remainder of the quotient)
    num_batches_gpu = np.floor(num_batches / total_num_gpu).astype(int)

    # Create GPU specific dataset
    start_idx = world_gpu_id * num_batches_gpu * batch_size
    end_idx = start_idx + num_batches_gpu * batch_size if world_gpu_id != (total_num_gpu - 1) else len(dataset)
    gpu_dataset = torch.utils.data.Subset(dataset, list(range(start_idx, end_idx)))

    return gpu_dataset, start_idx, end_idx

# Adopted from: https://github.com/Alibaba-MIIL/ASL/blob/main/src/helper_functions/helper_functions.py
class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        # Obtain MSCOCO ids in the order which they appear in the directory
        # NOTE: images with no objects are ignored. To incorporate these, use coco.getImgIds(catIds=[]) instead of coco.imgToAnns.keys()
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
            img, _ = self.transform((img, path))

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, path
    
# Adopted from: https://stackoverflow.com/questions/75722946/pytorch-custom-transformation-with-additional-argument-in-call
class TransformWrapper:
    """ A wrapper for transforms that operate on image data; allows for interoperability
        of transforms which apply solely to image data and those which apply  
        to (image_data, filename) tuples within the transforms.Compose API
    
    Args:
        transform (torchvision.transforms): transform to be wrapped
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        """
        Args:
            data: tuple containing both an image and its file name
        Returns:
            transformed image data
        """
        im, file_name = data
        return self.transform(im), file_name