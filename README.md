# PatchDEMUX: A Certifiably Robust Framework for Multi-label Classifiers Against Adversarial Patches

Coming soon...

## Directory Structure

```
multi-label-patchcleanser/
├── PatchCleanser/              # Original PatchCleanser implementation
│   ├── assets/                 # Documentation images
│   ├── checkpoints/            # Model checkpoints
│   ├── data/                   # Dataset files (ImageNette)
│   ├── dump/                   # Output dumps and predictions
│   ├── misc/                   # Miscellaneous utilities
│   ├── utils/                  # Core utilities (cutout, defense, setup)
│   ├── pc_certification.py    # Certification script
│   ├── pc_clean_acc.py        # Clean accuracy evaluation
│   ├── train_model.py         # Model training
│   └── vanilla_clean_acc.py   # Vanilla accuracy evaluation
├── certify/                   # Certification modules
├── defenses/                  # Defense implementations
│   └── patchcleanser/         # PatchCleanser defense utilities
├── inference/                 # Inference scripts
├── preprocessing/             # Data preprocessing utilities
├── performance/               # Performance evaluation
├── train/                     # Training scripts for different datasets
├── utils/                     # Shared utilities
├── scripts/                   # SLURM job scripts
│   ├── certify.slurm         # ViT certification jobs
│   ├── certifyf.slurm        # ResNet/ViT certification jobs
│   ├── cache.slurm           # Caching jobs
│   ├── train_cutout.slurm    # Training with cutout augmentation
│   └── pascal_voc/           # Pascal VOC specific scripts
├── dump/                      # Output data and metrics
└── runtime/                   # Runtime evaluation data
```

## Key Components

- **Certification**: Scripts for certifying robustness against adversarial patches
- **Defense**: PatchCleanser defense implementation for multi-label classification
- **Training**: Model training with cutout augmentations and EMA
- **Evaluation**: Clean accuracy and runtime performance evaluation
- **Datasets**: Support for MS-COCO, Pascal VOC, and ImageNette

## Scripts

- `scripts/certify.slurm`: ViT model certification
- `scripts/certifyf.slurm`: ResNet/ViT certification with residual robustness
- `scripts/cache.slurm`: Generate cached outputs for faster evaluation
- `scripts/train_cutout.slurm`: Train models with cutout augmentation

## Usage

The repository uses SLURM job scripts for running experiments on compute clusters. Key parameters include:
- Patch size, mask numbers for first/second rounds
- Batch size and GPU configuration
- Model paths and dataset directories
- Threshold values for certification

## Models

Supports both ResNet (TResNet-L) and Vision Transformer (Q2L-CvT) architectures for multi-label classification tasks.

NOTE: We should mention here that the cached and non-cached APIs have general quantitative agreement, but the raw outputs are not always the exact same, likely due to rounding errors.