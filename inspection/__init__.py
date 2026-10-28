from .losses import AgeLoss  # Import custom AgeLoss function
from .verification import FGNetVerification, CelebAVerification, RAFVerification, LAPVerification  # Import custom verification functions
from .task_name import ANALYSIS_TASKS  # Import task names for analysis
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # Import default mean and std for ImageNet
from timm.data import Mixup  # Import Mixup function from timm
from timm.data import create_transform  # Import create_transform function from timm

import torch  # Import PyTorch library
import numpy as np  # Import NumPy library
import torch.distributed as dist  # Import distributed utilities from PyTorch
import os  # Import OS library

from typing import Iterable  # Import Iterable type for type hinting
from functools import partial  # Import partial function from functools
from torchvision import transforms  # Import transforms from torchvision
from utils.utils_distributed_sampler import DistributedSampler  # Import custom DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn  # Import utility functions for distributed sampling

from .datasets import AgeGenderDataset, CelebADataset, RAFDataset, FGnetDataset, ExpressionDataset, LAPDataset  # Import custom dataset classes
from .samplers import SubsetRandomSampler  # Import custom SubsetRandomSampler
from data import MXFaceDataset  # Import custom MXFaceDataset


def get_analysis_train_dataloader(data_choose, config, local_rank) -> Iterable:
    """
    Create a training dataloader for the specified analysis task.

    Args:
        data_choose (str): The type of data to use (e.g., "recognition", "age_gender", etc.).
        config (EasyDict): Configuration dictionary containing training parameters.
        local_rank (int): Local rank for distributed training.

    Returns:
        Iterable: Training dataloader.
    """
    if data_choose == "recognition":
        batch_size = config.recognition_bz  # Batch size for recognition task
        root_dir = config.rec  # Root directory for recognition dataset
        dataset_train = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)  # Initialize MXFaceDataset

    elif data_choose == "age_gender":
        batch_size = config.age_gender_bz  # Batch size for age and gender task
        transform = create_transform(
            input_size=config.img_size,  # Image size
            scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,  # Scale augmentation
            ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,  # Ratio augmentation
            is_training=True,  # Training mode
            color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,  # Color jitter
            auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,  # Auto augmentation
            re_prob=config.AUG_REPROB,  # Random erasing probability
            re_mode=config.AUG_REMODE,  # Random erasing mode
            re_count=config.AUG_RECOUNT,  # Random erasing count
            interpolation=config.INTERPOLATION,  # Interpolation method
            mean=[0.5, 0.5, 0.5],  # Mean for normalization
            std=[0.5, 0.5, 0.5],  # Std for normalization
        )
        dataset_train = AgeGenderDataset(config=config, dataset=config.age_gender_data_list, transform=transform)  # Initialize AgeGenderDataset

    elif data_choose == "CelebA":
        batch_size = config.CelebA_bz  # Batch size for CelebA task
        transform = create_transform(
            input_size=config.img_size,  # Image size
            scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,  # Scale augmentation
            ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,  # Ratio augmentation
            is_training=True,  # Training mode
            color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,  # Color jitter
            auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,  # Auto augmentation
            re_prob=config.AUG_REPROB,  # Random erasing probability
            re_mode=config.AUG_REMODE,  # Random erasing mode
            re_count=config.AUG_RECOUNT,  # Random erasing count
            interpolation=config.INTERPOLATION,  # Interpolation method
            mean=[0.5, 0.5, 0.5],  # Mean for normalization
            std=[0.5, 0.5, 0.5],  # Std for normalization
        )
        dataset_train = CelebADataset(config=config, choose="train", transform=transform)  # Initialize CelebADataset

    elif data_choose == "expression":
        batch_size = config.expression_bz  # Batch size for expression task
        transform = create_transform(
            input_size=config.img_size,  # Image size
            scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,  # Scale augmentation
            ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,  # Ratio augmentation
            is_training=True,  # Training mode
            color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,  # Color jitter
            auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,  # Auto augmentation
            re_prob=config.AUG_REPROB,  # Random erasing probability
            re_mode=config.AUG_REMODE,  # Random erasing mode
            re_count=config.AUG_RECOUNT,  # Random erasing count
            interpolation=config.INTERPOLATION,  # Interpolation method
            mean=[0.5, 0.5, 0.5],  # Mean for normalization
            std=[0.5, 0.5, 0.5],  # Std for normalization
        )
        dataset_train = ExpressionDataset(config=config, transform=transform)  # Initialize ExpressionDataset

    rank, world_size = get_dist_info()  # Get rank and world size for distributed training
    sampler_train = DistributedSampler(
        dataset_train, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)  # Initialize DistributedSampler

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=config.train_num_workers,
        pin_memory=config.train_pin_memory,
        drop_last=True,
    )
    return data_loader_train  # Return the training dataloader


def get_mixup_fn(config):
    """
    Create a Mixup function based on the configuration.

    Args:
        config (EasyDict): Configuration dictionary containing mixup parameters.

    Returns:
        Mixup function or None.
    """
    mixup_fn = None
    mixup_active = config.AUG_MIXUP > 0 or config.AUG_CUTMIX > 0. or config.AUG_CUTMIX_MINMAX is not None  # Check if mixup is active
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG_MIXUP,  # Mixup alpha value
            cutmix_alpha=config.AUG_CUTMIX,  # Cutmix alpha value
            cutmix_minmax=config.AUG_CUTMIX_MINMAX,  # Cutmix min/max values
            prob=config.AUG_MIXUP_PROB,  # Mixup probability
            switch_prob=config.AUG_MIXUP_SWITCH_PROB,  # Mixup switch probability
            mode=config.AUG_MIXUP_MODE,  # Mixup mode
            label_smoothing=config.RAF_LABEL_SMOOTHING,  # Label smoothing value
            num_classes=config.RAF_NUM_CLASSES  # Number of classes
        )
    return mixup_fn  # Return the Mixup function


def get_analysis_val_dataloader(data_choose, config):
    """
    Create a validation dataloader for the specified analysis task.

    Args:
        data_choose (str): The type of data to use (e.g., "CelebA", "LAP", etc.).
        config (EasyDict): Configuration dictionary containing validation parameters.

    Returns:
        Validation dataloader.
    """
    if data_choose == "CelebA":
        dataset_val = CelebADataset(config=config, choose="test")  # Initialize CelebADataset for validation
    elif data_choose == "LAP":
        dataset_val = LAPDataset(config=config, choose="test")  # Initialize LAPDataset for validation
    elif data_choose == "FGNet":
        dataset_val = FGnetDataset(config=config, choose="all")  # Initialize FGnetDataset for validation
    elif data_choose == "RAF":
        dataset_val = RAFDataset(config=config, choose="test")  # Initialize RAFDataset for validation

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())  # Get indices for distributed validation
    sampler_val = SubsetRandomSampler(indices)  # Initialize SubsetRandomSampler

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.val_num_workers,
        pin_memory=config.val_pin_memory,
        drop_last=False
    )
    return data_loader_val  # Return the validation dataloader