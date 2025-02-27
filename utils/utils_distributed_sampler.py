import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler as _DistributedSampler


def setup_seed(seed, cuda_deterministic=True):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): Seed value for random number generators.
        cuda_deterministic (bool, optional): Whether to use deterministic CUDA operations. Defaults to True.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id, num_workers, rank, seed):
    """
    Initialize the random seed for each worker in a DataLoader.

    Args:
        worker_id (int): ID of the worker.
        num_workers (int): Total number of workers.
        rank (int): Rank of the process.
        seed (int): Seed value for random number generators.
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_dist_info():
    """
    Get the distributed training information.

    Returns:
        tuple: Rank and world size of the process.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def sync_random_seed(seed=None, device="cuda"):
    """
    Synchronize the random seed across all processes in a distributed setting.

    Args:
        seed (int, optional): Seed value for random number generators. Defaults to None.
        device (str, optional): Device to use for synchronization. Defaults to "cuda".

    Returns:
        int: Synchronized seed value.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    dist.broadcast(random_num, src=0)

    return random_num.item()


class DistributedSampler(_DistributedSampler):
    """
    Distributed sampler that extends PyTorch's DistributedSampler to support synchronization of random seeds.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        num_replicas (int, optional): Number of processes participating in distributed training. Defaults to None.
        rank (int, optional): Rank of the current process within num_replicas. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        seed (int, optional): Seed value for random number generators. Defaults to 0.
    """
    def __init__(
        self,
        dataset,
        num_replicas=None,  # world_size
        rank=None,  # local_rank
        shuffle=True,
        seed=0,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.seed = sync_random_seed(seed)

    def __iter__(self):
        """
        Generate indices for the current epoch.

        Returns:
            iter: Iterator over the indices.
        """
        # Deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices = (indices * math.ceil(self.total_size / len(indices)))[
            : self.total_size
        ]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)