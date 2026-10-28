import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn


def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali=False,
    seed=2048,
    num_workers=2,
) -> Iterable:
    """
    Create and return a data loader for training.

    Args:
        root_dir (str): Path to the dataset root directory.
        local_rank (int): Local rank of the process.
        batch_size (int): Batch size for the data loader.
        dali (bool, optional): Whether to use DALI for data loading. Defaults to False.
        seed (int, optional): Seed for random number generation. Defaults to 2048.
        num_workers (int, optional): Number of worker threads. Defaults to 2.

    Returns:
        Iterable: Data loader for the training dataset.
    """
    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic dataset
    if root_dir == "synthetic":
        train_set = SyntheticDataset()

    # Mxnet RecordIO dataset
    elif os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)

    # Image Folder dataset
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_set = ImageFolder(root_dir, transform)

    # DALI data loader
    if dali:
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader


class BackgroundGenerator(threading.Thread):
    """
    A background thread to prefetch data from a generator.
    """
    def __init__(self, generator, local_rank, max_prefetch=6):
        """
        Initialize the BackgroundGenerator.

        Args:
            generator (Iterable): Data generator.
            local_rank (int): Local rank of the process.
            max_prefetch (int, optional): Maximum number of prefetched items. Defaults to 6.
        """
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        """
        Run the background thread.
        """
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        """
        Get the next item from the queue.

        Returns:
            Any: Next item from the queue.
        """
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    """
    A custom DataLoader that prefetches data in a background thread.
    """
    def __init__(self, local_rank, **kwargs):
        """
        Initialize the DataLoaderX.

        Args:
            local_rank (int): Local rank of the process.
            **kwargs: Additional keyword arguments for DataLoader.
        """
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        """
        Create an iterator for the DataLoaderX.
        """
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        """
        Preload the next batch of data.
        """
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        """
        Get the next batch of data.

        Returns:
            Any: Next batch of data.
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    """
    A dataset class for Mxnet RecordIO files.
    """
    def __init__(self, root_dir, local_rank):
        """
        Initialize the MXFaceDataset.

        Args:
            root_dir (str): Path to the dataset root directory.
            local_rank (int): Local rank of the process.
        """
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank

        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        """
        Get a single item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Image and label.
        """
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.imgidx)


class SyntheticDataset(Dataset):
    """
    A synthetic dataset for testing.
    """
    def __init__(self):
        """
        Initialize the SyntheticDataset.
        """
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        """
        Get a single item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Image and label.
        """
        return self.img, self.label

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5)):
    """
    Create a DALI data iterator.

    Args:
        batch_size (int): Batch size for the data loader.
        rec_file (str): Path to the RecordIO file.
        idx_file (str): Path to the index file.
        num_threads (int): Number of worker threads.
        initial_fill (int, optional): Initial fill size. Defaults to 32768.
        random_shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        prefetch_queue_depth (int, optional): Prefetch queue depth. Defaults to 1.
        local_rank (int, optional): Local rank of the process. Defaults to 0.
        name (str, optional): Name of the reader. Defaults to "reader".
        mean (tuple, optional): Mean values for normalization. Defaults to (127.5, 127.5, 127.5).
        std (tuple, optional): Standard deviation values for normalization. Defaults to (127.5, 127.5, 127.5).

    Returns:
        DALIWarper: DALI data iterator.
    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", result_type=types.RGB)
        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_results(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    """
    A wrapper class for DALI data iterator.
    """
    def __init__(self, dali_iter):
        """
        Initialize the DALIWarper.

        Args:
            dali_iter (DALIClassificationIterator): DALI data iterator.
        """
        self.iter = dali_iter

    def __next__(self):
        """
        Get the next batch of data.

        Returns:
            tuple: Image and label tensors.
        """
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        """
        Create an iterator for the DALIWarper.
        """
        return self

    def reset(self):
        """
        Reset the DALI iterator.
        """
        self.iter.reset()