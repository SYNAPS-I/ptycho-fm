"""Current training loaders with deterministic mid-epoch resume for dynamic sharding."""

import hashlib
import json
from itertools import islice, pairwise
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler, Subset

from ptycho_fm.data import RankShardedSubset
from ptycho_fm.prefetcher import CUDAPrefetcher


class ObjectDatasetView(Dataset):
    """Sequential view of every diffraction pattern belonging to one object."""

    def __init__(self, dataset: Dataset, object_index: int):
        self.dataset = dataset
        self.object_index = int(object_index)
        offsets = dataset.object_offsets
        if self.object_index < 0 or self.object_index >= len(offsets) - 1:
            raise IndexError(object_index)
        self.start = int(offsets[self.object_index])
        self.stop = int(offsets[self.object_index + 1])
        self.object_key = dataset.object_keys[self.object_index]
        self._metadata = None

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.dataset[self.start + index]

    def _plot_metadata(self):
        if self._metadata is None:
            self._metadata = self.dataset.get_object_plot_metadata(self.object_index)
            position_count = len(self._metadata["positions"])
            if position_count != len(self):
                raise ValueError(
                    f"Object {self.object_key!r} has {len(self)} patterns but "
                    f"{position_count} probe positions"
                )
        return self._metadata

    @property
    def pattern_shape(self):
        return self._plot_metadata()["pattern_shape"]

    @property
    def object_shape(self):
        return self._plot_metadata()["object_shape"]

    def get_probe_positions(self):
        return self._plot_metadata()["positions"]


def split_dataset_by_object(dataset, train_fraction, random_seed, subset_fraction=None):
    """Split complete objects, returning pattern-level subsets for existing loaders."""
    train_fraction = float(train_fraction)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("data.train_split must be in the range (0, 1).")
    offsets = [int(value) for value in dataset.object_offsets]
    object_keys = list(dataset.object_keys)
    if len(offsets) != len(object_keys) + 1 or offsets[0] != 0 or offsets[-1] != len(dataset):
        raise ValueError("Dataset object catalog is inconsistent with its pattern count.")
    if any(stop <= start for start, stop in pairwise(offsets)):
        raise ValueError("Every object must contain at least one diffraction pattern.")

    object_count = len(object_keys)
    if subset_fraction is not None:
        subset_fraction = float(subset_fraction)
        if not 0.0 < subset_fraction <= 1.0:
            raise ValueError("data.subset_fraction must be in the range (0, 1].")
        object_count = int(object_count * subset_fraction)
        if object_count < 1:
            raise ValueError(
                f"data.subset_fraction={subset_fraction} results in 0 objects. "
                "Increase subset_fraction or use a larger dataset."
            )

    train_object_count = int(object_count * train_fraction)
    if train_object_count < 1 or train_object_count >= object_count:
        raise ValueError(
            f"Object-level train/validation split requires at least one object in each "
            f"split; got {object_count} selected object(s) and train_split={train_fraction}."
        )
    generator = torch.Generator().manual_seed(int(random_seed))
    permutation = torch.randperm(object_count, generator=generator).tolist()
    train_object_indices = permutation[:train_object_count]
    val_object_indices = permutation[train_object_count:]

    def pattern_indices(object_indices):
        return [
            pattern_index
            for object_index in object_indices
            for pattern_index in range(offsets[object_index], offsets[object_index + 1])
        ]

    train_subset = Subset(dataset, pattern_indices(train_object_indices))
    val_subset = Subset(dataset, pattern_indices(val_object_indices))
    return train_subset, val_subset, train_object_indices, val_object_indices


class ResumeSampler(Sampler):
    """Skip a saved prefix only in the resumed epoch, keeping length stable.

    Consuming an iterator never changes __len__; epoch-boundary decisions
    therefore remain valid with DataLoader workers and CUDA prefetching.
    """

    def __init__(self, sampler, resume_epoch, resume_num_samples):
        if resume_num_samples < 0 or resume_num_samples > len(sampler):
            raise ValueError('Resume sample position is outside the rank sampler')
        self.sampler = sampler
        self.resume_epoch = resume_epoch
        self.resume_num_samples = resume_num_samples
        self.epoch = resume_epoch
        self.sampler.set_epoch(resume_epoch)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

    @property
    def offset(self):
        return self.resume_num_samples if self.epoch == self.resume_epoch else 0

    def __iter__(self):
        return islice(iter(self.sampler), self.offset, None)

    def __len__(self):
        return len(self.sampler) - self.offset


def _subset_training_subset(train_subset_base, fraction):
    total = len(train_subset_base)
    subset_size = int(total * fraction)
    if subset_size < 1:
        raise ValueError(
            f"training.data_subsetting_schedule fraction {fraction} results in 0 samples. "
            "Increase the fraction or use a larger dataset."
        )
    if hasattr(train_subset_base, 'indices'):
        base_indices = train_subset_base.indices
        if isinstance(base_indices, torch.Tensor):
            base_indices = base_indices.tolist()
        return Subset(train_subset_base.dataset, base_indices[:subset_size])
    return Subset(train_subset_base, list(range(subset_size)))


def build_train_loader(
    fraction: float,
    train_subset_base: Subset,
    sharding_strategy: str,
    rank: int,
    world_size: int,
    debug_mode: bool,
    train_dataloader_kwargs_base: dict,
    random_seed: int,
    device: torch.device,
    use_cuda_prefetcher: bool,
    drop_last: bool,
    resume_epoch: int = 0,
    resume_num_samples: int = 0,
):
    if resume_num_samples and sharding_strategy != "dynamic":
        raise ValueError("Mid-epoch resume requires dynamic sharding")
    train_subset_epoch = _subset_training_subset(train_subset_base, fraction)

    if sharding_strategy == 'static':
        train_dataset = RankShardedSubset(
            train_subset_epoch,
            rank,
            world_size,
            debug=debug_mode,
            subset_type='train'
        )
        train_dataloader_kwargs = train_dataloader_kwargs_base.copy()
        train_loader = DataLoader(train_dataset, **train_dataloader_kwargs)
        train_sampler = None
    elif sharding_strategy == 'dynamic':
        train_dataset = train_subset_epoch
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=random_seed,
            drop_last=drop_last,
        )
        if resume_num_samples:
            train_sampler = ResumeSampler(train_sampler, resume_epoch, resume_num_samples)
        train_dataloader_kwargs = train_dataloader_kwargs_base.copy()
        train_dataloader_kwargs['sampler'] = train_sampler
        train_dataloader_kwargs['shuffle'] = False
        train_loader = DataLoader(train_dataset, **train_dataloader_kwargs)
    else:
        raise ValueError(
            f"Invalid sharding_strategy: {sharding_strategy}. Must be 'static' or 'dynamic'"
        )

    if torch.cuda.is_available() and use_cuda_prefetcher:
        train_prefetcher = CUDAPrefetcher(train_loader, device)
    else:
        train_prefetcher = train_loader

    return (
        train_dataset,
        train_loader,
        train_sampler,
        train_prefetcher,
        train_subset_epoch,
        train_dataloader_kwargs,
    )


def build_val_loader(
    val_subset: Subset,
    sharding_strategy: str,
    rank: int,
    world_size: int,
    debug_mode: bool,
    val_dataloader_kwargs_base: dict,
    random_seed: int,
    device: torch.device,
    use_cuda_prefetcher: bool,
    drop_last: bool,
):
    if sharding_strategy == 'static':
        val_dataset = RankShardedSubset(
            val_subset,
            rank,
            world_size,
            debug=debug_mode,
            subset_type='val'
        )
        val_sampler = None
        val_dataloader_kwargs = val_dataloader_kwargs_base.copy()
        val_loader = DataLoader(val_dataset, **val_dataloader_kwargs)
    elif sharding_strategy == 'dynamic':
        val_dataset = val_subset
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=random_seed,
            drop_last=drop_last,
        )
        val_dataloader_kwargs = val_dataloader_kwargs_base.copy()
        val_dataloader_kwargs['sampler'] = val_sampler
        val_dataloader_kwargs['shuffle'] = False
        val_loader = DataLoader(val_dataset, **val_dataloader_kwargs)
    else:
        raise ValueError(
            f"Invalid sharding_strategy: {sharding_strategy}. Must be 'static' or 'dynamic'"
        )

    if torch.cuda.is_available() and use_cuda_prefetcher:
        val_prefetcher = CUDAPrefetcher(val_loader, device)
    else:
        val_prefetcher = val_loader

    return val_dataset, val_loader, val_sampler, val_prefetcher, val_dataloader_kwargs



def dataset_fingerprint(dataset, normalization_path=None):
    """Hash ordered native/packed file identities and pattern offsets.

    Size and nanosecond mtime detect changed files without rereading terabytes
    of HDF5 payloads. This is an identity check, not a content-integrity hash.
    """
    paths = list(getattr(dataset, 'shard_paths', getattr(dataset, 'file_paths', [])))
    if hasattr(dataset, 'file_paths'):
        paths = [part for path in paths for part in
                 (Path(path), Path(str(path).replace('_dp.hdf5', '_para.hdf5')))]
    if normalization_path:
        paths.append(Path(normalization_path))
    files = []
    for path in paths:
        path = Path(path).resolve()
        info = path.stat()
        files.append((str(path), info.st_size, info.st_mtime_ns))
    offsets = getattr(dataset, 'shard_offsets', getattr(dataset, 'file_offsets', []))
    return hashlib.sha256(json.dumps([files, offsets, len(dataset)]).encode()).hexdigest()


def validate_loader_lengths(train_loader, val_loader, device, distributed=False):
    """Fail together before DDP can hang on an empty or uneven rank loader."""
    lengths = torch.tensor([len(train_loader), len(val_loader)], device=device)
    minimum, maximum = lengths.clone(), lengths.clone()
    if distributed:
        import torch.distributed as dist
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if (minimum == 0).any():
        raise ValueError('Every rank requires at least one training and validation batch; check split, batch size and drop_last')
    if not torch.equal(minimum, maximum):
        raise ValueError('Ranks have unequal loader lengths; use dynamic sharding or adjust data/batch sizes')
