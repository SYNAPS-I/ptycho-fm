"""Distributed train/val DataLoader construction (shared by iteration and epoch training scripts)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset, Sampler

from data import RankShardedSubset
from prefetcher import CUDAPrefetcher


class ResumeSampler(Sampler):
    """Skip the first `resume_num_samples` indices from the wrapped sampler once, on `resume_epoch`."""

    def __init__(self, sampler, resume_epoch, resume_num_samples):
        self.sampler = sampler
        self.resume_epoch = resume_epoch
        self.resume_num_samples = resume_num_samples
        self._did_skip = False
        self._current_epoch = None

    def set_epoch(self, epoch):
        self._current_epoch = epoch
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        indices = list(self.sampler)

        if (
            not self._did_skip
            and self._current_epoch == self.resume_epoch
            and self.resume_num_samples > 0
        ):
            indices = indices[self.resume_num_samples :]
            self._did_skip = True

        return iter(indices)

    def __len__(self):
        n = len(self.sampler)
        if (
            not self._did_skip
            and self._current_epoch == self.resume_epoch
        ):
            return max(0, n - self.resume_num_samples)
        return n


def build_train_loader(
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
    is_main: bool = False,
):
    if sharding_strategy == 'static':
        if resume_num_samples > 0:
            if is_main:
                print(
                    "Warning: mid-epoch resume is not supported with static sharding; "
                    "this epoch will restart from the beginning on this rank.",
                    flush=True,
                )
            resume_num_samples = 0
        train_dataset = RankShardedSubset(
            train_subset_base,
            rank,
            world_size,
            debug=debug_mode,
            subset_type='train'
        )
        train_dataloader_kwargs = train_dataloader_kwargs_base.copy()
        train_loader = DataLoader(train_dataset, **train_dataloader_kwargs)
        train_sampler = None
    elif sharding_strategy == 'dynamic':
        train_dataset = train_subset_base
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=random_seed,
            drop_last=drop_last,
        )
        train_dataloader_kwargs = train_dataloader_kwargs_base.copy()

        if resume_num_samples > 0:
            train_sampler = ResumeSampler(
                sampler=train_sampler,
                resume_epoch=resume_epoch,
                resume_num_samples=resume_num_samples,
            )
            if is_main:
                print(
                    f"Resumed train sampler: epoch {resume_epoch}, skip {resume_num_samples} samples per rank",
                    flush=True,
                )

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
        train_subset_base,
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
