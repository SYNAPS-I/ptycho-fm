"""External train/validation splitting and static/dynamic rank sharding."""

import torch
from torch.utils.data import DataLoader, DistributedSampler, random_split

from ptycho_fm.data import CombinedDataset, RankShardedSubset

from .test_data_refactor import make_pair


def test_static_rank_sharding(tmp_path):
    for name in ("a", "b", "c", "d"):
        make_pair(tmp_path, name)
    dataset = CombinedDataset(tmp_path, apply_noise=False)
    train, validation = random_split(dataset, [12, 4], generator=torch.Generator().manual_seed(42))
    repeated, _ = random_split(dataset, [12, 4], generator=torch.Generator().manual_seed(42))
    assert train.indices == repeated.indices
    assert set(train.indices).isdisjoint(validation.indices)
    assert set(train.indices) | set(validation.indices) == set(range(len(dataset)))

    for subset in (train, validation):
        ranks = [RankShardedSubset(subset, rank=rank, world_size=2) for rank in range(2)]
        assert set(ranks[0].sharded_indices).isdisjoint(ranks[1].sharded_indices)
        assert set(ranks[0].sharded_indices) | set(ranks[1].sharded_indices) == set(subset.indices)
        for rank in ranks:
            loader = DataLoader(rank, batch_size=2)
            assert sum(batch[0].shape[0] for batch in loader) == len(rank)
            for actual, expected in zip(rank[0], dataset[rank.sharded_indices[0]], strict=True):
                if torch.is_tensor(actual):
                    torch.testing.assert_close(actual, expected)
                else:
                    assert actual == expected


def test_dynamic_rank_sharding(tmp_path):
    for name in ("a", "b", "c", "d"):
        make_pair(tmp_path, name)
    dataset = CombinedDataset(tmp_path, apply_noise=False, cache_object=False)
    train, _ = random_split(dataset, [12, 4], generator=torch.Generator().manual_seed(42))
    samplers = [
        DistributedSampler(train, num_replicas=2, rank=rank, seed=42)
        for rank in range(2)
    ]
    epochs = []
    for epoch in range(2):
        for sampler in samplers:
            sampler.set_epoch(epoch)
        indices = [list(sampler) for sampler in samplers]
        assert set(indices[0]).isdisjoint(indices[1])
        assert set(indices[0]) | set(indices[1]) == set(range(len(train)))
        for sampler in samplers:
            loader = DataLoader(train, sampler=sampler, batch_size=2)
            assert sum(batch[0].shape[0] for batch in loader) == 6
        epochs.append(indices)
    assert epochs[0] != epochs[1]
    assert dataset._array_cache.nbytes == 0
