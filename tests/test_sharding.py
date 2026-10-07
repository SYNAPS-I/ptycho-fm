"""External train/validation splitting and static/dynamic rank sharding."""

import torch
from torch.utils.data import DataLoader, DistributedSampler, random_split

from ptycho_fm.data import CombinedDataset, RankShardedSubset
from ptycho_fm.utils.data import ObjectDatasetView, split_dataset_by_object

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


def test_object_level_split_keeps_every_object_whole(tmp_path):
    for name in ("a", "b", "c", "d"):
        make_pair(tmp_path, name)
    dataset = CombinedDataset(tmp_path, apply_noise=False)

    train, validation, train_objects, val_objects = split_dataset_by_object(
        dataset, train_fraction=0.5, random_seed=11
    )
    repeated = split_dataset_by_object(dataset, train_fraction=0.5, random_seed=11)

    assert train_objects == repeated[2]
    assert val_objects == repeated[3]
    assert set(train_objects).isdisjoint(val_objects)
    assert set(train_objects) | set(val_objects) == set(range(4))

    for subset, object_indices in ((train, train_objects), (validation, val_objects)):
        expected = {
            pattern_index
            for object_index in object_indices
            for pattern_index in range(
                dataset.object_offsets[object_index],
                dataset.object_offsets[object_index + 1],
            )
        }
        assert set(subset.indices) == expected

    view = ObjectDatasetView(dataset, val_objects[0])
    assert len(view) == 4
    assert view.pattern_shape == (8, 12)
    assert view.object_shape == (32, 40)
    assert view.get_probe_positions().shape == (4, 2)
    for local_index in range(len(view)):
        global_index = dataset.object_offsets[val_objects[0]] + local_index
        torch.testing.assert_close(view[local_index][0], dataset[global_index][0])


def test_object_level_subset_fraction_selects_whole_prefix_objects(tmp_path):
    for name in ("a", "b", "c", "d"):
        make_pair(tmp_path, name)
    dataset = CombinedDataset(tmp_path, apply_noise=False)
    train, validation, train_objects, val_objects = split_dataset_by_object(
        dataset, train_fraction=0.5, random_seed=3, subset_fraction=0.5
    )
    assert set(train_objects + val_objects) == {0, 1}
    assert len(train) == len(validation) == 4
