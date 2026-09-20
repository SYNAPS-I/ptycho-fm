"""Deterministic noise and sampler continuation across ranks and epochs."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler

from ptycho_fm.data import CombinedDataset, PtychographyDataset
from ptycho_fm.utils.data import ResumeSampler
from tests.test_data_refactor import make_pair


@pytest.mark.parametrize('rank', [0, 1])
def test_resume_sampler_matches_uninterrupted_indices(rank):
    dataset = list(range(23))
    sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=12)
    sampler.set_epoch(3)
    expected = list(sampler)
    resumed = ResumeSampler(sampler, resume_epoch=3, resume_num_samples=5)
    loader = DataLoader(dataset, sampler=resumed, batch_size=3)
    expected_batches = len(loader)
    assert list(resumed) == expected[5:]
    assert len(loader) == expected_batches
    assert torch.cat(list(loader)).tolist() == expected[5:]
    assert len(loader) == expected_batches
    resumed.set_epoch(4)
    assert list(resumed) == list(sampler)
    assert len(resumed) == len(sampler)


def test_noise_is_repeatable_and_uses_global_index(tmp_path):
    make_pair(tmp_path, 'a')
    make_pair(tmp_path, 'b')
    dataset = CombinedDataset(tmp_path, scale=10, default_normalization=1,
                              deterministic_noise=True, noise_seed=17)
    first = dataset[0][0]
    torch.testing.assert_close(first, dataset[0][0], rtol=0, atol=0)
    # Both fixture files have identical clean images; global indices must differ.
    assert not torch.equal(first, dataset[4][0])
    image = dataset[4][0][0].numpy() ** 2
    _, _, clean, *_ = make_pair(tmp_path / 'reference')
    expected = np.random.default_rng(17 + 4).poisson(clean[0] * 10)
    np.testing.assert_allclose(image, expected, atol=1e-5)
    # Single-file indexing reproduces the corresponding CombinedDataset offset.
    single = PtychographyDataset(tmp_path / 'b_dp.hdf5', scale=10,
                                default_normalization=1, deterministic_noise=True,
                                noise_seed=21)
    torch.testing.assert_close(single[0][0], dataset[4][0], rtol=0, atol=0)


def test_dataset_identity_detects_changed_file_order_and_content_metadata(tmp_path):
    from ptycho_fm.utils.data import dataset_fingerprint

    make_pair(tmp_path, 'a')
    make_pair(tmp_path, 'b')
    dataset = CombinedDataset(tmp_path, apply_noise=False)
    original = dataset_fingerprint(dataset)
    dataset.file_paths.reverse()
    assert dataset_fingerprint(dataset) != original
    dataset.file_paths.reverse()
    assert dataset_fingerprint(dataset) == original
    import os
    path = dataset.file_paths[0]
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1000))
    assert dataset_fingerprint(dataset) != original


def test_empty_loader_rejected_before_training():
    from ptycho_fm.utils.data import validate_loader_lengths

    with pytest.raises(ValueError, match='at least one'):
        validate_loader_lengths([], [1], torch.device('cpu'))
