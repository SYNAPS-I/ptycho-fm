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
    expected = np.random.default_rng(17 + 4).poisson(clean[0]) * 10
    np.testing.assert_allclose(image, expected, atol=1e-5)
    # Single-file indexing reproduces the corresponding CombinedDataset offset.
    single = PtychographyDataset(tmp_path / 'b_dp.hdf5', scale=10,
                                default_normalization=1, deterministic_noise=True,
                                noise_seed=21)
    torch.testing.assert_close(single[0][0], dataset[4][0], rtol=0, atol=0)


def test_packed_noise_is_applied_before_normalization(tmp_path, monkeypatch):
    import h5py

    from ptycho_fm.data_simple_pack import PtychographyDatasetPacked

    raw = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    with h5py.File(tmp_path / "packed_00000.hdf5", "w") as packed:
        packed["n_dp"] = np.array([1])
        packed["dp"] = raw[None, None]
        packed["object"] = np.ones((1, 1, 8, 8), dtype=np.complex64)
        packed["probe"] = np.ones((1, 1, 1, 4, 4), dtype=np.complex64)
        packed["probe_position_y_m"] = np.zeros((1, 1), dtype=np.float32)
        packed["probe_position_x_m"] = np.zeros((1, 1), dtype=np.float32)
        packed["pixel_height_m"] = np.ones(1, dtype=np.float32)
        packed["normalization"] = np.array([4.0])

    class Noise:
        def __init__(self):
            self.inputs = []

        def poisson(self, image):
            self.inputs.append(image.copy())
            return np.full_like(image, 8.0)

    noise = Noise()
    monkeypatch.setattr(np.random, "default_rng", lambda _seed=None: noise)
    dataset = PtychographyDatasetPacked(
        tmp_path,
        apply_noise=True,
        deterministic_noise=True,
        noise_seed=9,
        scale=2.0,
        max_probe_modes=1,
    )

    diffraction_amp = dataset[0][0]
    np.testing.assert_array_equal(noise.inputs[0], raw)
    torch.testing.assert_close(diffraction_amp, torch.full((1, 4, 4), 2.0))


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


@pytest.mark.parametrize('source', ['native', 'packed'])
def test_dataloader_real_fields_are_float32(tmp_path, source):
    import h5py

    from ptycho_fm.data_simple_pack import PtychographyDatasetPacked

    if source == 'native':
        make_pair(tmp_path, dtype=np.complex128)
        dataset = CombinedDataset(tmp_path, apply_noise=False, scale=7.3,
                                  default_normalization=3.2, max_probe_modes=2)
    else:
        path = tmp_path / 'packed_00000.hdf5'
        with h5py.File(path, 'w') as packed:
            packed['n_dp'] = np.array([4])
            packed['dp'] = np.ones((1, 4, 8, 12), dtype=np.float32)
            packed['object'] = np.ones((1, 1, 32, 40), dtype=np.complex128)
            packed['probe'] = np.ones((1, 1, 1, 8, 12), dtype=np.complex128)
            packed['probe_position_y_m'] = np.array([[0, 10, 20, 30]], dtype=np.float64)
            packed['probe_position_x_m'] = np.array([[0, 12, 24, 36]], dtype=np.float64)
            packed['pixel_height_m'] = np.array([1e-8])
            packed['normalization'] = np.array([3.2], dtype=np.float64)
        dataset = PtychographyDatasetPacked(tmp_path, apply_noise=False,
                                            scale=7.3, max_probe_modes=2)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert all(torch.is_tensor(field) for field in batch)
    assert [field.dtype for field in batch] == [
        torch.float32, torch.float32, torch.float32, torch.complex64,
        torch.float32, torch.float32, torch.float32,
    ]


def test_packed_dataset_exposes_object_catalog_and_views(tmp_path):
    import h5py

    from ptycho_fm.data_simple_pack import PtychographyDatasetPacked
    from ptycho_fm.utils.data import ObjectDatasetView

    with h5py.File(tmp_path / "packed_00000.hdf5", "w") as packed:
        packed["n_dp"] = np.array([2, 3])
        packed["object_key"] = np.array(
            ["first", "second"], dtype=h5py.string_dtype("utf-8")
        )
        packed["dp"] = np.ones((2, 3, 4, 4), dtype=np.float32)
        packed["object"] = np.ones((2, 1, 8, 8), dtype=np.complex64)
        packed["probe"] = np.ones((2, 1, 1, 4, 4), dtype=np.complex64)
        packed["probe_position_y_m"] = np.zeros((2, 3), dtype=np.float32)
        packed["probe_position_x_m"] = np.zeros((2, 3), dtype=np.float32)
        packed["pixel_height_m"] = np.ones(2, dtype=np.float32)
        packed["normalization"] = np.ones(2, dtype=np.float32)

    dataset = PtychographyDatasetPacked(
        tmp_path, apply_noise=False, scale=1.0, max_probe_modes=1
    )
    assert dataset.object_offsets == [0, 2, 5]
    assert dataset.object_keys == ["first", "second"]
    second = ObjectDatasetView(dataset, 1)
    assert len(second) == 3
    assert second.pattern_shape == (4, 4)
    assert second.object_shape == (8, 8)
    assert second.get_probe_positions().shape == (3, 2)
    torch.testing.assert_close(second[0][0], dataset[2][0])
