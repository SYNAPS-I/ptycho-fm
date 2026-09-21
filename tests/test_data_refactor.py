"""Scoped HDF5 reads, native OPR samples, and worker-local memory budgets."""

import multiprocessing
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from ptycho_fm.data import CombinedDataset, PtychographyDataset, _ArrayCache
from ptycho_fm.prefetcher import CUDAPrefetcher


def make_pair(directory, name="sample", opr=1, modes=2, meters=False, dtype=np.complex64,
              pattern_shape=(8, 12)):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    image = rng.uniform(1, 10, (4, *pattern_shape)).astype(np.float32)
    obj = (rng.uniform(0.2, 1, (32, 40)) * np.exp(1j * rng.normal(size=(32, 40)))).astype(dtype)
    probe = (rng.normal(size=(opr, modes, *pattern_shape)) + 1j).astype(dtype)
    # Includes fractional positions and patches crossing an object boundary.
    positions = np.array([[-16.25, -20.5], [-2.25, 1.75], [0.125, 0.375], [14.25, 18.5]])
    raw_positions = positions * 1e-8 if meters else positions
    dp_path = directory / f"{name}_dp.hdf5"
    para_path = directory / f"{name}_para.hdf5"
    with h5py.File(dp_path, "w") as dp:
        dp["dp"] = image
    with h5py.File(para_path, "w") as para:
        para["object"] = obj[None]
        para["object"].attrs["pixel_height_m"] = 1e-8
        para["probe"] = probe
        para["probe_position_y_m"] = raw_positions[:, 0]
        para["probe_position_x_m"] = raw_positions[:, 1]
    return dp_path, para_path, image, obj, probe


def assert_samples_equal(actual, expected):
    assert len(actual) == len(expected) == 7
    for a, b in zip(actual, expected, strict=True):
        if torch.is_tensor(a):
            torch.testing.assert_close(a, b)
        else:
            assert a == b


@pytest.mark.parametrize("meters", [False, True])
@pytest.mark.parametrize("cache", [False, True])
def test_combined_matches_single_pair_and_fixture(tmp_path, meters, cache):
    pairs = [make_pair(tmp_path / "nested", meters=meters),
             make_pair(tmp_path, "other", meters=meters)]
    norm_path = tmp_path / "norm.pkl"
    norm_path.write_bytes(pickle.dumps({"nested/sample": 3.0}))
    settings = {"scale": 7.0, "apply_noise": False, "max_probe_modes": 3,
                "normalization_dict_path": norm_path, "default_normalization": 2.0}
    combined = CombinedDataset(tmp_path, cache_object=cache, **settings)
    expected_positions = torch.tensor([[0.25, 0.0], [14.25, 22.25],
                                       [16.625, 20.875], [30.75, 39.0]])
    for file_index, (path, _, image, _, probe) in enumerate(pairs):
        name = "nested/sample" if file_index == 0 else "other"
        expected_norm = 3.0 if file_index == 0 else 2.0
        single = PtychographyDataset(path, object_name=name, cache_object=False, **settings)
        for i in range(len(single)):
            sample = combined[file_index * 4 + i]
            assert_samples_equal(sample, single[i])
            assert sample[5:] == (expected_norm, 7.0)
            np.testing.assert_allclose(sample[0][0], np.sqrt(image[i] / expected_norm * 7.0))
            torch.testing.assert_close(sample[4], expected_positions[i])
            np.testing.assert_array_equal(sample[3][:, :2], probe)
            assert torch.count_nonzero(sample[3][:, 2:]) == 0
    if not cache:
        assert combined._array_cache.nbytes == 0
        assert not combined._array_cache.arrays


@pytest.mark.parametrize("opr,modes,target_opr,target_modes", [
    (2, 3, 4, 5), (2, 5, 4, 3), (3, 4, 1, 2), (1, 2, 1, 2),
])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_opr_padding_and_cache_parity(tmp_path, opr, modes, target_opr, target_modes, dtype):
    path, _, image, _, probe = make_pair(tmp_path, opr=opr, modes=modes, dtype=dtype)
    settings = {"apply_noise": False, "scale": 2.0, "default_normalization": 4.0,
                "max_OPR_modes": target_opr, "max_probe_modes": target_modes}
    cached = PtychographyDataset(path, **settings)
    uncached = PtychographyDataset(path, cache_object=False, **settings)
    assert cached._array_cache.nbytes == 0
    for i in range(len(cached)):
        sample = cached[i]
        assert_samples_equal(sample, uncached[i])
        assert sample[0].shape == sample[1].shape == sample[2].shape == (1, 8, 12)
        np.testing.assert_array_equal(sample[0][0], np.sqrt(image[i] / 2))
        assert sample[3].shape == (max(opr, target_opr), max(modes, target_modes), 8, 12)
        np.testing.assert_allclose(sample[3][:opr, :modes], probe, rtol=1e-6, atol=1e-6)
        assert torch.count_nonzero(sample[3][opr:]) == 0
        assert torch.count_nonzero(sample[3][:, modes:]) == 0
    assert cached._cached_probe_positions is None
    retained_probe = next(v for k, v in cached._array_cache.arrays.items() if k[1] == "probe")
    assert cached._array_cache.nbytes == 32 * 40 * np.dtype(dtype).itemsize + retained_probe.nbytes


@pytest.mark.parametrize("budget_kind", ["zero", "object", "probe", "both", "too_small"])
def test_budget_admission_and_independent_arrays(tmp_path, monkeypatch, budget_kind):
    path, _, _, obj, probe = make_pair(tmp_path)
    sizes = {"zero": 0, "object": obj.nbytes, "probe": probe.nbytes,
             "both": obj.nbytes + probe.nbytes, "too_small": probe.nbytes - 1}
    dataset = PtychographyDataset(path, apply_noise=False, max_probe_modes=2,
                                 cache_memory_budget_mb=sizes[budget_kind] / 1024**2)
    counts = {"/object": 0, "/probe": 0}
    original = h5py.Dataset.__getitem__

    def record_read(array, key):
        if array.name in counts:
            counts[array.name] += 1
        return original(array, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record_read)
    first = dataset[0]
    assert_samples_equal(first, dataset[0])
    retained = {key[1] for key in dataset._array_cache.arrays}
    expected = {"zero": set(), "object": {"object"}, "probe": {"probe"},
                "both": {"object", "probe"}, "too_small": set()}[budget_kind]
    assert retained == expected
    for name in ("object", "probe"):
        assert counts[f"/{name}"] == (1 if name in expected else 2)
    assert dataset._array_cache.nbytes <= sizes[budget_kind]
    # In-place consumer edits and clear() must not corrupt another sample.
    first[3].zero_()
    np.testing.assert_array_equal(dataset[0][3], probe)
    sample = dataset[0]
    dataset.close()
    assert dataset._array_cache.nbytes == 0
    assert not dataset._array_cache.arrays
    assert_samples_equal(sample, dataset[0])


def test_combined_budget_has_no_eviction_or_per_file_allowance(tmp_path):
    _, _, _, obj, probe = make_pair(tmp_path, "a")
    make_pair(tmp_path, "b")
    dataset = CombinedDataset(tmp_path, apply_noise=False, max_probe_modes=2,
                              cache_memory_budget_mb=(obj.nbytes + probe.nbytes) / 1024**2)
    dataset[0]
    keys = list(dataset._array_cache.arrays)
    dataset[4]
    dataset[0]
    assert list(dataset._array_cache.arrays) == keys
    assert dataset._array_cache.nbytes == obj.nbytes + probe.nbytes
    assert not hasattr(dataset, "dataset_cache")
    assert not hasattr(dataset, "current_indices")


def test_cache_owns_views_and_can_admit_smaller_later_entries():
    cache = _ArrayCache(64 / 1024**2)
    backing = np.ones(1024, dtype=np.uint8)
    cache.load("view", 16, lambda: backing[:16])
    cache.load("too_big", 64, lambda: np.zeros(64, dtype=np.uint8))
    cache.load("small", 48, lambda: np.zeros(48, dtype=np.uint8))
    assert list(cache.arrays) == ["view", "small"]
    assert cache.nbytes == 64
    assert cache.arrays["view"].base is None
    backing[:] = 0
    assert cache.arrays["view"].sum() == 16


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf"), -float("inf")])
def test_invalid_budgets(tmp_path, budget):
    path, *_ = make_pair(tmp_path)
    with pytest.raises(ValueError, match="finite and non-negative"):
        PtychographyDataset(path, cache_memory_budget_mb=budget)
    with pytest.raises(ValueError, match="finite and non-negative"):
        CombinedDataset(tmp_path, cache_memory_budget_mb=budget)


def test_files_close_on_init_sample_and_exception(tmp_path, monkeypatch):
    path, *_ = make_pair(tmp_path)
    opened = []
    original = h5py.File

    def record_open(*args, **kwargs):
        handle = original(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(h5py, "File", record_open)
    dataset = PtychographyDataset(path, apply_noise=False)
    assert all(not f.id.valid for f in opened)
    dataset[0]
    assert all(not f.id.valid for f in opened)

    def fail(*args):
        raise RuntimeError("position conversion failed")

    monkeypatch.setattr(dataset, "_positions", fail)
    with pytest.raises(RuntimeError, match="position conversion failed"):
        dataset[0]
    assert all(not f.id.valid for f in opened)
    assert not hasattr(dataset, "dp_handle")
    assert not hasattr(dataset, "para_handle")


def test_index_initialization_reads_only_missing_dp_metadata(tmp_path, monkeypatch):
    a, *_ = make_pair(tmp_path, "a")
    b, *_ = make_pair(tmp_path, "b")
    (tmp_path / "index.csv").write_text("dp_path,n_dps\n./a_dp.hdf5,4\n")
    opened = []
    original = h5py.File

    def record_open(path, *args, **kwargs):
        opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", record_open)
    dataset = CombinedDataset(tmp_path)
    assert opened == [b]
    assert dataset.file_paths == [a, b]
    assert dataset.file_offsets == [0, 4, 8]
    assert dataset._array_cache.nbytes == 0
    limited = CombinedDataset(tmp_path, max_files=1)
    assert len(limited) == 4
    for index in (-1, len(dataset)):
        with pytest.raises(IndexError):
            dataset[index]


@pytest.mark.parametrize("entry", ["/absolute.hdf5,4", "a_dp.hdf5,-1", "a_dp.hdf5,1.5",
                                   "a_dp.hdf5,nan", "../escape.hdf5,4", "missing_dp.hdf5,4"])
def test_invalid_csv_raises(tmp_path, entry):
    make_pair(tmp_path, "a")
    (tmp_path / "index.csv").write_text(f"dp_path,n_dps\n{entry}\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        CombinedDataset(tmp_path)


def test_normalization_loaded_once_and_invalid_files_raise(tmp_path, monkeypatch):
    path, *_ = make_pair(tmp_path)
    norm_path = tmp_path / "norm.pkl"
    norm_path.write_bytes(pickle.dumps({"sample": 3.0}))
    original = pickle.load
    calls = []

    def load(handle):
        calls.append(handle.name)
        return original(handle)

    monkeypatch.setattr(pickle, "load", load)
    dataset = CombinedDataset(tmp_path, normalization_dict_path=norm_path, apply_noise=False)
    for i in range(3):
        assert dataset[i][5] == 3.0
    assert calls == [str(norm_path)]
    for contents in (b"invalid pickle", pickle.dumps([])):
        norm_path.write_bytes(contents)
        for factory, source in ((CombinedDataset, tmp_path), (PtychographyDataset, path)):
            with pytest.raises((TypeError, ValueError, pickle.UnpicklingError)):
                factory(source, normalization_dict_path=norm_path)
    with pytest.raises(FileNotFoundError):
        CombinedDataset(tmp_path, normalization_dict_path=tmp_path / "missing.pkl")


def test_explicit_stitching_positions_and_pickle(tmp_path):
    path, para, *_ = make_pair(tmp_path, meters=True)
    dataset = PtychographyDataset(para, apply_noise=False)
    assert dataset.dp_file == path
    positions = dataset.get_probe_positions()
    assert dataset._cached_probe_positions is None
    for i in range(len(dataset)):
        torch.testing.assert_close(positions[i], dataset[i][4])
    dataset._cache_positions()
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._cached_probe_positions is None
    assert restored._array_cache.nbytes == 0
    assert_samples_equal(restored[0], dataset[0])
    dataset.close()
    assert dataset._cached_probe_positions is None
    assert positions.shape == (4, 2)


def test_noise_is_sampled_on_every_read(tmp_path, monkeypatch):
    path, *_ = make_pair(tmp_path)
    dataset = PtychographyDataset(path, apply_noise=True)

    class Noise:
        count = 0

        def poisson(self, image):
            self.count += 1
            return np.full_like(image, self.count)

    noise = Noise()
    monkeypatch.setattr(np.random, "default_rng", lambda: noise)
    assert torch.all(dataset[0][0] == 1)
    torch.testing.assert_close(dataset[0][0], torch.full((1, 8, 12), 2**0.5))
    assert noise.count == 2


class CacheReportingDataset(CombinedDataset):
    """Return cache ownership diagnostics from actual worker processes."""

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        names = sorted({key[0].name for key in self._array_cache.arrays})
        return sample, os.getpid(), self._array_cache.nbytes, ",".join(names)


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_persistent_workers_start_empty_and_repeat_epochs(tmp_path, start_method):
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} unavailable")
    _, _, _, obj, probe = make_pair(tmp_path, "a")
    make_pair(tmp_path, "b")
    budget = obj.nbytes + probe.nbytes
    dataset = CacheReportingDataset(tmp_path, apply_noise=False, max_probe_modes=2,
                                     cache_memory_budget_mb=budget / 1024**2)
    dataset[0]  # Deliberately populate in the parent to test fork/spawn isolation.
    loader = DataLoader(Subset(dataset, [4, 5, 6, 7]), batch_size=2, num_workers=2,
                        persistent_workers=True, multiprocessing_context=start_method, timeout=45)
    try:
        epochs = [list(loader), list(loader)]
        pids = set()
        for batches in epochs:
            for sample, worker_pids, retained, names in batches:
                assert sample[0].shape == (2, 1, 8, 12)
                assert torch.all(retained == budget)
                assert all(name == "b_para.hdf5" for name in names)
                pids.update(worker_pids.tolist())
        assert len(pids) == 2
        assert os.getpid() not in pids
        for a, b in zip(*epochs, strict=True):
            assert_samples_equal(a[0], b[0])
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_prefetcher_preserves_samples(tmp_path):
    make_pair(tmp_path)
    dataset = CombinedDataset(tmp_path, apply_noise=False)
    loader = DataLoader(dataset, batch_size=2, pin_memory=True)
    expected = list(loader)
    actual = list(CUDAPrefetcher(loader, torch.device("cuda")))
    for cpu_batch, gpu_batch in zip(expected, actual, strict=True):
        assert_samples_equal([v.cpu() for v in gpu_batch], cpu_batch)
