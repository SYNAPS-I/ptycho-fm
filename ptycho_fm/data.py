"""Paired HDF5 datasets with scoped reads and optional bounded CPU array caching."""

import bisect
import math
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from ptychi.image_proc import extract_patches_fourier_shift
from torch import Tensor
from torch.utils.data import Dataset, Subset


def _load_normalization_map(path):
    if path is None:
        return {}
    with open(path, "rb") as handle:
        values = pickle.load(handle)
    if not isinstance(values, dict):
        raise TypeError("normalization file must contain a dict")
    return values


class _ArrayCache:
    """Admission-only CPU cache; the budget counts owning NumPy array payloads.

    Each process starts with an empty cache, including after fork or spawn.
    Entries that do not fit are used for the current sample without eviction.
    """

    def __init__(self, budget_mb):
        budget_mb = float(budget_mb)
        if not math.isfinite(budget_mb) or budget_mb < 0:
            raise ValueError("cache_memory_budget_mb must be finite and non-negative")
        self.budget_bytes = int(budget_mb * 1024**2)
        self.clear()

    def clear(self):
        self.arrays = {}
        self.nbytes = 0
        self._pid = os.getpid()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(arrays={}, nbytes=0)
        return state

    def load(self, key, estimated_bytes, read):
        if self._pid != os.getpid():
            self.clear()
        if key in self.arrays:
            return self.arrays[key]
        can_admit = estimated_bytes <= self.budget_bytes - self.nbytes
        array = read()
        if can_admit and array.nbytes <= self.budget_bytes - self.nbytes:
            # Own each retained allocation: views must not hide larger backing
            # buffers or cause shared storage to be counted more than once.
            if not array.flags.owndata:
                array = array.copy()
            self.arrays[key] = array
            self.nbytes += array.nbytes
        return array


class _SampleReader(Dataset):
    """Common native-resolution preprocessing for single- and multi-pair readers."""

    def __init__(
        self,
        scale=10000.0,
        apply_noise=True,
        cache_object=True,
        max_probe_modes=8,
        max_OPR_modes=1,
        cache_memory_budget_mb=512,
        deterministic_noise=False,
        noise_seed=0,
    ):
        self.scale = scale
        self.apply_noise = apply_noise
        self.deterministic_noise = deterministic_noise
        self.noise_seed = int(noise_seed)
        if self.noise_seed < 0:
            raise ValueError("noise_seed must be non-negative")
        self.cache_object = cache_object
        self.max_probe_modes = max_probe_modes
        self.max_OPR_modes = max_OPR_modes
        self.cache_memory_budget_mb = cache_memory_budget_mb
        self._array_cache = _ArrayCache(cache_memory_budget_mb)

    def close(self):
        """Release retained arrays. Previously returned samples remain valid."""
        self._array_cache.clear()

    @staticmethod
    def _layout(dp, para):
        if "dp" not in dp:
            raise KeyError(f"Missing 'dp' in {dp.filename}")
        for key in ("object", "probe", "probe_position_x_m", "probe_position_y_m"):
            if key not in para:
                raise KeyError(f"Missing '{key}' in {para.filename}")
        return int(dp["dp"].shape[0]), dp["dp"].shape[1:], para["object"].shape[1:]

    @staticmethod
    def _positions(para, num_patterns, object_shape, index=None):
        """Convert selected or all positions, casting to float32 before scaling."""
        py = para["probe_position_y_m"][...]
        px = para["probe_position_x_m"][...]
        if py.shape != (num_patterns,) or px.shape != (num_patterns,):
            raise ValueError(f"Expected {num_patterns} positions, got {py.shape} and {px.shape}")
        if num_patterns == 0:
            return torch.empty((0, 2), dtype=torch.float32)
        oh, ow = object_shape
        ry, rx = float(py.max() - py.min()), float(px.max() - px.min())
        in_pixels = 0.1 * oh < ry < 10 * oh and 0.1 * ow < rx < 10 * ow
        pixel_size = float(para["object"].attrs["pixel_height_m"])
        origin = (np.array(object_shape, dtype=np.float32) / 2.0).round() + 0.5
        if index is None:
            positions = np.column_stack((py, px)).astype(np.float32)
        else:
            positions = np.array([py[index], px[index]], dtype=np.float32)
        if not in_pixels:
            positions = positions / pixel_size
        return torch.from_numpy(positions + origin)

    @staticmethod
    def _pad_probe(probe, target_modes=8, target_OPR_modes=1):
        """Pad mode axes without truncating modes or resizing spatial dimensions."""
        if probe.ndim != 4:
            raise ValueError(f"Expected probe shape (M, N, H, W), got {probe.shape}")
        opr, modes, height, width = probe.shape
        shape = (max(opr, target_OPR_modes), max(modes, target_modes), height, width)
        if shape == probe.shape:
            return probe
        padded = np.zeros(shape, dtype=probe.dtype)
        padded[:opr, :modes] = probe
        return padded

    def _read_array(self, key, estimated_bytes, read):
        if not self.cache_object or self._array_cache.budget_bytes == 0:
            return read()
        return self._array_cache.load(key, estimated_bytes, read)

    def _read_sample(self, dp_path, para_path, index, normalization, sample_id=None):
        with (
            h5py.File(dp_path, "r", libver="latest", swmr=True) as dp,
            h5py.File(para_path, "r", libver="latest", swmr=True) as para,
        ):
            num_patterns, pattern_shape, object_shape = self._layout(dp, para)
            if index < 0 or index >= num_patterns:
                raise IndexError(index)
            image = (dp["dp"][index] / normalization) * self.scale
            if self.apply_noise:
                identity = index if sample_id is None else sample_id
                seed = self.noise_seed + identity if self.deterministic_noise else None
                rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
                image = rng.poisson(image)
            diffraction_amp = torch.from_numpy(np.sqrt(image.astype(np.float32))).unsqueeze(0)
            position = self._positions(para, num_patterns, object_shape, index)

            key = Path(para_path).resolve()
            obj_data = para["object"]
            obj = self._read_array(
                (key, "object"),
                math.prod(object_shape) * obj_data.dtype.itemsize,
                lambda: obj_data[0],
            )
            probe_data = para["probe"]
            if probe_data.ndim != 4:
                raise ValueError(f"Expected probe shape (M, N, H, W), got {probe_data.shape}")
            opr, modes, height, width = probe_data.shape
            probe_bytes = (
                max(opr, self.max_OPR_modes) * max(modes, self.max_probe_modes)
                * height * width * probe_data.dtype.itemsize
            )
            probe = self._read_array(
                (key, "probe", self.max_OPR_modes, self.max_probe_modes),
                probe_bytes,
                lambda: self._pad_probe(probe_data[...], self.max_probe_modes, self.max_OPR_modes),
            )

        patch = extract_patches_fourier_shift(
            torch.from_numpy(obj), position.unsqueeze(0), pattern_shape
        )[0]
        # A sample owns its probe tensor so downstream in-place edits cannot
        # corrupt an admitted cache entry or another sample.
        # Keep the DataLoader's real fields float32 even when HDF5 contains
        # double-precision objects/probes. Complex probes use complex64.
        probe_tensor = torch.from_numpy(probe.astype(np.complex64, copy=True))
        return (
            diffraction_amp, torch.abs(patch).unsqueeze(0).float(),
            torch.angle(patch).unsqueeze(0).float(), probe_tensor,
            position.float(), torch.tensor(normalization, dtype=torch.float32),
            torch.tensor(self.scale, dtype=torch.float32),
        )


class PtychographyDataset(_SampleReader):
    """Read one Ptychodus HDF5 pair at native spatial resolution.

    Accepts either the *_dp.hdf5 or *_para.hdf5 path. Samples contain diffraction
    amplitude, object amplitude/phase patches, the full OPR probe, position,
    normalization, and scale. Object/probe caching defaults to enabled, with
    512 MiB of array payload per worker (not total process or GPU memory).
    Mode counts are padded to max_probe_modes/max_OPR_modes without truncation.
    Normal sample reads retain neither file handles nor position arrays.
    """

    def __init__(
        self,
        file_path: str,
        scale: float = 10000.0,
        normalization_dict_path: str | None = None,
        default_normalization: float = 100000.0,
        apply_noise: bool = True,
        cache_object: bool = True,
        max_probe_modes: int = 8,
        max_OPR_modes: int = 1,
        object_name: str | None = None,
        cache_memory_budget_mb: float = 512,
        deterministic_noise: bool = False,
        noise_seed: int = 0,
    ):
        super().__init__(
            scale, apply_noise, cache_object, max_probe_modes, max_OPR_modes, cache_memory_budget_mb,
            deterministic_noise=deterministic_noise, noise_seed=noise_seed,
        )
        self.file_path = Path(file_path)
        self.normalization_dict_path = normalization_dict_path
        self.default_normalization = default_normalization
        self._cached_probe_positions = None
        if self.file_path.suffix.lower() != ".hdf5":
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
        stem = self.file_path.stem
        if stem.endswith("_dp"):
            self.dp_file = self.file_path
            self.para_file = self.file_path.with_name(f"{stem[:-3]}_para{self.file_path.suffix}")
        elif stem.endswith("_para"):
            self.para_file = self.file_path
            self.dp_file = self.file_path.with_name(f"{stem[:-5]}_dp{self.file_path.suffix}")
        else:
            raise ValueError(f"HDF5 file must end with '_dp' or '_para': {self.file_path.name}")
        for path in (self.dp_file, self.para_file):
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
        with (
            h5py.File(self.dp_file, "r", libver="latest", swmr=True) as dp,
            h5py.File(self.para_file, "r", libver="latest", swmr=True) as para,
        ):
            self.num_patterns, self.pattern_shape, self.object_shape = self._layout(dp, para)
        self.object_name = object_name if object_name is not None else self.dp_file.stem[:-3]
        values = _load_normalization_map(normalization_dict_path)
        self.normalization = float(values.get(self.object_name, default_normalization))

    def __len__(self):
        return self.num_patterns

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        return self._read_sample(self.dp_file, self.para_file, idx, self.normalization)

    def get_probe_positions(self) -> Tensor:
        """Read all positions for stitching, without retaining them on the dataset."""
        with h5py.File(self.para_file, "r", libver="latest", swmr=True) as para:
            return self._positions(para, self.num_patterns, self.object_shape)

    def _cache_positions(self):
        """Compatibility helper for callers explicitly retaining stitching positions."""
        if self._cached_probe_positions is None:
            self._cached_probe_positions = self.get_probe_positions()

    def close(self):
        super().close()
        self._cached_probe_positions = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cached_probe_positions"] = None
        return state

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return (image / self.normalization) * self.scale


class CombinedDataset(_SampleReader):
    """Globally indexed HDF5 pairs with one bounded CPU array cache per worker.

    Files are opened and closed per sample. cache_object=False (or a zero
    cache_memory_budget_mb) disables array retention; no child datasets are kept.
    A full cache serves misses without eviction. Train/validation splitting and
    rank sharding are performed externally.
    """

    @staticmethod
    def find_paired_files(directory):
        """
        Scan directory and find all objects that have paired *_dp.hdf5 and *_para.hdf5 files.

        Returns only ONE file per object (the _dp.hdf5 file).
        PtychographyDataset will automatically find and open the paired _para.hdf5 file.

        Args:
            directory: Path to directory to scan

        Returns:
            list: List of paths to _dp.hdf5 files ONLY (one per object, not both files)
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Find all _dp.hdf5 files recursively
        dp_files = list(directory.rglob('*_dp.hdf5'))

        # Verify each has a matching _para.hdf5 file
        paired_files = []
        for dp_file in sorted(dp_files):
            # Extract object name
            object_name = dp_file.stem[:-3]  # Remove '_dp' suffix
            para_file = dp_file.with_name(f"{object_name}_para.hdf5")

            if para_file.exists():
                paired_files.append(dp_file)
            else:
                print(f"Warning: Skipping {dp_file.name} - no matching {para_file.name}", flush=True)

        if len(paired_files) == 0:
            raise ValueError(f"No paired HDF5 files found in {directory}")

        print(f"Found {len(paired_files)} paired dataset(s) in {directory}", flush=True)
        # Only print first 10 and last 10 to avoid huge output files
        if len(paired_files) > 20:
            for f in paired_files[:10]:
                rel_path = f.relative_to(directory)
                object_name = f.stem[:-3]
                print(f"  - {rel_path} ({object_name})", flush=True)
            print(f"  ... ({len(paired_files) - 20} more datasets) ...", flush=True)
            for f in paired_files[-10:]:
                rel_path = f.relative_to(directory)
                object_name = f.stem[:-3]
                print(f"  - {rel_path} ({object_name})", flush=True)
        else:
            for f in paired_files:
                rel_path = f.relative_to(directory)
                object_name = f.stem[:-3]
                print(f"  - {rel_path} ({object_name})", flush=True)

        return paired_files

    @staticmethod
    def derive_relative_path(file_path: Path, base_dir: Path | None) -> Path | None:
        """Return file_path relative to base_dir when possible."""
        if base_dir is None:
            return None
        try:
            return Path(file_path).resolve().relative_to(Path(base_dir).resolve())
        except (OSError, ValueError):
            return None

    @staticmethod
    def derive_object_name(file_path: Path, base_dir: Path | None) -> str:
        """Derive a unique object name from a file path relative to base_dir."""
        file_path = Path(file_path)
        rel = CombinedDataset.derive_relative_path(file_path, base_dir)
        if rel is not None:
            rel_no_suffix = rel.with_suffix('')
            rel_name = rel_no_suffix.name
            if rel_name.endswith('_dp'):
                rel_no_suffix = rel_no_suffix.with_name(rel_name[:-3])
            return rel_no_suffix.as_posix()

        stem = file_path.stem
        return stem.removesuffix('_dp')

    def __init__(
        self, file_paths, rank=0, world_size=1, debug=False, max_files=None,
        scale=10000.0, normalization_dict_path=None, default_normalization=100000.0,
        apply_noise=True, cache_object=True, max_probe_modes=8, max_OPR_modes=1,
        cache_memory_budget_mb=512,
        deterministic_noise=False,
        noise_seed=0,
    ):
        super().__init__(
            scale, apply_noise, cache_object, max_probe_modes, max_OPR_modes, cache_memory_budget_mb,
            deterministic_noise=deterministic_noise, noise_seed=noise_seed,
        )
        self.data_dir = Path(file_paths)
        self.file_paths = self.find_paired_files(self.data_dir)
        if max_files is not None:
            if max_files < 0:
                raise ValueError("max_files must be non-negative")
            self.file_paths = self.file_paths[:max_files]
        self.rank, self.world_size = rank, world_size
        self.debug, self.debug_call_count = debug, 0

        values = _load_normalization_map(normalization_dict_path)
        self._norm_by_path = {
            path: float(values.get(self.derive_object_name(path, self.data_dir), default_normalization))
            for path in self.file_paths
        }
        csv_counts = {}
        index_csv = self.data_dir / "index.csv"
        if index_csv.exists():
            for _, row in pd.read_csv(index_csv).iterrows():
                relative = Path(row["dp_path"])
                if relative.is_absolute():
                    raise ValueError(f"index.csv dp_path must be relative: {relative}")
                full_path = (self.data_dir / relative).resolve()
                if not full_path.is_relative_to(self.data_dir.resolve()):
                    raise ValueError(f"index.csv dp_path escapes data directory: {relative}")
                if not full_path.is_file():
                    raise FileNotFoundError(f"index.csv dp_path not found: {relative}")
                raw_count = float(row["n_dps"])
                if not math.isfinite(raw_count) or raw_count < 0 or not raw_count.is_integer():
                    raise ValueError(f"index.csv n_dps must be a non-negative integer: {row['n_dps']}")
                count = int(raw_count)
                if full_path in csv_counts and csv_counts[full_path] != count:
                    raise ValueError(f"Conflicting index.csv counts for {relative}")
                csv_counts[full_path] = count

        self.file_offsets = [0]
        for path in self.file_paths:
            count = csv_counts.get(path.resolve())
            if count is None:
                with h5py.File(path, "r", libver="latest", swmr=True) as dp:
                    count = int(dp["dp"].shape[0])
            self.file_offsets.append(self.file_offsets[-1] + count)
        self.total_patterns = self.file_offsets[-1]
        print(
            f"[Rank {rank}] CombinedDataset: {len(self.file_paths)} files, "
            f"{self.total_patterns} patterns (splitting and sharding handled externally)",
            flush=True,
        )

    def __len__(self):
        return self.total_patterns

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        file_idx = bisect.bisect_right(self.file_offsets, idx) - 1
        local_idx = idx - self.file_offsets[file_idx]
        path = self.file_paths[file_idx]
        if self.debug and self.debug_call_count < 15:
            print(
                f"[DEBUG Rank {self.rank}] DatasetIdx={idx}, File={path.name}, Pattern={local_idx}",
                flush=True,
            )
            self.debug_call_count += 1
        para_path = path.with_name(f"{path.stem[:-3]}_para{path.suffix}")
        return self._read_sample(path, para_path, local_idx, self._norm_by_path[path], sample_id=idx)


class RankShardedSubset(Dataset):
    """
    Wrapper class that applies rank-based sharding to a PyTorch Subset.
    
    This class takes a Subset object (typically from random_split) and applies
    rank sharding to distribute the subset's indices across multiple processes
    in distributed training.
    
    Args:
        subset: PyTorch Subset object containing a dataset and indices
        rank: Rank of current process (default: 0)
        world_size: Total number of processes (default: 1)
    
    Example:
        >>> full_dataset = CombinedDataset(file_paths, ...)
        >>> train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
        >>> train_dataset = RankShardedSubset(train_subset, rank=0, world_size=8)
        >>> val_dataset = RankShardedSubset(val_subset, rank=0, world_size=8)
    """
    
    def __init__(self, subset: Subset, rank: int = 0, world_size: int = 1, debug: bool = False, subset_type: str = 'unknown'):
        """
        Initialize RankShardedSubset with rank-based sharding.
        
        Args:
            subset: PyTorch Subset object
            rank: Rank of current process
            world_size: Total number of processes
            debug: If True, enable debug logging (default: False)
            subset_type: Type of subset ('train' or 'val') for debug logging (default: 'unknown')
        """
        self.subset = subset
        self.rank = rank
        self.world_size = world_size
        self.debug = debug
        self.subset_type = subset_type
        self.debug_call_count = 0  # Track number of __getitem__ calls for debug logging
        
        # Extract indices from the Subset
        if hasattr(subset, 'indices'):
            subset_indices = subset.indices
        else:
            raise AttributeError("Subset object does not have 'indices' attribute. "
                               "This may indicate an incompatible PyTorch version.")
        
        # Convert to list if it's a tensor for easier handling
        if isinstance(subset_indices, torch.Tensor):
            subset_indices = subset_indices.tolist()
        
        # Apply rank-based sharding to the subset's indices
        total_indices = len(subset_indices)
        start = total_indices * rank // world_size
        end = total_indices * (rank + 1) // world_size
        self.sharded_indices = subset_indices[start:end]
        
        print(f"[Rank {rank}] RankShardedSubset ({subset_type}): {total_indices} total indices, "
              f"{len(self.sharded_indices)} indices for this rank (indices {start} to {end-1})", flush=True)
        
        if debug:
            print(f"[DEBUG Rank {rank}] RankShardedSubset ({subset_type}): First 10 sharded indices = {self.sharded_indices[:10]}", flush=True)
    
    def __len__(self) -> int:
        """Return number of indices in current rank's shard."""
        return len(self.sharded_indices)
    
    def __getitem__(self, idx: int):
        """
        Get a sample by rank-local index.
        
        Maps rank-local index to the original dataset's global index, then retrieves
        the sample from the underlying dataset.
        
        Args:
            idx: Rank-local index (0 to len(self.sharded_indices)-1)
        
        Returns:
            Sample from the underlying dataset
        """
        if idx >= len(self.sharded_indices):
            raise IndexError(f"Index {idx} out of range for rank shard with {len(self.sharded_indices)} indices")
        
        # self.sharded_indices contains the actual dataset indices (sliced from subset.indices)
        # So we can use them directly to access the underlying dataset
        dataset_idx = self.sharded_indices[idx]
        
        if self.debug and self.debug_call_count < 15:  # Log first 15 calls (covers first few batches)
            print(f"[DEBUG Rank {self.rank}] RankShardedSubset.__getitem__ ({self.subset_type}): "
                  f"RankLocalIdx={idx} (pos in rank shard), SplitIdx={dataset_idx} (index in {self.subset_type} split)", flush=True)
            self.debug_call_count += 1
        
        return self.subset.dataset[dataset_idx]
