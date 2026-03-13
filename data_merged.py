import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
import h5py
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from collections import OrderedDict
from utils.ptychi_utils import extract_patches_fourier_shift


class MergedShardDataset(Dataset):
    """
    Dataset for merged shard files produced by merge_dp_para_shards_append.py.

    Expects datasets:
    - dp_all, pos_x_all, pos_y_all, pos_idx_all
    - object_all, probe_all
    - dp_start, dp_count, probe_start, probe_count
    - sample_name (optional, for normalization)
    """

    def __init__(
        self,
        file_path: str,
        scale: float = 100000.0,
        normalization_dict_path: Optional[str] = None,
        apply_noise: bool = True,
        cache_object: bool = True,
        max_probe_modes: int = 8,
        pixel_size_m: Optional[float] = None,
    ):
        self.file_path = Path(file_path)
        self.scale = scale
        self.normalization_dict_path = normalization_dict_path
        self.apply_noise = apply_noise
        self.cache_object = cache_object
        self.max_probe_modes = max_probe_modes
        self.pixel_size_m_override = pixel_size_m

        self.dp_handle = None
        self._cached_object_all = None
        self._cached_probe_all = None

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self._get_handles()
        self._load_file_info()
        self._load_normalization()

    def _get_handles(self):
        if self.dp_handle is None:
            self.dp_handle = h5py.File(self.file_path, "r", libver="latest", swmr=True)
        return self.dp_handle

    def close(self):
        if self.dp_handle is not None:
            self.dp_handle.close()
            self.dp_handle = None

    def _load_file_info(self):
        f = self.dp_handle
        self.dp_all = f["dp_all"]
        self.pos_x_all = f["pos_x_all"]
        self.pos_y_all = f["pos_y_all"]
        self.object_all = f["object_all"]
        self.probe_all = f["probe_all"]
        self.dp_start = f["dp_start"][...]
        self.dp_count = f["dp_count"][...]
        self.probe_start = f["probe_start"][...]
        self.probe_count = f["probe_count"][...]

        self.sample_count = self.dp_start.shape[0]
        self.num_patterns = self.dp_all.shape[0]
        self.pattern_shape = self.dp_all.shape[1:]
        self.object_shape = self.object_all.shape[1:]

        self.dp_end = self.dp_start + self.dp_count

        if self.pixel_size_m_override is not None:
            self.pixel_size_m = float(self.pixel_size_m_override)
        else:
            self.pixel_size_m = float(self.object_all.attrs.get("pixel_height_m", 1.0))

        self._pos_origin_coords = torch.tensor(self.object_shape, dtype=torch.float32) / 2.0
        self._pos_origin_coords = self._pos_origin_coords.round() + 0.5

        if self.cache_object:
            if self.object_all.nbytes < 100 * 1024 * 1024:
                self._cached_object_all = self.object_all[...]
            if self.probe_all.nbytes < 100 * 1024 * 1024:
                self._cached_probe_all = self.probe_all[...]

        if "sample_name" in f:
            self.sample_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in f["sample_name"][...]]
        else:
            self.sample_names = None

    def _load_normalization(self):
        default_normalization = 100000.0
        self.normalization_per_sample = np.full(self.sample_count, default_normalization, dtype=np.float32)

        if self.normalization_dict_path is None or self.sample_names is None:
            return

        try:
            with open(self.normalization_dict_path, "rb") as f:
                normalization_dict = pickle.load(f)
            if not isinstance(normalization_dict, dict):
                raise ValueError(f"Normalization file must contain a dictionary, got {type(normalization_dict)}")
        except Exception:
            return

        for i, name in enumerate(self.sample_names):
            object_name = Path(name).stem[:-3] if name.endswith("_dp.hdf5") else Path(name).stem
            if object_name in normalization_dict:
                self.normalization_per_sample[i] = normalization_dict[object_name]

    def __len__(self) -> int:
        return self.num_patterns

    def _pad_probe(self, probe: np.ndarray, target_modes: int = 30) -> np.ndarray:
        if probe.ndim != 3:
            raise ValueError(f"Expected probe shape (M, H, W), got {probe.shape}")
        current_modes = probe.shape[0]
        if current_modes >= target_modes:
            return probe
        modes_to_add = target_modes - current_modes
        pad_shape = (modes_to_add, probe.shape[1], probe.shape[2])
        padding = np.zeros(pad_shape, dtype=probe.dtype)
        return np.concatenate([probe, padding], axis=0)

    def _extract_patch(self, full_object: np.ndarray, probe_position: Tensor) -> Tensor:
        return extract_patches_fourier_shift(
            torch.from_numpy(full_object),
            probe_position.unsqueeze(0),
            (self.pattern_shape[0], self.pattern_shape[1]),
        )[0]

    def __getitem__(self, idx: int):
        if idx >= self.num_patterns:
            raise IndexError(f"Index {idx} out of range for {self.num_patterns} patterns")

        sample_idx = int(np.searchsorted(self.dp_end, idx, side="right"))
        normalization = float(self.normalization_per_sample[sample_idx])

        diffraction_pattern = self.dp_all[idx]
        diffraction_pattern = (diffraction_pattern / normalization) * self.scale
        if self.apply_noise:
            diffraction_pattern = np.random.default_rng().poisson(diffraction_pattern)
        diffraction_amp = np.sqrt(diffraction_pattern.astype(np.float32))

        pos_x = self.pos_x_all[idx]
        pos_y = self.pos_y_all[idx]
        probe_position = torch.tensor([pos_y, pos_x], dtype=torch.float32) / self.pixel_size_m
        probe_position = probe_position + self._pos_origin_coords

        if self._cached_object_all is not None:
            full_object = self._cached_object_all[sample_idx]
        else:
            full_object = self.object_all[sample_idx]

        probe_start = int(self.probe_start[sample_idx])
        probe_count = int(self.probe_count[sample_idx])
        if self._cached_probe_all is not None:
            probe = self._cached_probe_all[probe_start : probe_start + probe_count]
        else:
            probe = self.probe_all[probe_start : probe_start + probe_count]
        probe = self._pad_probe(probe, target_modes=self.max_probe_modes)

        patch = self._extract_patch(full_object, probe_position)
        amplitude_patch = torch.abs(patch)
        phase_patch = torch.angle(patch)

        diffraction_amp = torch.from_numpy(diffraction_amp)
        if diffraction_amp.dim() == 2:
            diffraction_amp = diffraction_amp.unsqueeze(0)
        if amplitude_patch.dim() == 2:
            amplitude_patch = amplitude_patch.unsqueeze(0)
        if phase_patch.dim() == 2:
            phase_patch = phase_patch.unsqueeze(0)

        probe_tensor = torch.from_numpy(probe)
        probe_tensor = probe_tensor.unsqueeze(0)

        return diffraction_amp, amplitude_patch, phase_patch, probe_tensor, probe_position, normalization, self.scale


class CombinedMergedDataset(Dataset):
    """
    Dataset for multiple merged shard files (*.hdf5).
    """

    @staticmethod
    def find_shard_files(directory: Path) -> List[Path]:
        shard_files = sorted(directory.glob("*_shard*.hdf5"))
        if len(shard_files) == 0:
            raise ValueError(f"No shard files found in {directory}")
        return shard_files

    def __init__(self, file_paths, rank=0, world_size=1, debug=False, data_fraction=1.0, **dataset_kwargs):
        data_dir = Path(file_paths)
        if not data_dir.is_dir():
            raise ValueError(f"file_paths must be a directory, got: {file_paths}")

        if not (0 < data_fraction <= 1.0):
            raise ValueError(f"data_fraction must be between 0 and 1, got {data_fraction}")

        all_file_paths = self.find_shard_files(data_dir)

        if data_fraction < 1.0:
            total_files = len(all_file_paths)
            num_files_to_use = max(1, int(total_files * data_fraction))
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(total_files, size=num_files_to_use, replace=False)
            self.file_paths = [all_file_paths[i] for i in sorted(indices)]
        else:
            self.file_paths = all_file_paths

        self.dataset_kwargs = dataset_kwargs
        self.rank = rank
        self.world_size = world_size
        self.debug = debug
        self.debug_call_count = 0

        self.max_cached_datasets = 16
        self.dataset_cache = OrderedDict()
        self.file_offsets = [0]
        self.file_map = []
        total_patterns = 0

        for file_path in self.file_paths:
            with h5py.File(file_path, "r") as f:
                num_patterns = int(f["dp_all"].shape[0])
            self.file_map.append(file_path)
            total_patterns += num_patterns
            self.file_offsets.append(total_patterns)

        self.total_patterns = total_patterns
        self.current_indices = np.arange(total_patterns)

    def __len__(self) -> int:
        return len(self.current_indices)

    def __getitem__(self, idx: int):
        if idx >= len(self.current_indices):
            raise IndexError(f"Index {idx} out of range for {len(self.current_indices)} patterns")

        global_idx = self.current_indices[idx]
        import bisect

        file_idx = bisect.bisect_right(self.file_offsets, global_idx) - 1
        local_idx = global_idx - self.file_offsets[file_idx]
        file_path = self.file_map[file_idx]

        if self.debug and self.debug_call_count < 15:
            print(
                f"[DEBUG Rank {self.rank}] CombinedMergedDataset.__getitem__: "
                f"DatasetIdx={idx}, GlobalIdx={global_idx}, File={file_path.name}, Pattern={local_idx}",
                flush=True,
            )
            self.debug_call_count += 1

        if file_path not in self.dataset_cache:
            dataset = MergedShardDataset(str(file_path), **self.dataset_kwargs)
            if len(self.dataset_cache) >= self.max_cached_datasets:
                _, oldest_dataset = self.dataset_cache.popitem(last=False)
                if hasattr(oldest_dataset, "close"):
                    oldest_dataset.close()
            self.dataset_cache[file_path] = dataset
        else:
            dataset = self.dataset_cache.pop(file_path)
            self.dataset_cache[file_path] = dataset

        return dataset[local_idx]


class RankShardedSubset(Dataset):
    def __init__(self, subset: Subset, rank: int = 0, world_size: int = 1, debug: bool = False, subset_type: str = "unknown"):
        self.subset = subset
        self.rank = rank
        self.world_size = world_size
        self.debug = debug
        self.subset_type = subset_type
        self.debug_call_count = 0

        if hasattr(subset, "indices"):
            subset_indices = subset.indices
        else:
            raise AttributeError("Subset object does not have 'indices' attribute.")

        if isinstance(subset_indices, torch.Tensor):
            subset_indices = subset_indices.tolist()

        total_indices = len(subset_indices)
        start = total_indices * rank // world_size
        end = total_indices * (rank + 1) // world_size
        self.sharded_indices = subset_indices[start:end]

        if debug:
            print(
                f"[DEBUG Rank {rank}] RankShardedSubset ({subset_type}): "
                f"{len(self.sharded_indices)} indices for this rank",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.sharded_indices)

    def __getitem__(self, idx: int):
        if idx >= len(self.sharded_indices):
            raise IndexError(f"Index {idx} out of range for rank shard")
        dataset_idx = self.sharded_indices[idx]
        if self.debug and self.debug_call_count < 15:
            print(
                f"[DEBUG Rank {self.rank}] RankShardedSubset.__getitem__ ({self.subset_type}): "
                f"RankLocalIdx={idx}, SplitIdx={dataset_idx}",
                flush=True,
            )
            self.debug_call_count += 1
        return self.subset.dataset[dataset_idx]
