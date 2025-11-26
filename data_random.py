"""
A small, self-contained random dataset mimicking `CombinedDataset` from `data.py`.

This module is intended for quick offline testing when you don't have real HDF5
ptychography data. It implements a `CombinedDataset` class with the same
constructor signature (a subset) and a compatible __getitem__ return value so
it can be used as a drop-in fake during development.

Notes:
- `file_paths` may be a directory path, a list of paths, or a single string.
  If a directory or single string is provided, this file will synthesize a
  small list of fake file names.
- The dataset returns tuples similar to `PtychographyDataset.__getitem__`:
  (diffraction_amp, amplitude_patch, phase_patch, probe, probe_position, normalization, scale)
  where the first five are torch.Tensors and normalization/scale are floats.
- The important attribute `file_paths` is provided so code that prints
  `len(full_dataset.file_paths)` will work without modification.

This file is purposely lightweight and has no external dependencies beyond
numpy/torch. It's not intended to perfectly model the physics—only to provide
shaped tensors and indexing behavior for training loop testing.
"""

from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Sequence, Union


class CombinedDataset(Dataset):
    """
    Minimal CombinedDataset-like dataset that generates random patterns.

    Args (subset used by main.py):
        file_paths: list, Path, or str. If directory or single path given, a few
                    synthetic file names will be created internally.
        rank: int (kept for API compatibility)
        world_size: int (kept for API compatibility)
        train_split: float fraction for train/val splitting
        shuffle: whether to shuffle global indices
        random_seed: PRNG seed for deterministic shuffling
        mode: 'train', 'val', None or 'full' (kept for API compatibility)
        debug: enable basic debug prints and debug_call_count
        patterns_per_file: number of synthetic patterns per file (default: 50)
        pattern_shape: tuple (H, W) of diffraction patterns (default: (512,512))
        num_modes: number of probe modes to generate (default: 1)
        scale: float returned as scale value
        apply_noise: if True, add Poisson-like noise to patterns (approximate)
    """

    def __init__(
        self,
        file_paths: Union[str, Path, Sequence[str]],
        rank: int = 0,
        world_size: int = 1,
        train_split: float = 0.95,
        shuffle: bool = True,
        random_seed: int = 42,
        mode: Optional[str] = None,
        debug: bool = False,
        patterns_per_file: int = 50,
        pattern_shape: tuple = (512, 512),
        num_modes: int = 1,
        scale: float = 100000.0,
        apply_noise: bool = False,
        **kwargs,
    ):
        # Normalize file_paths to a list of Path objects
        if isinstance(file_paths, (str, Path)):
            p = Path(file_paths)
            if p.is_dir():
                # synthesize a few file names
                self.file_paths = [p / f"object_{i}_dp.hdf5" for i in range(3)]
                self.data_dir = p
            else:
                # single value -> treat as single file
                self.file_paths = [p]
                self.data_dir = p.parent
        else:
            self.file_paths = [Path(fp) for fp in file_paths]
            self.data_dir = self.file_paths[0].parent if self.file_paths else None

        self.rank = rank
        self.world_size = world_size
        self.train_split = train_split
        self.shuffle = shuffle
        self.random_seed = int(random_seed)
        self.mode = mode
        self.debug = debug
        self.debug_call_count = 0

        # Synthetic dataset params
        # For multi-rank training, ensure sufficient data for all ranks after random_split
        # With train_split=0.95, val gets only 5% of data
        # Example: 8 ranks × 32 batch_size = need at least 256 val patterns to get 1 batch per rank
        # Safe default: generate enough patterns so each rank gets at least 64 val samples
        if patterns_per_file == 50:  # using default
            if world_size > 1:
                # Estimate: need (world_size * 64) / 0.05 ≈ world_size * 1280 total patterns
                # With 3 files: patterns_per_file = world_size * 427 (round up to 500)
                self.patterns_per_file = max(500, world_size * 500)
            else:
                self.patterns_per_file = 50
        else:
            self.patterns_per_file = int(patterns_per_file)
        # Hardcode image size to (256, 256) for temporary compatibility with ViT
        self.pattern_shape = (256, 256)
        self.num_modes = int(num_modes)
        self.scale = float(scale)
        self.apply_noise = bool(apply_noise)

        # Build file_info & offsets
        self.file_info = []
        self.file_map = []
        self.file_offsets = [0]
        total = 0
        for fp in self.file_paths:
            n = self.patterns_per_file
            self.file_info.append({'path': fp, 'num_patterns': n, 'start_idx': total, 'end_idx': total + n})
            self.file_map.append(fp)
            total += n
            self.file_offsets.append(total)

        self.total_patterns = total

        # Global index array
        global_indices = np.arange(self.total_patterns)
        if shuffle or (train_split < 1.0 and mode not in (None, 'full')):
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(global_indices)

        # Split
        if mode is None or mode == 'full':
            selected = global_indices
        else:
            train_size = int(self.total_patterns * self.train_split)
            train_idx = global_indices[:train_size]
            val_idx = global_indices[train_size:]
            if mode == 'train':
                selected = train_idx
            elif mode == 'val':
                selected = val_idx
            else:
                raise ValueError("Invalid mode: must be 'train', 'val', None, or 'full'")

        # For mode being None/full we don't apply rank sharding here (keeps parity with real CombinedDataset)
        if mode is None or mode == 'full':
            self.current_indices = selected
        else:
            total_sel = len(selected)
            start = total_sel * rank // world_size
            end = total_sel * (rank + 1) // world_size
            self.current_indices = selected[start:end]

        # Provide normalization dict path compatibility
        self.normalization = float(self.scale)  # simple default

        if self.debug or rank == 0:
            print(f"[Rank {rank}] CombinedDataset: {len(self.file_paths)} files, {self.total_patterns} total patterns, mode={mode}, train_split={train_split}", flush=True)
            print(f"[Rank {rank}]   After split: {len(selected)} indices selected, this rank gets {len(self.current_indices)} patterns", flush=True)

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx: int):
        if idx >= len(self.current_indices):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.current_indices)} items")

        global_idx = int(self.current_indices[idx])
        # find file using file_offsets
        import bisect
        file_idx = bisect.bisect_right(self.file_offsets, global_idx) - 1
        local_idx = global_idx - self.file_offsets[file_idx]
        file_path = self.file_map[file_idx]

        if self.debug and self.debug_call_count < 15:
            fi = self.file_info[file_idx]
            print(f"[DEBUG Rank {self.rank}] CombinedDatasetRandom.__getitem__: DatasetIdx={idx}, GlobalIdx={global_idx}, File={file_path.name}, Pattern={local_idx}/{fi['num_patterns']}", flush=True)
            self.debug_call_count += 1

        H, W = self.pattern_shape

        # Create synthetic diffraction intensity (positive) and amplitude/phase patches
        # Diffraction intensity before sqrt
        rng = np.random.default_rng(self.random_seed + global_idx)
        intensity = rng.random((H, W)).astype(np.float32) * 1e5
        if self.apply_noise:
            # approximate Poisson-like variability
            noisy = rng.poisson(intensity).astype(np.float32)
            intensity = noisy

        diffraction_amp = np.sqrt(intensity)

        # amplitude and phase patches (complex object patch represented as amplitude & phase)
        amplitude_patch = rng.random((H, W)).astype(np.float32)
        phase_patch = (rng.random((H, W)).astype(np.float32) - 0.5) * 2.0 * np.pi


        # probe: shape (1, num_modes, H, W, 2) as complex (real, imag)

        # probe: shape (1, num_modes, H, W) as complex64
        probe_real = np.zeros((1, max(1, self.num_modes), H, W), dtype=np.float32)
        probe_imag = np.zeros((1, max(1, self.num_modes), H, W), dtype=np.float32)
        # fill first mode with a smooth gaussian-ish pattern for real, random for imag
        yy = np.linspace(-1, 1, H)[:, None]
        xx = np.linspace(-1, 1, W)[None, :]
        gauss = np.exp(-((xx**2 + yy**2) * 4.0)).astype(np.float32)
        probe_real[0, 0] = gauss
        probe_imag[0, 0] = rng.random((H, W)).astype(np.float32) * 0.1
        # other modes: random complex
        for m in range(1, probe_real.shape[1]):
            probe_real[0, m] = rng.random((H, W)).astype(np.float32) * 0.2
            probe_imag[0, m] = rng.random((H, W)).astype(np.float32) * 0.2

        # Convert to torch tensors and combine to complex
        probe_real_t = torch.from_numpy(probe_real)
        probe_imag_t = torch.from_numpy(probe_imag)
        probe_t = torch.complex(probe_real_t, probe_imag_t)

        # probe_position as 2-vector
        probe_position = np.array([H // 2, W // 2], dtype=np.float32) + (rng.random(2).astype(np.float32) - 0.5) * 5.0

        # Convert to torch tensors and add channels where appropriate
        diffraction_amp_t = torch.from_numpy(diffraction_amp)
        amplitude_t = torch.from_numpy(amplitude_patch)
        phase_t = torch.from_numpy(phase_patch)
        probe_pos_t = torch.from_numpy(probe_position)

        # Ensure channel dims like real dataset: add channel dim for 2D -> (1,H,W)
        if diffraction_amp_t.dim() == 2:
            diffraction_amp_t = diffraction_amp_t.unsqueeze(0)
        if amplitude_t.dim() == 2:
            amplitude_t = amplitude_t.unsqueeze(0)
        if phase_t.dim() == 2:
            phase_t = phase_t.unsqueeze(0)

        # Return tuple compatible with PtychographyDataset.__getitem__
        return diffraction_amp_t, amplitude_t, phase_t, probe_t, probe_pos_t, float(self.normalization), float(self.scale)


if __name__ == '__main__':
    # Quick manual smoke test when running this file directly
    ds = CombinedDataset(file_paths=['/tmp/fake1_dp.hdf5', '/tmp/fake2_dp.hdf5'], patterns_per_file=10, pattern_shape=(64,64), debug=True)
    print('Dataset length:', len(ds))
    sample = ds[0]
    print('Sample shapes:', [s.shape if hasattr(s, 'shape') else type(s) for s in sample[:5]], 'norm,scale=', sample[5], sample[6])
