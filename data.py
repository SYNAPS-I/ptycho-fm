import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import h5py
import pickle
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
from collections import OrderedDict
from ptychi_utils import extract_patches_fourier_shift


class PtychographyDataset(Dataset):
    """
    PyTorch Dataset for ptychography data.

    For HDF5 files: Expects paired files in the same directory:
    - *_dp.hdf5: Contains diffraction patterns
    - *_para.hdf5: Contains probe positions, object amplitude/phase, and probe information

    Args:
        file_path (str): Path to data file or corresponding parameters file(*_dp.hdf5 or *_para.hdf5)
        scale (float): Factor by which to scale all diffraction intensity to
        normalization_dict_path (str): Path to .pkl file containing dict of {object_name: normalization_factor}
        apply_noise (bool): Whether to simulate noise by sampling from a Poisson distribution (set to False for experimental data)
        cache_object (bool): Whether to cache object data in memory
    """

    def __init__(
        self,
        file_path: str,
        scale: float = 100000., 
        normalization_dict_path: Optional[str] = None,
        apply_noise: bool = True,
        cache_object: bool = True
    ):
        self.file_path = Path(file_path)
        self.scale = scale
        self.normalization_dict_path = normalization_dict_path
        self.apply_noise = apply_noise
        self.cache_object = cache_object

        # Initialize cache variables
        self._cached_object = None
        self._cached_probe_positions = None
        self._cached_probe = None
        
        # Initialize persistent HDF5 file handles (Phase 2: Persistent Handles)
        self.dp_handle = None
        self.para_handle = None

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type and find paired files for HDF5
        if self.file_path.suffix.lower() == '.hdf5':
            self._find_hdf5_pair()
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}. Only Ptychodus format .hdf5 is supported.")

        # Load data and get dimensions
        self._load_file_info()

        # Extract object name from file path
        # Assumes format: .../object_name/object_name_dp.hdf5
        self.object_name = self.dp_file.stem[:-3]  # Remove '_dp' suffix

        # Load normalization factor
        self._load_normalization()

    def _load_normalization(self):
        """
        Load normalization factor from pickle file or use default.

        If normalization_dict_path is provided, loads the dictionary and looks up
        the normalization factor using self.object_name as the key.
        Falls back to a default value if key not found or file not provided.
        """
        default_normalization = 100000.0  # Default fallback value

        if self.normalization_dict_path is not None:
            try:
                with open(self.normalization_dict_path, 'rb') as f:
                    normalization_dict = pickle.load(f)

                if not isinstance(normalization_dict, dict):
                    raise ValueError(f"Normalization file must contain a dictionary, got {type(normalization_dict)}")

                # Look up normalization factor using object name
                if self.object_name in normalization_dict:
                    self.normalization = normalization_dict[self.object_name]
                else:
                    print(f"Warning: Object '{self.object_name}' not found in normalization dictionary. "
                          f"Using default: {default_normalization}", flush=True)
                    self.normalization = default_normalization

            except FileNotFoundError:
                print(f"Warning: Normalization file not found at {self.normalization_dict_path}. "
                      f"Using default: {default_normalization}", flush=True)
                self.normalization = default_normalization
            except Exception as e:
                print(f"Warning: Error loading normalization file: {e}. "
                      f"Using default: {default_normalization}", flush=True)
                self.normalization = default_normalization
        else:
            # No normalization dict provided, use default
            self.normalization = default_normalization

    def _find_hdf5_pair(self):
        """Find the paired HDF5 files (*_dp.hdf5 and *_para.hdf5)."""
        file_stem = self.file_path.stem
        file_dir = self.file_path.parent
        
        # Determine object name and file types
        if file_stem.endswith('_dp'):
            object_name = file_stem[:-3]  # Remove '_dp' suffix
            self.dp_file = self.file_path
            self.para_file = file_dir / f"{object_name}_para{self.file_path.suffix}"
        elif file_stem.endswith('_para'):
            object_name = file_stem[:-5]  # Remove '_para' suffix
            self.para_file = self.file_path
            self.dp_file = file_dir / f"{object_name}_dp{self.file_path.suffix}"
        else:
            raise ValueError(f"HDF5 file must end with '_dp' or '_para': {self.file_path.name}")
        
        # Check that both files exist
        if not self.dp_file.exists():
            raise FileNotFoundError(f"Diffraction patterns file not found: {self.dp_file}")
        if not self.para_file.exists():
            raise FileNotFoundError(f"Parameters file not found: {self.para_file}")
    
    def _load_file_info(self):
        """Load file and extract basic information about the dataset."""
        # Load from diffraction patterns file
        with h5py.File(self.dp_file, 'r') as f:
            if 'dp' not in f.keys():
                raise KeyError(f"Missing diffraction patterns 'dp' in {self.dp_file.name}")
            self.num_patterns = f['dp'].shape[0]
            self.pattern_shape = f['dp'].shape[1:]
        
        # Load from parameters file
        with h5py.File(self.para_file, 'r') as f:
            required_keys = ['object', 'probe', 'probe_position_x_m', 'probe_position_y_m']
            missing_keys = [key for key in required_keys if key not in f.keys()]
            if missing_keys:
                raise KeyError(f"Missing required keys in {self.para_file.name}: {missing_keys}")
            
            self.object_shape = f['object'][0].shape
        
    def __len__(self) -> int:
        return self.num_patterns
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single data sample.
        
        Returns:
            tuple: (diffraction_pattern, amplitude_patch, phase_patch, probe, probe_position)
                - diffraction_amp: Single diffraction amplitude [sqrt(intensity)]
                - amplitude_patch: Corresponding object amplitude patch
                - phase_patch: Corresponding object phase patch
                - probe: Probe function (if available, else zeros)
                - probe_position: Probe position coordinates
        """
        if idx >= self.num_patterns:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_patterns} patterns")
        
        # Load data from file
        diffraction_amp, amplitude_patch, phase_patch, probe, probe_position = self._load_hdf5_pattern(idx)
        
        # Convert to tensors
        diffraction_amp = torch.from_numpy(diffraction_amp) if isinstance(diffraction_amp, np.ndarray) else diffraction_amp
        amplitude_patch = amplitude_patch if isinstance(amplitude_patch, torch.Tensor) else torch.from_numpy(amplitude_patch)
        phase_patch = phase_patch if isinstance(phase_patch, torch.Tensor) else torch.from_numpy(phase_patch)
        probe = torch.from_numpy(probe) if probe is not None else None
        probe_position = probe_position if isinstance(probe_position, torch.Tensor) else torch.from_numpy(probe_position)
        
        # Add channel dimension if needed
        if diffraction_amp.dim() == 2:
            diffraction_amp = diffraction_amp.unsqueeze(0)
        if amplitude_patch.dim() == 2:
            amplitude_patch = amplitude_patch.unsqueeze(0)
        if phase_patch.dim() == 2:
            phase_patch = phase_patch.unsqueeze(0)
        if probe is not None and probe.dim() == 2:
            probe = probe.unsqueeze(0).unsqueeze(0)
        if probe is not None and probe.dim() == 3:
            probe = probe.unsqueeze(0)
            
        return diffraction_amp, amplitude_patch, phase_patch, probe, probe_position, self.normalization, self.scale
    
    def _extract_patch(self, full_object: np.ndarray, probe_position: Tensor) -> Tensor:
        """Extract patch from full object at given probe position."""
        return extract_patches_fourier_shift(torch.from_numpy(full_object), probe_position.unsqueeze(0), (self.pattern_shape[0], self.pattern_shape[1]))[0]

    def _pad_probe(self, probe: np.ndarray, target_modes: int = 30) -> np.ndarray:
        """
        Pad probe array to have target number of modes along axis 1.

        Args:
            probe: Probe array with shape (1, N, H, W) where N is current number of modes
            target_modes: Target number of modes (default: 30)

        Returns:
            Padded probe array with shape (1, target_modes, H, W)
        """
        current_shape = probe.shape
        if len(current_shape) != 4:
            raise ValueError(f"Expected probe shape (1, N, H, W), got {current_shape}")

        current_modes = current_shape[1]
        if current_modes >= target_modes:
            # Already has enough modes, no padding needed
            return probe

        # Calculate padding: add zeros to the end of axis 1
        modes_to_add = target_modes - current_modes
        pad_shape = (current_shape[0], modes_to_add, current_shape[2], current_shape[3])

        # Create zero padding with same dtype as probe
        padding = np.zeros(pad_shape, dtype=probe.dtype)

        # Concatenate along axis 1
        padded_probe = np.concatenate([probe, padding], axis=1)

        return padded_probe

    def _get_handles(self):
        """
        Get persistent HDF5 file handles, opening them if not already open.
        
        Files stay open for the lifetime of the PtychographyDataset instance,
        only closed on cache eviction or explicit close() call.
        Uses SWMR mode for safe concurrent reading.
        """
        if self.dp_handle is None:
            self.dp_handle = h5py.File(self.dp_file, 'r', libver='latest', swmr=True)
        if self.para_handle is None:
            self.para_handle = h5py.File(self.para_file, 'r', libver='latest', swmr=True)
        return self.dp_handle, self.para_handle
    
    def close(self):
        """Close persistent HDF5 file handles."""
        if self.dp_handle is not None:
            self.dp_handle.close()
            self.dp_handle = None
        if self.para_handle is not None:
            self.para_handle.close()
            self.para_handle = None

    def _cache_object_data(self):
        """Cache object and probe position data for efficient access."""
        if self._cached_object is not None:
            return

        with h5py.File(self.para_file, 'r') as f:
            # Load pixel size (needed for position conversion)
            self.pixel_size_m = f['object'].attrs['pixel_height_m']  # Ptychodus format supports non-square pixels, but we don't consider that for now

            # Cache full object
            if self.cache_object:
                self._cached_object = f['object'][0]

            # Cache probe if small enough (padded probe will be ~15MB for (1, 30, 256, 256))
            probe_data = f['probe']
            if probe_data.nbytes < 100 * 1024 * 1024:  # Cache if < 100MB
                probe = probe_data[...]
                # Pad probe to (1, 30, H, W) if needed
                self._cached_probe = self._pad_probe(probe, target_modes=30)

            # Cache probe positions and convert from meters to pixels
            positions_m = np.column_stack([f['probe_position_y_m'][...], f['probe_position_x_m'][...]])
            self._cached_probe_positions = torch.from_numpy(positions_m / self.pixel_size_m)

        # Validate that there are the same number of probe positions as diffraction patterns
        if self._cached_probe_positions.shape[0] != self.num_patterns:
            raise ValueError(f"Mismatch in number of patterns: {self.num_patterns} diffraction patterns vs {self._cached_probe_positions.shape[0]} probe positions")
        # Initialize position origin coordinates
        self._pos_origin_coords = torch.tensor(self.object_shape, dtype=torch.float32) / 2.0
        self._pos_origin_coords = self._pos_origin_coords.round() + 0.5
        self._cached_probe_positions = self._cached_probe_positions + self._pos_origin_coords 
    
    def _load_hdf5_pattern(self, pattern_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load specific pattern from paired HDF5 files with efficient caching and persistent handles."""
        # Cache object data on first access
        self._cache_object_data()

        # Get persistent file handles (Phase 2: Persistent Handles)
        dp_handle, para_handle = self._get_handles()
        
        # Load diffraction pattern using persistent handle
        diffraction_pattern = dp_handle['dp'][pattern_idx]
        diffraction_pattern = self.normalize(diffraction_pattern)
        # Add noise by sampling from a Poisson distribution
        if self.apply_noise:
            diffraction_pattern = np.random.default_rng().poisson(diffraction_pattern)
        diffraction_amp = np.sqrt(diffraction_pattern.astype(np.float32))
        
        # Get probe position from cache
        probe_position = self._cached_probe_positions[pattern_idx]
        
        # Get probe from cache or file
        if self._cached_probe is not None:
            probe = self._cached_probe
        else:
            probe = para_handle['probe'][...]
            # Pad probe to (1, 30, H, W) if needed
            probe = self._pad_probe(probe, target_modes=30)
        
        # Get object data and extract patches
        if self._cached_object is not None:
            full_object = self._cached_object
        else:
            # If not cached, load on demand (for memory-constrained situations)
            full_object = para_handle['object'][...]
        
        # Extract patches at probe position
        patch = self._extract_patch(full_object, probe_position)
        amplitude_patch = torch.abs(patch)
        phase_patch = torch.angle(patch)
        
        return diffraction_amp, amplitude_patch, phase_patch, probe, probe_position

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return (image / self.normalization) * self.scale


class CombinedDataset(Dataset):
    """
    PyTorch Dataset for multiple ptychography datasets.

    Handles multiple pairs of .hdf5 files and provides a unified interface.
    Works with PyTorch DataLoader and DistributedSampler for DDP training.

    Args:
        file_paths: Either:
                   - List of paths to data files (*_dp.hdf5 or *_para.hdf5), OR
                   - Single directory path to scan for paired files
        **dataset_kwargs: Additional arguments passed to PtychographyDataset
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

        # Find all _dp.hdf5 files
        dp_files = list(directory.glob('*_dp.hdf5'))

        # Verify each has a matching _para.hdf5 file
        paired_files = []
        for dp_file in sorted(dp_files):
            # Extract object name
            object_name = dp_file.stem[:-3]  # Remove '_dp' suffix
            para_file = directory / f"{object_name}_para.hdf5"

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
                object_name = f.stem[:-3]
                print(f"  - {object_name}", flush=True)
            print(f"  ... ({len(paired_files) - 20} more datasets) ...", flush=True)
            for f in paired_files[-10:]:
                object_name = f.stem[:-3]
                print(f"  - {object_name}", flush=True)
        else:
            for f in paired_files:
                object_name = f.stem[:-3]
                print(f"  - {object_name}", flush=True)

        return paired_files

    def __init__(self, file_paths, rank=0, world_size=1, train_split=0.95, shuffle=True, random_seed=42, mode='train', **dataset_kwargs):
        """
        Initialize CombinedDataset with global indexing and rank-based sharding.
        
        Args:
            file_paths: Directory path or list of file paths
            rank: Rank of current process (default: 0)
            world_size: Total number of processes (default: 1)
            train_split: Fraction of data for training (default: 0.95)
            shuffle: Whether to shuffle global indices (default: True)
            random_seed: Random seed for shuffling and splitting (default: 42)
            mode: 'train' or 'val' to select which split to use (default: 'train')
            **dataset_kwargs: Additional arguments passed to PtychographyDataset
        """
        # Auto-detect if file_paths is a directory or a list
        if isinstance(file_paths, (str, Path)):
            file_path = Path(file_paths)
            if file_path.is_dir():
                # Scan directory for paired files
                self.file_paths = self.find_paired_files(file_path)
                self.data_dir = file_path
            else:
                # Single file provided as string
                self.file_paths = [file_path]
                self.data_dir = file_path.parent
        else:
            # List of files provided
            self.file_paths = [Path(f) for f in file_paths]
            # Assume all files are in the same directory (use first file's parent)
            self.data_dir = self.file_paths[0].parent if self.file_paths else None

        self.dataset_kwargs = dataset_kwargs
        self.rank = rank
        self.world_size = world_size
        self.train_split = train_split
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.mode = mode

        # Try to load index.csv to avoid opening all HDF5 files
        self.index_df = None
        self.pattern_counts: Dict[Path, int] = {}
        
        if self.data_dir is not None:
            index_csv_path = Path(self.data_dir) / 'index.csv'
            if index_csv_path.exists():
                try:
                    print(f"[Rank {rank}] Loading index.csv to avoid opening all HDF5 files...", flush=True)
                    self.index_df = pd.read_csv(index_csv_path)
                    
                    # Create a fast lookup: map filename to n_dps
                    # This is much faster than nested loops
                    csv_lookup = {}
                    for _, row in self.index_df.iterrows():
                        csv_path = Path(row['dp_path'])
                        # Get just the filename for fast matching
                        filename = csv_path.name
                        csv_lookup[filename] = int(row['n_dps'])
                    
                    # Match our file_paths to CSV entries
                    for file_path in self.file_paths:
                        if file_path.name in csv_lookup:
                            self.pattern_counts[file_path] = csv_lookup[file_path.name]
                    
                    print(f"[Rank {rank}] Loaded pattern counts for {len(self.pattern_counts)}/{len(self.file_paths)} files from index.csv", flush=True)
                except Exception as e:
                    print(f"[Rank {rank}] Warning: Could not load index.csv ({e}), falling back to opening files", flush=True)
                    self.index_df = None

        # Build global index map using index.csv when available
        # Use LRU cache to limit number of open datasets in memory (prevents OOM)
        # Cache size: increased to 64 to keep more files open (64 files × 2 handles = 128 FDs per rank)
        # This reduces cache misses and file handle churn, improving sustained IO performance
        self.max_cached_datasets = 64
        self.dataset_cache = OrderedDict()  # LRU cache: {file_path: dataset}
        self.file_info = []
        self.file_offsets = [0]  # Cumulative offsets for bisect search
        self.file_map = []  # List of file paths corresponding to offsets
        total_patterns = 0

        total_files = len(self.file_paths)
        if total_files > 100:
            if self.index_df is not None:
                print(f"[Rank {rank}] Initializing {total_files} datasets using index.csv (fast)...", flush=True)
            else:
                print(f"[Rank {rank}] Initializing {total_files} datasets (opening files, this may take a while)...", flush=True)

        for idx, file_path in enumerate(self.file_paths):
            # Show progress every 1000 files
            if total_files > 100 and (idx + 1) % 1000 == 0:
                print(f"[Rank {rank}]   Processed {idx + 1}/{total_files} files...", flush=True)
            
            # Get num_patterns from index.csv if available, otherwise create dataset to get it
            if file_path in self.pattern_counts:
                num_patterns = self.pattern_counts[file_path]
            else:
                # Fallback: create dataset to get num_patterns (opens HDF5 files)
                # Disable caching here too to prevent memory issues
                fallback_kwargs = dataset_kwargs.copy()
                fallback_kwargs['cache_object'] = False
                dataset = PtychographyDataset(str(file_path), **fallback_kwargs)
                num_patterns = len(dataset)
                # Don't keep fallback dataset in cache - we'll recreate lazily

            self.file_info.append({
                'path': file_path,
                'num_patterns': num_patterns,
                'start_idx': total_patterns,
                'end_idx': total_patterns + num_patterns
            })
            self.file_map.append(file_path)
            total_patterns += num_patterns
            self.file_offsets.append(total_patterns)

        self.total_patterns = total_patterns

        # Build global indices and apply train/val split
        global_indices = np.arange(total_patterns)
        
        # Global shuffle with deterministic seed
        # IMPORTANT: For train/val split to work correctly, both train and val must use the SAME
        # shuffled indices. When doing a split (train_split < 1.0), we always shuffle with the same
        # seed to ensure train and val use the same shuffled order (just different portions).
        # The shuffle parameter controls whether shuffling happens when train_split=1.0 (no split),
        # but when doing splits, we always shuffle to ensure consistency.
        if shuffle or (train_split < 1.0):
            # Always shuffle when doing splits to ensure train and val use same shuffled indices
            rng = np.random.default_rng(random_seed)
            rng.shuffle(global_indices)
        
        # Split into train and val
        train_size = int(total_patterns * train_split)
        train_indices = global_indices[:train_size]
        val_indices = global_indices[train_size:]
        
        # Select indices based on mode
        if mode == 'train':
            selected_indices = train_indices
            split_name = 'train'
        else:
            selected_indices = val_indices
            split_name = 'val'
        
        # Rank-based sharding: Each rank gets a subset of indices
        total_selected = len(selected_indices)
        start = total_selected * rank // world_size
        end = total_selected * (rank + 1) // world_size
        self.current_indices = selected_indices[start:end]
        
        print(f"[Rank {rank}] CombinedDataset ({mode}): {len(self.file_paths)} files, {total_patterns} total patterns", flush=True)
        print(f"[Rank {rank}]   Global {split_name} split: {total_selected} patterns", flush=True)
        print(f"[Rank {rank}]   Rank shard: {len(self.current_indices)} patterns (indices {start} to {end-1})", flush=True)

    def __len__(self) -> int:
        """Return number of patterns in current rank's shard."""
        return len(self.current_indices)

    def __getitem__(self, idx: int):
        """
        Get a sample by rank-local index.

        Maps rank-local index to global index, then to file and local file index.
        Uses LRU cache to limit number of open datasets in memory.
        """
        if idx >= len(self.current_indices):
            raise IndexError(f"Index {idx} out of range for rank shard with {len(self.current_indices)} patterns")

        # Map rank-local index to global index
        global_idx = self.current_indices[idx]
        
        # Map global index to file using bisect (fast O(log n) search)
        import bisect
        file_idx = bisect.bisect_right(self.file_offsets, global_idx) - 1
        
        # Calculate local index within file
        local_idx = global_idx - self.file_offsets[file_idx]
        file_path = self.file_map[file_idx]
        
        # Lazy dataset creation with LRU cache: create dataset only when first accessed
        # This avoids opening all HDF5 files during initialization
        # LRU cache prevents memory accumulation by limiting number of open datasets
        if file_path not in self.dataset_cache:
            # Create new dataset
            lazy_kwargs = self.dataset_kwargs.copy()
            # Disable object caching for lazily created datasets to prevent memory accumulation
            # when many datasets are accessed during training (each would cache full objects)
            # Objects will be loaded on-demand from HDF5 files instead
            lazy_kwargs['cache_object'] = False
            dataset = PtychographyDataset(str(file_path), **lazy_kwargs)
            
            # Add to cache, removing oldest if cache is full (LRU eviction)
            if len(self.dataset_cache) >= self.max_cached_datasets:
                # Remove least recently used (first item in OrderedDict)
                oldest_path, oldest_dataset = self.dataset_cache.popitem(last=False)
                # Close HDF5 files if dataset has them open
                if hasattr(oldest_dataset, 'close'):
                    oldest_dataset.close()
            
            self.dataset_cache[file_path] = dataset
        else:
            # Move to end (most recently used) for LRU ordering
            dataset = self.dataset_cache.pop(file_path)
            self.dataset_cache[file_path] = dataset
        
        return dataset[local_idx]