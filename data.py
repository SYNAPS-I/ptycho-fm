import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
import h5py
import pickle
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
from collections import OrderedDict
from utils.ptychi_utils import extract_patches_fourier_shift
import pdb

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
        cache_object (bool): Whether to cache object and probe data in memory
        max_probe_modes (int): Maximum number of probe modes to pad to (default: 8)
    """

    def __init__(
        self,
        file_path: str,
        scale: float = 100000.,
        normalization_dict_path: Optional[str] = None,
        apply_noise: bool = True,
        cache_object: bool = True,
        max_probe_modes: int = 8
    ):
        self.file_path = Path(file_path)
        self.scale = scale
        self.normalization_dict_path = normalization_dict_path
        self.apply_noise = apply_noise
        self.cache_object = cache_object
        self.max_probe_modes = max_probe_modes

        # Initialize cache variables
        self._cached_object = None
        self._cached_probe_positions = None
        self._cached_probe = None
        
        # Initialize persistent HDF5 file handles
        self.dp_handle = None
        self.para_handle = None

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type and find paired files for HDF5
        if self.file_path.suffix.lower() == '.hdf5':
            self._find_hdf5_pair()
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}. Only Ptychodus format .hdf5 is supported.")

        # Open persistent file handles early
        self._get_handles()

        # Load data and get dimensions (uses persistent handles)
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
        """Load file and extract basic information about the dataset using persistent handles."""
        # Use persistent handles (already opened in __init__)
        if 'dp' not in self.dp_handle.keys():
            raise KeyError(f"Missing diffraction patterns 'dp' in {self.dp_file.name}")
        self.num_patterns = self.dp_handle['dp'].shape[0]
        self.pattern_shape = self.dp_handle['dp'].shape[1:]

        # Load from parameters file
        required_keys = ['object', 'probe', 'probe_position_x_m', 'probe_position_y_m']
        missing_keys = [key for key in required_keys if key not in self.para_handle.keys()]
        if missing_keys:
            raise KeyError(f"Missing required keys in {self.para_file.name}: {missing_keys}")

        self.object_shape = self.para_handle['object'][0].shape
        
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
        try:
            diffraction_amp, amplitude_patch, phase_patch, probe, probe_position = self._load_hdf5_pattern(idx)
        except Exception as e:
            print(f"[Error] Failed to load pattern {idx} from {self.file_path}", flush=True)
            raise e
        
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

    def _cache_positions(self):
        """Cache probe positions on first access. Always called on first __getitem__."""
        if self._cached_probe_positions is not None:
            return

        # Load pixel size (needed for position conversion)
        self.pixel_size_m = self.para_handle['object'].attrs['pixel_height_m']

        # Cache probe positions and convert from meters to pixels
        positions_m = np.column_stack([self.para_handle['probe_position_y_m'][...], self.para_handle['probe_position_x_m'][...]])
        self._cached_probe_positions = torch.from_numpy(positions_m / self.pixel_size_m)

        # Validate that there are the same number of probe positions as diffraction patterns
        if self._cached_probe_positions.shape[0] != self.num_patterns:
            raise ValueError(f"Mismatch in number of patterns: {self.num_patterns} diffraction patterns vs {self._cached_probe_positions.shape[0]} probe positions")

        # Initialize position origin coordinates and apply offset
        self._pos_origin_coords = torch.tensor(self.object_shape, dtype=torch.float32) / 2.0
        self._pos_origin_coords = self._pos_origin_coords.round() + 0.5
        self._cached_probe_positions = self._cached_probe_positions + self._pos_origin_coords

    def _cache_object_data(self):
        """Cache full object and probe data if cache_object=True. Only called when cache_object is enabled."""
        if self._cached_object is not None:
            return

        # Cache full object
        self._cached_object = self.para_handle['object'][0]

        # Cache probe if small enough (padded probe will be ~15MB for (1, max_probe_modes, 256, 256))
        probe_data = self.para_handle['probe']
        if probe_data.nbytes < 100 * 1024 * 1024:  # Cache if < 100MB
            probe = probe_data[...]
            # Pad probe to (1, max_probe_modes, H, W) if needed
            self._cached_probe = self._pad_probe(probe, target_modes=self.max_probe_modes) 
    
    def _load_hdf5_pattern(self, pattern_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load specific pattern from paired HDF5 files with efficient caching and persistent handles."""
        # Always cache positions on first access
        self._cache_positions()

        # Cache object/probe data only if cache_object is enabled
        if self.cache_object:
            self._cache_object_data()

        # Get persistent file handles
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
            # Pad probe to (1, max_probe_modes, H, W) if needed
            probe = self._pad_probe(probe, target_modes=self.max_probe_modes)
        
        # Get object data and extract patches
        if self._cached_object is not None:
            full_object = self._cached_object
        else:
            # If not cached, load on demand (for memory-constrained situations)
            full_object = para_handle['object'][0]
        
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
    Returns the full dataset with sequential indices - train/val splitting
    should be handled externally using PyTorch's random_split() function.

    Args:
        file_paths: Directory path to scan for all paired *_dp.hdf5 and *_para.hdf5 files
        rank: Rank of current process for distributed training (default: 0)
        world_size: Total number of processes for distributed training (default: 1)
        debug: Enable debug logging for data access (default: False)
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

    def __init__(self, file_paths, rank=0, world_size=1, debug=False, data_fraction=1.0, **dataset_kwargs):
        """
        Initialize CombinedDataset with sequential global indexing.

        Args:
            file_paths: Directory path to scan for all paired files
            rank: Rank of current process (default: 0)
            world_size: Total number of processes (default: 1)
            debug: If True, enable debug logging for CSV usage and data access (default: False)
            data_fraction: Fraction of data to use (0 < data_fraction <= 1.0, default: 1.0 for all data)
            **dataset_kwargs: Additional arguments passed to PtychographyDataset
        """
        # file_paths must be a directory
        data_dir = Path(file_paths)
        if not data_dir.is_dir():
            raise ValueError(f"file_paths must be a directory, got: {file_paths}")

        # Validate data_fraction
        if not (0 < data_fraction <= 1.0):
            raise ValueError(f"data_fraction must be between 0 and 1, got {data_fraction}")

        # Scan directory for all paired files
        all_file_paths = self.find_paired_files(data_dir)
        
        # Apply data fraction if specified
        if data_fraction < 1.0:
            total_files = len(all_file_paths)
            num_files_to_use = max(1, int(total_files * data_fraction))
            rng = np.random.default_rng(seed=42)  # Use deterministic seed for reproducibility
            indices = rng.choice(total_files, size=num_files_to_use, replace=False)
            self.file_paths = [all_file_paths[i] for i in sorted(indices)]
            print(f"[Rank {rank}] Using {data_fraction*100:.1f}% of data: {len(self.file_paths)}/{total_files} files", flush=True)
            if debug:
                print(f"[DEBUG Rank {rank}] Data fraction sampling: Selected {len(self.file_paths)} out of {total_files} files", flush=True)
        else:
            self.file_paths = all_file_paths
        self.data_dir = data_dir

        self.dataset_kwargs = dataset_kwargs
        self.rank = rank
        self.world_size = world_size
        self.debug = debug
        self.debug_call_count = 0  # Track number of __getitem__ calls for debug logging

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
                    if debug:
                        print(f"[DEBUG Rank {rank}] CSV Usage: index.csv found and loaded successfully", flush=True)
                        print(f"[DEBUG Rank {rank}] CSV Usage: Matched {len(self.pattern_counts)}/{len(self.file_paths)} files from CSV", flush=True)
                        # Print sample of pattern counts
                        sample_files = list(self.pattern_counts.items())[:5]
                        for file_path, count in sample_files:
                            print(f"[DEBUG Rank {rank}] CSV Sample: {file_path.name} -> {count} patterns", flush=True)
                except Exception as e:
                    print(f"[Rank {rank}] Warning: Could not load index.csv ({e}), falling back to opening files", flush=True)
                    self.index_df = None
                    if debug:
                        print(f"[DEBUG Rank {rank}] CSV Usage: index.csv NOT found or failed to load", flush=True)

        # Build global index map using index.csv when available
        # Use LRU cache to limit number of open datasets in memory (prevents OOM)
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

        # Build sequential global indices (no shuffling)
        # Shuffling is handled by random_split() in main.py with a deterministic seed
        self.current_indices = np.arange(total_patterns)

        print(f"[Rank {rank}] CombinedDataset: {len(self.file_paths)} files, {total_patterns} total patterns", flush=True)
        print(f"[Rank {rank}]   Sequential indices (train/val split and rank sharding handled externally)", flush=True)

        if debug:
            print(f"[DEBUG Rank {rank}] Sequential indices (no shuffle in CombinedDataset)", flush=True)
            print(f"[DEBUG Rank {rank}] First 10 indices = {self.current_indices[:10].tolist()}", flush=True)
            print(f"[DEBUG Rank {rank}] Train/val split will be handled by random_split(), rank sharding by RankShardedSubset", flush=True)

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
        
        # Debug logging (only for first few calls per rank)
        if self.debug and self.debug_call_count < 15:  # Log first 15 calls (covers first few batches)
            file_info = self.file_info[file_idx]
            num_patterns_in_file = file_info['num_patterns']
            # Note: 'idx' parameter is the dataset index (could be split index or global index depending on mode)
            # 'global_idx' is the actual global index across all files
            print(f"[DEBUG Rank {self.rank}] CombinedDataset.__getitem__: "
                  f"DatasetIdx={idx}, GlobalIdx={global_idx}, "
                  f"File={file_path.name}, Pattern={local_idx}/{num_patterns_in_file}", flush=True)
            self.debug_call_count += 1
        
        # Lazy dataset creation with LRU cache: create dataset only when first accessed
        # This avoids opening all HDF5 files during initialization
        # LRU cache prevents memory accumulation by limiting number of open datasets
        if file_path not in self.dataset_cache:
            # Create new dataset
            lazy_kwargs = self.dataset_kwargs.copy()
            # Disable object caching for lazily created datasets to prevent memory accumulation
            # Objects will be loaded on-demand from HDF5 files instead
            dataset = PtychographyDataset(str(file_path), **lazy_kwargs)
            
            # Add to cache, removing oldest if cache is full (LRU eviction)
            if len(self.dataset_cache) >= self.max_cached_datasets:
                # Remove least recently used (first item in OrderedDict)
                _, oldest_dataset = self.dataset_cache.popitem(last=False)
                # Close HDF5 files if dataset has them open
                if hasattr(oldest_dataset, 'close'):
                    oldest_dataset.close()
            
            self.dataset_cache[file_path] = dataset
        else:
            # Move to end (most recently used) for LRU ordering
            dataset = self.dataset_cache.pop(file_path)
            self.dataset_cache[file_path] = dataset
        
        return dataset[local_idx]


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


"""
if __name__ == 'main':
    full_dataset = CombinedDataset(
    file_paths='/lus/flare/projects/PPFL_FM/simulated_data',
    rank=0,
    world_size=1,
    scale=10000.0,
    normalization_dict_path='/flare/datascience/vsastry/projects/ptcho_vit/ptycho-vit/normalization_fulldata.pkl',
    apply_noise=True,
    cache_object=True,
    max_probe_modes=10,
    debug=True
    )
"""


