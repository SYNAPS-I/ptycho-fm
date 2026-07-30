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
from scipy.ndimage import zoom
from ptycho_vit.utils.ptychi_utils import extract_patches_fourier_shift


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
        default_normalization (float): Fallback normalization value when object not found (default: 100000.0)
        apply_noise (bool): Whether to simulate noise by sampling from a Poisson distribution (set to False for experimental data)
        cache_object (bool): Whether to cache object and probe data in memory
        max_probe_modes (int): Maximum number of probe modes to pad to (default: 8)
    """

    def __init__(
        self,
        file_path: str,
        scale: float = 100000.,
        normalization_dict_path: Optional[str] = None,
        default_normalization: float = 100000.0,
        apply_noise: bool = True,
        cache_object: bool = True,
        max_probe_modes: int = 8,
        target_size: Optional[int] = 256,
        object_name: Optional[str] = None
    ):
        self.file_path = Path(file_path)
        self.scale = scale
        self.normalization_dict_path = normalization_dict_path
        self.default_normalization = default_normalization
        self.apply_noise = apply_noise
        self.cache_object = cache_object
        self.max_probe_modes = max_probe_modes
        self.target_size = target_size  # Target size for diffraction patterns (e.g., 256)

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

        # Extract object name from file path unless provided
        # Assumes format: .../object_name/object_name_dp.hdf5
        if object_name is not None:
            self.object_name = object_name
        else:
            stem = self.dp_file.stem
            self.object_name = stem[:-3] if stem.endswith('_dp') else stem

        # Load normalization factor
        self._load_normalization()

    def _load_normalization(self):
        """
        Load normalization factor from pickle file or use default.

        If normalization_dict_path is provided, loads the dictionary and looks up
        the normalization factor using self.object_name as the key.
        Falls back to a default value if key not found or file not provided.
        """
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
                          f"Using default: {self.default_normalization}", flush=True)
                    self.normalization = self.default_normalization

            except FileNotFoundError:
                print(f"Warning: Normalization file not found at {self.normalization_dict_path}. "
                      f"Using default: {self.default_normalization}", flush=True)
                self.normalization = self.default_normalization
            except Exception as e:
                print(f"Warning: Error loading normalization file: {e}. "
                      f"Using default: {self.default_normalization}", flush=True)
                self.normalization = self.default_normalization
        else:
            # No normalization dict provided, use default
            self.normalization = self.default_normalization

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
        self._raw_pattern_shape = self.dp_handle['dp'].shape[1:]  # Original shape from file

        # Set pattern_shape to target_size if specified, otherwise use raw shape
        if self.target_size is not None:
            self.pattern_shape = (self.target_size, self.target_size)
        else:
            self.pattern_shape = self._raw_pattern_shape

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
        """Extract patch from full object at given probe position.

        Always extracts at the raw pattern size (from file), not the target size.
        Padding to target size is handled separately after extraction.
        """
        # Use raw pattern shape for extraction (e.g., 128x128), not target size (e.g., 256x256)
        return extract_patches_fourier_shift(torch.from_numpy(full_object), probe_position.unsqueeze(0), (self._raw_pattern_shape[0], self._raw_pattern_shape[1]))[0]

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

    def _normalize_probe_shape(self, probe: np.ndarray) -> np.ndarray:
        """
        Normalize probe array to shape (1, N, H, W).

        Accepted input shapes:
            - (1, N, H, W): already normalized
            - (A, N, H, W): drop to first A slice
        """
        if probe.ndim != 4:
            raise ValueError(f"Expected 4D probe array (A, N, H, W), got {probe.shape}")
        if probe.shape[0] == 1:
            return probe
        return probe[:1, ...]

    def _zero_pad_to_target(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """
        Zero-pad a 2D image to target size, keeping the original centered.

        Args:
            image: 2D array of shape (H, W)
            target_size: Target size for both dimensions

        Returns:
            Zero-padded array of shape (target_size, target_size)
        """
        h, w = image.shape
        if h == target_size and w == target_size:
            return image

        if h > target_size or w > target_size:
            raise ValueError(f"Image size ({h}, {w}) larger than target size ({target_size})")

        # Calculate padding for each side
        pad_h = target_size - h
        pad_w = target_size - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Zero-pad
        padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        return padded

    def _upsample_probe(self, probe: np.ndarray, target_size: int) -> np.ndarray:
        """
        Upsample probe to target spatial size using bilinear interpolation.

        Args:
            probe: Probe array with shape (1, N, H, W) - complex values
            target_size: Target spatial size

        Returns:
            Upsampled probe array with shape (1, N, target_size, target_size)
        """
        _, n_modes, h, w = probe.shape
        if h == target_size and w == target_size:
            return probe

        # Calculate zoom factors for spatial dimensions
        zoom_h = target_size / h
        zoom_w = target_size / w

        # Upsample real and imaginary parts separately for each mode
        upsampled_real = np.zeros((1, n_modes, target_size, target_size), dtype=np.float64)
        upsampled_imag = np.zeros((1, n_modes, target_size, target_size), dtype=np.float64)

        for mode_idx in range(n_modes):
            mode_data = probe[0, mode_idx]  # (H, W) complex
            # Use scipy zoom for upsampling (order=1 for bilinear)
            upsampled_real[0, mode_idx] = zoom(mode_data.real, (zoom_h, zoom_w), order=1)
            upsampled_imag[0, mode_idx] = zoom(mode_data.imag, (zoom_h, zoom_w), order=1)

        # Recombine into complex
        upsampled_probe = upsampled_real + 1j * upsampled_imag
        return upsampled_probe.astype(probe.dtype)

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

        # Load raw positions (labeled _m but may already be in pixels)
        pos_y = self.para_handle['probe_position_y_m'][...]
        pos_x = self.para_handle['probe_position_x_m'][...]
        positions_raw = np.column_stack([pos_y, pos_x])

        # Auto-detect if positions are in meters or already in pixels
        # If position range is similar to object size, assume pixels
        # If position range is tiny (sub-millimeter), assume meters
        pos_range_y = pos_y.max() - pos_y.min()
        pos_range_x = pos_x.max() - pos_x.min()

        # Heuristic: if ranges are within 2x of object dimensions, positions are likely in pixels
        # Object dimensions are typically 100s-1000s of pixels
        obj_h, obj_w = self.object_shape
        positions_likely_pixels = (
            0.1 * obj_h < pos_range_y < 10 * obj_h and
            0.1 * obj_w < pos_range_x < 10 * obj_w
        )

        if positions_likely_pixels:
            # Positions are already in pixels, no conversion needed
            self._cached_probe_positions = torch.from_numpy(positions_raw.astype(np.float32))
        else:
            # Positions are in meters, convert to pixels
            self._cached_probe_positions = torch.from_numpy((positions_raw / self.pixel_size_m).astype(np.float32))

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
            probe = self._normalize_probe_shape(probe)
            # Pad probe to (1, max_probe_modes, H, W) if needed
            probe = self._pad_probe(probe, target_modes=self.max_probe_modes)
            # Upsample probe to target size if needed
            if self.target_size is not None:
                probe = self._upsample_probe(probe, self.target_size)
            self._cached_probe = probe 
    
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

        # Zero-pad diffraction pattern to target size if needed
        if self.target_size is not None and diffraction_amp.shape[0] != self.target_size:
            diffraction_amp = self._zero_pad_to_target(diffraction_amp, self.target_size)

        # Get probe position from cache
        probe_position = self._cached_probe_positions[pattern_idx]

        # Get probe from cache or file
        if self._cached_probe is not None:
            probe = self._cached_probe
        else:
            probe = para_handle['probe'][...]
            probe = self._normalize_probe_shape(probe)
            # Pad probe to (1, max_probe_modes, H, W) if needed
            probe = self._pad_probe(probe, target_modes=self.max_probe_modes)
            # Upsample probe to target size if needed
            if self.target_size is not None:
                probe = self._upsample_probe(probe, self.target_size)
        
        # Get object data and extract patches
        if self._cached_object is not None:
            full_object = self._cached_object
        else:
            # If not cached, load on demand (for memory-constrained situations)
            full_object = para_handle['object'][0]

        # Extract patches at probe position (at raw pattern size, e.g., 128x128)
        patch = self._extract_patch(full_object, probe_position)
        amplitude_patch = torch.abs(patch)
        phase_patch = torch.angle(patch)

        # Zero-pad amplitude and phase patches to target size if needed
        if self.target_size is not None and amplitude_patch.shape[0] != self.target_size:
            amplitude_patch = torch.from_numpy(
                self._zero_pad_to_target(amplitude_patch.detach().cpu().numpy(), self.target_size)
            )
            phase_patch = torch.from_numpy(
                self._zero_pad_to_target(phase_patch.detach().cpu().numpy(), self.target_size)
            )

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
    def derive_relative_path(file_path: Path, base_dir: Optional[Path]) -> Optional[Path]:
        """Return file_path relative to base_dir when possible."""
        if base_dir is None:
            return None
        try:
            return Path(file_path).resolve().relative_to(Path(base_dir).resolve())
        except Exception:
            return None

    @staticmethod
    def derive_object_name(file_path: Path, base_dir: Optional[Path]) -> str:
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
        return stem[:-3] if stem.endswith('_dp') else stem

    def __init__(self, file_paths, rank=0, world_size=1, debug=False, max_files=None, **dataset_kwargs):
        """
        Initialize CombinedDataset with sequential global indexing.

        Args:
            file_paths: Directory path to scan for all paired files
            rank: Rank of current process (default: 0)
            world_size: Total number of processes (default: 1)
            debug: If True, enable debug logging for CSV usage and data access (default: False)
            max_files: Maximum number of files to use (default: None = use all)
            **dataset_kwargs: Additional arguments passed to PtychographyDataset
        """
        # file_paths must be a directory
        data_dir = Path(file_paths)
        if not data_dir.is_dir():
            raise ValueError(f"file_paths must be a directory, got: {file_paths}")

        # Scan directory for all paired files
        self.file_paths = self.find_paired_files(data_dir)

        # Limit number of files if max_files is specified
        if max_files is not None and max_files < len(self.file_paths):
            print(f"[Rank {rank}] Limiting to {max_files} files (out of {len(self.file_paths)} available)", flush=True)
            self.file_paths = self.file_paths[:max_files]

        self.data_dir = data_dir
        self.file_object_names: Dict[Path, str] = {}
        for file_path in self.file_paths:
            object_name = self.derive_object_name(file_path, self.data_dir)
            self.file_object_names[file_path] = object_name

        self.dataset_kwargs = dataset_kwargs
        self.rank = rank
        self.world_size = world_size
        self.debug = debug
        self.debug_call_count = 0  # Track number of __getitem__ calls for debug logging

        # dummy: set to below tensor if needed
        self.fake_data = None
        # (
        #     torch.randn([1, 256, 256], dtype=torch.float32),
        #     torch.randn([1, 256, 256], dtype=torch.float32),
        #     torch.randn([1, 256, 256], dtype=torch.float32),
        #     torch.randn([1, 10, 256, 256], dtype=torch.complex64),
        #     torch.randn([2], dtype=torch.float64),
        #     torch.randn([1], dtype=torch.float32),
        #     torch.randn([1], dtype=torch.float32),
        # )

        # Try to load index.csv to avoid opening all HDF5 files
        self.index_df = None
        self.pattern_counts: Dict[Path, int] = {}
        
        if self.data_dir is not None:
            index_csv_path = Path(self.data_dir) / 'index.csv'
            if index_csv_path.exists():
                try:
                    print(f"[Rank {rank}] Loading index.csv to avoid opening all HDF5 files...", flush=True)
                    self.index_df = pd.read_csv(index_csv_path)
                    
                    # Create a fast lookup: map relative paths to n_dps
                    csv_lookup = {}
                    for _, row in self.index_df.iterrows():
                        csv_path = Path(row['dp_path'])
                        if csv_path.is_absolute():
                            raise ValueError(
                                f"index.csv dp_path must be relative to data_dir, got absolute path: {csv_path}"
                            )
                        n_dps = int(row['n_dps'])
                        full_path = self.data_dir / csv_path
                        if not full_path.exists():
                            raise FileNotFoundError(
                                f"index.csv dp_path not found under data_dir: {csv_path}"
                            )
                        rel_path = self.derive_relative_path(full_path, self.data_dir)
                        if rel_path is None:
                            raise ValueError(
                                f"Unable to resolve dp_path relative to data_dir: {csv_path}"
                            )
                        csv_lookup[rel_path] = n_dps
                    
                    # Match our file_paths to CSV entries
                    for file_path in self.file_paths:
                        rel = self.derive_relative_path(file_path, self.data_dir)
                        if rel in csv_lookup:
                            self.pattern_counts[file_path] = csv_lookup[rel]
                    
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
                dataset = PtychographyDataset(
                    str(file_path),
                    object_name=self.file_object_names.get(file_path),
                    **fallback_kwargs
                )
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
        if self.fake_data:
            return self.fake_data

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
            dataset = PtychographyDataset(
                str(file_path),
                object_name=self.file_object_names.get(file_path),
                **lazy_kwargs
            )
            
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
