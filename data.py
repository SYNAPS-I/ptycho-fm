import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import h5py
import pickle
from pathlib import Path
from typing import Optional, Tuple
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

            # Cache probe if small enough
            probe_data = f['probe']
            if probe_data.nbytes < 100 * 1024 * 1024:  # Cache if < 100MB
                self._cached_probe = probe_data[...]

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
        """Load specific pattern from paired HDF5 files with efficient caching."""
        # Cache object data on first access
        self._cache_object_data()

        # Load diffraction pattern - open and close file for each access
        with h5py.File(self.dp_file, 'r') as dp_file:
            diffraction_pattern = dp_file['dp'][pattern_idx]
            diffraction_pattern = self.normalize(diffraction_pattern)
            # Add noise by sampling from a Poisson distribution
            diffraction_pattern = np.random.default_rng().poisson(diffraction_pattern).astype(np.float32)
            diffraction_amp = np.sqrt(diffraction_pattern)
        
        # Get probe position from cache
        probe_position = self._cached_probe_positions[pattern_idx]
        
        # Get probe from cache or file
        if self._cached_probe is not None:
            probe = self._cached_probe
        else:
            with h5py.File(self.para_file, 'r') as para_file:
                probe = para_file['probe'][...]
        
        # Get object data and extract patches
        if self._cached_object is not None:
            full_object = self._cached_object
        else:
            # If not cached, load on demand (for memory-constrained situations)
            with h5py.File(self.para_file, 'r') as para_file:
                full_object = para_file['object'][...]
        
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
        for f in paired_files:
            object_name = f.stem[:-3]
            print(f"  - {object_name}", flush=True)

        return paired_files

    def __init__(self, file_paths, **dataset_kwargs):
        # Auto-detect if file_paths is a directory or a list
        if isinstance(file_paths, (str, Path)):
            file_path = Path(file_paths)
            if file_path.is_dir():
                # Scan directory for paired files
                self.file_paths = self.find_paired_files(file_path)
            else:
                # Single file provided as string
                self.file_paths = [file_path]
        else:
            # List of files provided
            self.file_paths = [Path(f) for f in file_paths]

        self.dataset_kwargs = dataset_kwargs

        # Create a dataset for each file and track indices
        self.datasets = []
        self.file_info = []
        total_patterns = 0

        for file_path in self.file_paths:
            dataset = PtychographyDataset(str(file_path), **dataset_kwargs)
            num_patterns = len(dataset)

            self.datasets.append(dataset)
            self.file_info.append({
                'path': file_path,
                'num_patterns': num_patterns,
                'start_idx': total_patterns,
                'end_idx': total_patterns + num_patterns
            })
            total_patterns += num_patterns

        self.total_patterns = total_patterns

        print(f"CombinedDataset: {len(self.file_paths)} files, {total_patterns} total patterns", flush=True)

    def __len__(self) -> int:
        return self.total_patterns

    def __getitem__(self, idx: int):
        """
        Get a sample by global index.

        Maps the global index to the appropriate file and local index.
        """
        if idx >= self.total_patterns:
            raise IndexError(f"Index {idx} out of range for dataset with {self.total_patterns} patterns")

        # Find which file this index belongs to
        for i, file_info in enumerate(self.file_info):
            if file_info['start_idx'] <= idx < file_info['end_idx']:
                # Convert global index to local index within file
                local_idx = idx - file_info['start_idx']
                return self.datasets[i][local_idx]

        raise IndexError(f"Index {idx} not found in any dataset")