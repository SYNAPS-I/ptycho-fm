import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import h5py
import pickle
from pathlib import Path
from typing import Optional, Tuple


class PtychographyDataset(Dataset):
    """
    PyTorch Dataset for ptychography data.

    For HDF5 files: Expects paired files in the same directory:
    - *_dp.hdf5: Contains diffraction patterns
    - *_para.hdf5: Contains probe positions, object amplitude/phase, and probe information

    Args:
        file_path (str): Path to data file or corresponding parameters file(*_dp.hdf5 or *_para.hdf5)
        patch_size (int): Size of patches to extract from the full object in pixels
        scale (float): Factor by which to scale all diffraction intensity to
        cache_object (bool): Whether to cache object data in memory
        normalization_dict_path (str): Path to .pkl file containing dict of {object_name: normalization_factor}
    """

    def __init__(
        self,
        file_path: str,
        patch_size: int = 512,
        scale: float = 100000., 
        cache_object: bool = True,
        normalization_dict_path: Optional[str] = None
    ):
        self.file_path = Path(file_path)
        self.patch_size = patch_size
        self.scale = scale
        self.cache_object = cache_object
        self.normalization_dict_path = normalization_dict_path

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
            required_keys = ['object']
            missing_keys = [key for key in required_keys if key not in f.keys()]
            if missing_keys:
                raise KeyError(f"Missing required keys in {self.para_file.name}: {missing_keys}")
            
            self.object_shape = f['object'][0].shape
            
            # Check if probe data exists
            self.has_probe = 'probe' in f.keys()
        
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
        return extract_patches_fourier_shift(torch.from_numpy(full_object), probe_position.unsqueeze(0), (self.patch_size, self.patch_size))[0]
    
    def _cache_object_data(self):
        """Cache object and probe position data for efficient access."""
        if self._cached_object is not None:
            return
            
        with h5py.File(self.para_file, 'r') as f:
            # Cache full object
            if self.cache_object:
                self._cached_object = f['object'][0]
            
            # Cache probe if available and small enough
            if self.has_probe:
                probe_data = f['probe']
                if probe_data.nbytes < 100 * 1024 * 1024:  # Cache if < 100MB
                    self._cached_probe = probe_data[...]

        # Cache probe positions (small)
        if self.patch_size == 512:
            fwhm = 98
        elif self.patch_size == 256:
            fwhm = 49
        self._cached_probe_positions = create_positions(self.object_shape, self.pattern_shape, target_overlap=0.8, fwhm=fwhm)
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
        elif self.has_probe:
            with h5py.File(self.para_file, 'r') as para_file:
                probe = para_file['probe'][...]
        else:
            probe = None
        
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


def batch_slice(image: Tensor, sy: Tensor, sx: Tensor, patch_size: Tuple[int, int]) -> Tensor:
    """
    Slice patches from an image at given window positions. The patch size is determined
    from the starting and ending coordinates in each direction, and is assumed to be
    the same for all patches. From Ming Du's pty-chi:
    https://github.com/AdvancedPhotonSource/pty-chi 
    
    Parameters
    ----------
    image : Tensor
        A (H, W) tensor of the image.
    sy : Tensor
        A (N,) tensor of integers giving the starting y-coordinates of the patches.
    sx : Tensor
        A (N,) tensor of integers giving the starting x-coordinates of the patches.
    patch_size : tuple of int
        A tuple giving the patch shape in pixels.

    Returns
    -------
    Tensor
        A tensor of shape (N, h, w) containing the extracted patches.
    """
    h, w = image.shape[-2:]
    if (
        sy.min() < 0 
        or sy.max() + patch_size[0] > image.shape[-2] 
        or sx.min() < 0 
        or sx.max() + patch_size[1] > image.shape[-1]
    ):
        raise ValueError("Patch indices are out of bounds.")
    
    x = torch.arange(patch_size[1], device=sx.device)[None, :]
    y = torch.arange(patch_size[0], device=sy.device)[None, :]
    x = x.expand(len(sx), x.shape[1])
    y = y.expand(len(sy), y.shape[1])
    x = x + sx[:, None]
    y = y + sy[:, None]
    inds = (y * w).unsqueeze(-1) + x.unsqueeze(1)
    patches = image.view(-1)[inds.view(-1)]
    patches = patches.reshape(len(sy), patch_size[0], patch_size[1])
    return patches


def fourier_shift(images: Tensor, shifts: Tensor, strictly_preserve_zeros: bool = False) -> Tensor:
    """
    Apply Fourier shift to a batch of images. From Ming Du's pty-chi:
    https://github.com/AdvancedPhotonSource/pty-chi 

    Parameters
    ----------
    images : Tensor
        A [N, H, W] tensor of images.
    shifts : Tensor
        A [N, 2] tensor of shifts in pixels.
    strictly_preserve_zeros : bool
        If True, mask of strictly zero pixels will be generated and shifted
        by the same amount. Pixels that have a non-zero value in the shifted
        mask will be set to zero in the shifted image. This preserves the zero
        pixels in the original image, preventing FFT from introducing small
        non-zero values due to machine precision.

    Returns
    -------
    Tensor
        Shifted images.
    """
    if strictly_preserve_zeros:
        zero_mask = images == 0
        zero_mask = zero_mask.float()
        zero_mask_shifted = fourier_shift(zero_mask, shifts, strictly_preserve_zeros=False)
    # This version intended for torch.complex64 images only, though it inherits type from image
    ft_images = torch.fft.fft2(images.type(torch.complex128), norm=None).type(torch.complex64)
    freq_y, freq_x = torch.meshgrid(
        torch.fft.fftfreq(images.shape[-2]), torch.fft.fftfreq(images.shape[-1]), indexing="ij"
    )
    freq_x = freq_x.to(ft_images.device)
    freq_y = freq_y.to(ft_images.device)
    freq_x = freq_x.repeat(images.shape[0], 1, 1)
    freq_y = freq_y.repeat(images.shape[0], 1, 1)
    mult = torch.exp(
        1j
        * -2
        * torch.pi
        * (freq_x * shifts[:, 1].view(-1, 1, 1) + freq_y * shifts[:, 0].view(-1, 1, 1))
    )
    ft_images = ft_images * mult
    # Complex images (pty-chi original version supports real datatypes with higher precision)
    shifted_images = torch.fft.ifft2(ft_images.type(torch.complex128), norm=None).type(torch.complex64) 
    if not images.dtype.is_complex:
        shifted_images = shifted_images.real
    if strictly_preserve_zeros:
        shifted_images[zero_mask_shifted > 0] = 0
    return shifted_images


def extract_patches_fourier_shift(
    image: Tensor, positions: Tensor, shape: Tuple[int, int], pad: Optional[int] = 1
) -> Tensor:
    """
    Extract patches from 2D object. If a patch's footprint goes outside the image,
    the image is padded with zeros to account for the missing pixels. From Ming Du's pty-chi:
    https://github.com/AdvancedPhotonSource/pty-chi 

    Parameters
    ----------
    image : Tensor
        The whole image.
    positions : Tensor
        A tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    shape : tuple of int
        A tuple giving the patch shape in pixels.
    pad : Optional[int]
        If given, patches with larger size than the intended size by this amount are cropped
        out from the patches before shifting.

    Returns
    -------
    Tensor
        A tensor of shape (N, H, W) containing the extracted patches.
    """
    # Floating point ranges over which interpolations should be done
    sys_float = positions[:, 0] - (shape[0] - 1.0) / 2.0
    sxs_float = positions[:, 1] - (shape[1] - 1.0) / 2.0

    # Crop one more pixel each side for Fourier shift
    sys = sys_float.floor().int() - pad
    eys = sys + shape[0] + 2 * pad
    sxs = sxs_float.floor().int() - pad
    exs = sxs + shape[1] + 2 * pad

    fractional_shifts = torch.stack([sys_float - sys - pad, sxs_float - sxs - pad], -1)

    pad_lengths = [
        max(-sxs.min(), 0),
        max(exs.max() - image.shape[1], 0),
        max(-sys.min(), 0),
        max(eys.max() - image.shape[0], 0),
    ]
    image = torch.nn.functional.pad(image, pad_lengths)
    sys = sys + pad_lengths[2]
    eys = eys + pad_lengths[2]
    sxs = sxs + pad_lengths[0]
    exs = exs + pad_lengths[0]

    patches = batch_slice(image, sys, sxs, patch_size=[shape[i] + 2 * pad for i in range(2)])

    # Apply Fourier shift to account for fractional shifts
    if not torch.allclose(fractional_shifts, torch.zeros_like(fractional_shifts), atol=1e-7):
        patches = fourier_shift(patches, -fractional_shifts)
    patches = patches[:, pad : patches.shape[-2] - pad, pad : patches.shape[-1] - pad]
    return patches


def create_positions(object_shape, probe_lateral_shape, target_overlap=0.8, fwhm=98):
    """Create probe positions in pixels. Adapted from Ming Du's ptycho_simulation_factory:
    https://github.com/mdw771/ptycho_simulation_factory

    This version removes the choice to pre-define number of positions or spacing. 
    Instead, the spacing is calculated by overlap and probe size.
    
    Parameters
    ----------
    object_shape : tuple of int
        Lateral shape of the object.
    probe_lateral_shape : tuple of int
        Lateral shape of the probe. This is used to determine the safety margin,
        so that the probe does not reach outside the object.
    target_overlap : float
        Overlap ratio
    fwhm : int
        Full width at half maximum of the probe in pixels (approximate is fine here)
    """
    spacing = (1 - target_overlap) * fwhm # Spacing is now enforced to be the same in y and x

    margin = [probe_lateral_shape[i] // 2 for i in range(len(probe_lateral_shape))]
    y = np.arange(margin[0], object_shape[0] - margin[0] - 1, spacing)
    x = np.arange(margin[1], object_shape[1] - margin[1] - 1, spacing)

    y, x = np.meshgrid(y, x)
    positions = np.stack([y.reshape(-1), x.reshape(-1)], axis=1)
    positions = positions - positions.mean(0) # Center is (0, 0)
    return torch.from_numpy(positions)
