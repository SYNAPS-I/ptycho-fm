"""Utility functions for generating dummy test data."""
from pathlib import Path
import h5py
import numpy as np

from ptycho_vit.utils.ptychi_utils import create_positions


def create_dummy_hdf5_pair(output_dir, object_name, num_patterns=None, pattern_size=128,
                          object_size=512, target_overlap=0.8, fwhm=98):
    """
    Create a pair of dummy HDF5 files (_dp.hdf5 and _para.hdf5) for testing.

    Args:
        output_dir: Directory to save files
        object_name: Name of the object (e.g., 'test_object_1')
        num_patterns: Number of diffraction patterns (if None, auto-calculated from positions)
        pattern_size: Size of each diffraction pattern (pattern_size x pattern_size)
        object_size: Size of the object (object_size x object_size)
        target_overlap: Overlap ratio for probe positions (default 0.8)
        fwhm: Full width at half maximum of probe (default 98)

    Returns:
        tuple: (dp_file_path, para_file_path, actual_num_patterns)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dp_file = output_dir / f"{object_name}_dp.hdf5"
    para_file = output_dir / f"{object_name}_para.hdf5"

    # Calculate expected number of patterns based on create_positions logic
    # This matches what PtychographyDataset will expect
    positions = create_positions(
        object_shape=(object_size, object_size),
        probe_lateral_shape=(pattern_size, pattern_size),
        target_overlap=target_overlap,
        fwhm=fwhm
    )
    actual_num_patterns = positions.shape[0]

    if num_patterns is not None and num_patterns != actual_num_patterns:
        print(f"Warning: Requested {num_patterns} patterns, but using {actual_num_patterns} "
              f"to match create_positions calculation")

    # Create diffraction patterns file
    with h5py.File(dp_file, 'w') as f:
        # Random diffraction patterns (simulating intensity data)
        diffraction_patterns = np.random.rand(actual_num_patterns, pattern_size, pattern_size).astype(np.float32) * 1000
        f.create_dataset('dp', data=diffraction_patterns)

    # Create parameters file
    with h5py.File(para_file, 'w') as f:
        # Create a complex object (amplitude * exp(i * phase))
        amplitude = np.random.rand(object_size, object_size).astype(np.float32)
        phase = np.random.rand(object_size, object_size).astype(np.float32) * 2 * np.pi - np.pi
        complex_object = amplitude * np.exp(1j * phase)

        # Save as single element array (matches expected format)
        f.create_dataset('object', data=complex_object[np.newaxis, ...])

        # Create probe (8 modes of pattern_size x pattern_size, complex)
        probe_amplitude = np.random.rand(8, pattern_size, pattern_size).astype(np.float32)
        probe_phase = np.random.rand(8, pattern_size, pattern_size).astype(np.float32) * 2 * np.pi - np.pi
        probe = np.stack([
            probe_amplitude * np.cos(probe_phase),
            probe_amplitude * np.sin(probe_phase)
        ], axis=-1).astype(np.float32)

        f.create_dataset('probe', data=probe)

    print(f"Created dummy HDF5 pair: {dp_file.name} and {para_file.name} with {actual_num_patterns} patterns")
    return dp_file, para_file, actual_num_patterns


def cleanup_test_files(test_dir):
    """Remove all test files in a directory."""
    test_dir = Path(test_dir)
    if test_dir.exists():
        for file in test_dir.glob('*.hdf5'):
            file.unlink()
        print(f"Cleaned up test files in {test_dir}")
