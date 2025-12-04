import os
import numpy as np
from pathlib import Path
import h5py
import pickle
from data import PtychographyDataset

data_path = '/scratch/aileenluo/ptycho-vit/data'

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

files = find_paired_files(data_path) # This list has only the DP files!

probe_shapes = []
norm_dict = {}
for file_path in files:
    dataset = PtychographyDataset(file_path)
    object_name = dataset.object_name
    with h5py.File(os.path.join(data_path, object_name + '_para.hdf5'), 'r') as f:
        probe = f['probe'][:]
    probe_shapes.append(probe.shape)
    with h5py.File(file_path, 'r') as f:
        data = f['dp'][:]
    max_intensity = np.max(data)
    norm_dict[object_name] = max_intensity

filename = "normalization.pkl"
with open(filename, 'wb') as f:
    pickle.dump(norm_dict, f)

with open(filename, 'rb') as f:
    test = pickle.load(f)
for key, value in test.items():
    print(f"{key}: {value}")

probe_shapes = np.array(probe_shapes)
print("Max. OPR probe modes (index 0):", np.max(probe_shapes[:, 0]))
print("Max. incoherent modes (index 1):", np.max(probe_shapes[:, 1]))