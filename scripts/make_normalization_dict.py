"""Create normalization dict from paired *_dp/_para HDF5 files.

To run with multiple processes, use torchrun, e.g.:
  torchrun --nproc_per_node=4 scripts/make_normalization_dict.py /path/to/data_root
"""

import os
import sys
import numpy as np
from pathlib import Path
import h5py
import pickle
import tqdm
import argparse
import torch
import torch.distributed as dist

# Add parent directory to path to import data module
sys.path.insert(0, str(Path(__file__).parent.parent))
from data import CombinedDataset

parser = argparse.ArgumentParser(
    description="Create normalization dict from paired *_dp/_para HDF5 files."
)
parser.add_argument(
    "data_path",
    help="Root directory to scan for paired *_dp/_para HDF5 files",
)
parser.add_argument(
    "--output-path",
    default="normalization.pkl",
    help="Output path (including filename) for the normalization pickle",
)
args = parser.parse_args()
data_path = args.data_path
output_path = Path(args.output_path)


def _init_dist():
    if not dist.is_available():
        return 0, 1
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Use gloo to avoid NCCL GPU affinity issues for this CPU-bound script.
        dist.init_process_group(backend="gloo", init_method="env://")
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


rank, n_ranks = _init_dist()

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

    # Find all _dp.hdf5 files (recursive)
    dp_files = list(directory.rglob('*_dp.hdf5'))

    # Verify each has a matching _para.hdf5 file
    paired_files = []
    for dp_file in sorted(dp_files):
        base_name = dp_file.stem[:-3]  # Remove '_dp' suffix
        para_file = dp_file.with_name(f"{base_name}_para.hdf5")

        if para_file.exists():
            paired_files.append(dp_file)
        else:
            print(f"Warning: Skipping {dp_file.name} - no matching {para_file.name}", flush=True)

    if len(paired_files) == 0:
        raise ValueError(f"No paired HDF5 files found in {directory}")

    print(f"Found {len(paired_files)} paired dataset(s) in {directory}", flush=True)
    for f in paired_files:
        object_name = CombinedDataset.derive_object_name(f, directory)
        print(f"  - {object_name}", flush=True)

    return paired_files

files = find_paired_files(data_path) # This list has only the DP files!

probe_shapes = []
norm_dict = {}
for file_path in tqdm.tqdm(files[rank::n_ranks], disable=rank != 0):
    object_name = CombinedDataset.derive_object_name(file_path, Path(data_path))
    para_file = Path(file_path).with_name(f"{Path(file_path).stem[:-3]}_para.hdf5")
    with h5py.File(para_file, 'r') as f:
        probe = f['probe'][:]
    probe_shapes.append(probe.shape)
    with h5py.File(file_path, 'r') as f:
        data = f['dp'][:]
    max_intensity = np.max(data)
    norm_dict[object_name] = max_intensity

if n_ranks > 1 and dist.is_available() and dist.is_initialized():
    gathered_dicts = [None for _ in range(n_ranks)]
    gathered_probe_shapes = [None for _ in range(n_ranks)]
    dist.all_gather_object(gathered_dicts, norm_dict)
    dist.all_gather_object(gathered_probe_shapes, probe_shapes)
else:
    gathered_dicts = [norm_dict]
    gathered_probe_shapes = [probe_shapes]

if rank == 0:
    merged_norm_dict = {}
    for d in gathered_dicts:
        merged_norm_dict.update(d)

    merged_probe_shapes = []
    for shapes in gathered_probe_shapes:
        merged_probe_shapes.extend(shapes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(merged_norm_dict, f)

    with open(output_path, 'rb') as f:
        test = pickle.load(f)
    for key, value in test.items():
        print(f"{key}: {value}")

    probe_shapes = np.array(merged_probe_shapes)
    print("Max. OPR probe modes (index 0):", np.max(probe_shapes[:, 0]))
    print("Max. incoherent modes (index 1):", np.max(probe_shapes[:, 1]))

if n_ranks > 1 and dist.is_available() and dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
