"""Create normalization dict from paired *_dp/_para HDF5 files.

To run with multiple processes, use torchrun, e.g.:
  torchrun --nproc_per_node=4 scripts/make_normalization_dict.py /path/to/data_root
"""

import os
import numpy as np
from pathlib import Path
import h5py
import pickle
import argparse
try:
    import tqdm
except Exception:
    tqdm = None
try:
    import torch.distributed as dist
except Exception:
    dist = None

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
    if dist is None:
        return 0, 1
    if not dist.is_available():
        return 0, 1
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            # Use gloo to avoid NCCL GPU affinity issues for this CPU-bound script.
            dist.init_process_group(backend="gloo", init_method="env://")
            return dist.get_rank(), dist.get_world_size()
        except Exception as e:
            print(f"Warning: Failed to initialize torch.distributed ({e}). Falling back to single process.", flush=True)
            return 0, 1
    return 0, 1


rank, n_ranks = _init_dist()


def derive_object_name(file_path: Path, base_dir: Path) -> str:
    """Derive object name from file_path relative to base_dir, stripping _dp/_para suffix."""
    file_path = Path(file_path)
    base_dir = Path(base_dir)

    try:
        rel = file_path.resolve().relative_to(base_dir.resolve())
    except Exception:
        rel = None

    if rel is not None:
        rel_no_suffix = rel.with_suffix("")
        rel_name = rel_no_suffix.name
        if rel_name.endswith("_dp"):
            rel_no_suffix = rel_no_suffix.with_name(rel_name[:-3])
        elif rel_name.endswith("_para"):
            rel_no_suffix = rel_no_suffix.with_name(rel_name[:-5])
        return rel_no_suffix.as_posix()

    stem = file_path.stem
    if stem.endswith("_dp"):
        return stem[:-3]
    if stem.endswith("_para"):
        return stem[:-5]
    return stem

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
        object_name = derive_object_name(f, directory)
        print(f"  - {object_name}", flush=True)

    return paired_files

files = find_paired_files(data_path) # This list has only the DP files!

probe_shapes = []
norm_dict = {}
file_iter = files[rank::n_ranks]
if tqdm is not None:
    file_iter = tqdm.tqdm(file_iter, disable=rank != 0)
for file_path in file_iter:
    object_name = derive_object_name(file_path, Path(data_path))
    para_file = Path(file_path).with_name(f"{Path(file_path).stem[:-3]}_para.hdf5")
    with h5py.File(para_file, 'r') as f:
        probe = f['probe'][:]
    probe_shapes.append(probe.shape)
    with h5py.File(file_path, 'r') as f:
        data = f['dp'][:]
    max_intensity = np.max(data)
    norm_dict[object_name] = max_intensity

if n_ranks > 1 and dist is not None and dist.is_available() and dist.is_initialized():
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

if n_ranks > 1 and dist is not None and dist.is_available() and dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
