#!/usr/bin/env python3
"""
Create normalization dictionary for fine-tuning data.
Normalizes each scan by its maximum photon count.
"""

from __future__ import annotations

import argparse
import h5py
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_max_photons(dp_file: Path, chunk_size: int = 100) -> float:
    """Get the maximum photon count from a diffraction pattern file."""
    with h5py.File(dp_file, 'r') as f:
        if 'dp' not in f:
            raise KeyError(f"No 'dp' dataset in {dp_file}")
        # Get max across all patterns
        # For large files, we can't load all at once, so iterate through chunks
        dp_dataset = f['dp']
        max_val = 0.0

        # Process in chunks to avoid memory issues
        n_patterns = dp_dataset.shape[0]

        for start in range(0, n_patterns, chunk_size):
            end = min(start + chunk_size, n_patterns)
            chunk = dp_dataset[start:end]
            chunk_max = float(np.max(chunk))
            if chunk_max > max_val:
                max_val = chunk_max

        return max_val

def create_normalization_dict(
    data_dir: Path,
    output_file: Path | None = None,
    pattern: str = "*_dp.hdf5",
    chunk_size: int = 100,
) -> dict[str, float]:
    data_dir = Path(data_dir)
    if output_file is None:
        output_file = data_dir / "normalization.pkl"

    # Find all _dp.hdf5 files (these are symlinks)
    dp_files = sorted(data_dir.glob(pattern))

    if len(dp_files) == 0:
        raise FileNotFoundError(f"No files matching {pattern} found in {data_dir}")

    print(f"Scanning directory: {data_dir}")
    print(f"Found {len(dp_files)} diffraction pattern files")

    normalization_dict = {}

    for dp_file in tqdm(dp_files, desc="Computing max photons"):
        # Extract scan name (e.g., "S02252" from "S02252_dp.hdf5")
        scan_name = dp_file.stem[:-3] if dp_file.stem.endswith("_dp") else dp_file.stem

        try:
            max_photons = get_max_photons(dp_file, chunk_size=chunk_size)
            normalization_dict[scan_name] = max_photons
        except Exception as e:
            print(f"\nWARNING: Failed to process {dp_file.name}: {e}")
            continue

    # Save to pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(normalization_dict, f)

    print(f"\nSaved normalization dictionary to: {output_file}")
    print(f"Total scans: {len(normalization_dict)}")

    # Print some statistics
    values = list(normalization_dict.values())
    print(f"\nNormalization statistics:")
    print(f"  Min:    {min(values):.2f}")
    print(f"  Max:    {max(values):.2f}")
    print(f"  Mean:   {np.mean(values):.2f}")
    print(f"  Median: {np.median(values):.2f}")

    # Print a few examples
    print(f"\nFirst 5 entries:")
    for i, (name, val) in enumerate(list(normalization_dict.items())[:5]):
        print(f"  {name}: {val:.2f}")

    return normalization_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create normalization dict {scan_name: max_photons}.")
    parser.add_argument("data_dir", help="Directory containing *_dp.hdf5 files")
    parser.add_argument("--output", default=None, help="Output pickle path (default: <data_dir>/normalization.pkl)")
    parser.add_argument("--pattern", default="*_dp.hdf5", help="Glob pattern for dp files")
    parser.add_argument("--chunk-size", type=int, default=100, help="Chunk size when scanning HDF5 datasets")
    args = parser.parse_args(argv)

    output_file = Path(args.output) if args.output else None
    create_normalization_dict(
        Path(args.data_dir),
        output_file=output_file,
        pattern=args.pattern,
        chunk_size=args.chunk_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
