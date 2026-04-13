#!/usr/bin/env python3
"""
Create index.csv for fast dataset initialization.
This avoids opening all HDF5 files every time training starts.
"""

from __future__ import annotations

import argparse
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def create_index_csv(
    data_dir: Path,
    output_path: Path | None = None,
    pattern: str = "*_dp.hdf5",
) -> pd.DataFrame:
    """
    Create index.csv containing dp_path and n_dps for each file.

    Args:
        data_dir: Directory containing *_dp.hdf5 files
        output_path: Output path for index.csv (default: data_dir/index.csv)
    """
    data_dir = Path(data_dir)
    if output_path is None:
        output_path = data_dir / 'index.csv'

    # Find all dp files
    dp_files = sorted(data_dir.glob(pattern))
    print(f"Found {len(dp_files)} files matching {pattern} in {data_dir}")

    records = []
    for dp_file in tqdm(dp_files, desc="Reading HDF5 files"):
        try:
            with h5py.File(dp_file, 'r') as f:
                n_dps = f['dp'].shape[0]
            records.append({
                'dp_path': str(dp_file),
                'n_dps': n_dps
            })
        except Exception as e:
            print(f"Warning: Failed to read {dp_file}: {e}")

    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    print(f"\nCreated {output_path}")
    print(f"  Total files: {len(df)}")
    print(f"  Total patterns: {df['n_dps'].sum():,}")

    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create index.csv for faster CombinedDataset startup.")
    parser.add_argument("data_dir", help="Directory containing *_dp.hdf5 files")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <data_dir>/index.csv)")
    parser.add_argument("--pattern", default="*_dp.hdf5", help="Glob pattern for dp files")
    args = parser.parse_args(argv)

    output_path = Path(args.output) if args.output else None
    create_index_csv(Path(args.data_dir), output_path=output_path, pattern=args.pattern)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
