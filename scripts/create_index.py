import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

MINIMAL_COLUMNS = ["dp_path", "n_dps"]
FULL_COLUMNS = [
    "dp_path",
    "dp_height",
    "dp_width",
    "n_dps",
    "n_opr_modes",
    "n_incoherent_modes",
    "probe_height",
    "probe_width",
    "pixel_size_m",
    "object_height",
    "object_width",
]


def _find_dp_files(data_root: Path, pattern: str | None) -> list[Path]:
    if pattern is not None:
        return sorted(path for path in data_root.rglob(pattern) if path.is_file())

    return sorted(
        {
            *data_root.rglob("*_dp.hdf5"),
            *data_root.rglob("*_dp.h5"),
        }
    )


def _para_path(dp_path: Path) -> Path:
    return dp_path.with_name(
        f"{dp_path.stem.removesuffix('_dp')}_para{dp_path.suffix}"
    )


def _full_record(dp_path: Path, para_path: Path, relative_path: str) -> dict:
    with h5py.File(dp_path, "r") as f:
        n_dps, dp_height, dp_width = f["dp"].shape

    with h5py.File(para_path, "r") as f:
        probe = f["probe"]
        if probe.ndim == 4:
            n_opr_modes, n_incoherent_modes, probe_height, probe_width = probe.shape
        elif probe.ndim == 3:
            n_opr_modes = 1
            n_incoherent_modes, probe_height, probe_width = probe.shape
        else:
            raise ValueError(f"Expected a 3D or 4D probe, got shape {probe.shape}")

        object_dataset = f["object"]
        pixel_size_m = object_dataset.attrs["pixel_height_m"]
        object_magnitude = np.abs(object_dataset[...])
        if np.ptp(object_magnitude) > 1e-2:
            object_height, object_width = object_magnitude.shape[-2:]
        else:
            object_height, object_width = None, None

    return {
        "dp_path": relative_path,
        "dp_height": dp_height,
        "dp_width": dp_width,
        "n_dps": n_dps,
        "n_opr_modes": n_opr_modes,
        "n_incoherent_modes": n_incoherent_modes,
        "probe_height": probe_height,
        "probe_width": probe_width,
        "pixel_size_m": pixel_size_m,
        "object_height": object_height,
        "object_width": object_width,
    }


def create_index(
    data_root: str | Path,
    output_path: str | Path | None = None,
    *,
    index_type: str = "full",
    pattern: str | None = None,
) -> pd.DataFrame:
    """Create a minimal training index or a full HDF5 dataset catalog."""
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise ValueError(f"Not a directory: {data_root}")
    if index_type not in {"minimal", "full"}:
        raise ValueError("index_type must be 'minimal' or 'full'")

    output_path = (
        Path(output_path) if output_path is not None else data_root / "index.csv"
    )
    dp_files = _find_dp_files(data_root, pattern)
    records = []

    for dp_path in tqdm(dp_files, desc=f"Building {index_type} index"):
        para_path = _para_path(dp_path)
        if not para_path.is_file():
            logger.warning(
                "Skipping %s: matching file %s does not exist", dp_path, para_path
            )
            continue

        relative_path = dp_path.relative_to(data_root).as_posix()
        try:
            if index_type == "minimal":
                with h5py.File(dp_path, "r") as f:
                    n_dps = f["dp"].shape[0]
                records.append({"dp_path": relative_path, "n_dps": n_dps})
            else:
                records.append(_full_record(dp_path, para_path, relative_path))
        except (OSError, KeyError, ValueError) as exc:
            logger.error("Error processing %s: %s", dp_path, exc)

    columns = MINIMAL_COLUMNS if index_type == "minimal" else FULL_COLUMNS
    table = pd.DataFrame(records, columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(
        f"Created {index_type} index at {output_path}: "
        f"{len(table)} files, {table['n_dps'].sum():,} patterns"
    )
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a minimal training index or full HDF5 dataset catalog."
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root directory containing paired diffraction and parameter files",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output CSV path (default: <data_root>/index.csv)",
    )
    parser.add_argument(
        "--index-type",
        choices=("minimal", "full"),
        default="full",
        help="Use 'minimal' for dp_path/n_dps or 'full' for dataset metadata",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional recursive glob pattern (default: *_dp.hdf5 and *_dp.h5)",
    )
    args = parser.parse_args(argv)

    create_index(
        args.data_root,
        args.output_path,
        index_type=args.index_type,
        pattern=args.pattern,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
