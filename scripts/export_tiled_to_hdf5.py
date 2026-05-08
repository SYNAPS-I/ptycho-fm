"""Export a holoptycho Tiled run to the paired HDF5 format ptycho-vit reads.

Use case: training on Polaris (or any compute environment without outbound
internet from compute nodes). Run this on a login node — which can reach
Tiled — to materialize the run as ``<object_name>_dp.hdf5`` +
``<object_name>_para.hdf5`` on the parallel filesystem. Compute-node training
jobs then read those files via the existing ``PtychographyDataset`` /
``CombinedDataset`` paths, no Tiled access required.

Usage::

    export TILED_API_KEY=...
    python scripts/export_tiled_to_hdf5.py <run_uid> --out-dir /lus/eagle/.../staged

The script refuses to export runs whose metadata isn't marked
``fine_tunable=True`` — those runs lack the reconstructed ``final/probe`` /
``final/object`` that ``_para.hdf5`` requires.

Schema written (mirrors what ``PtychographyDataset._load_hdf5_pattern`` reads):

    <object_name>_dp.hdf5:
        /dp                       (nz, H, W) uint16 — *intensity*

    <object_name>_para.hdf5:
        /object                   (1, H_obj, W_obj) complex
            attr pixel_height_m   float (sample-plane pixel size in meters)
        /probe                    (N_modes, H, W) complex
        /probe_position_x_m       (nz,) float64 meters
        /probe_position_y_m       (nz,) float64 meters

Tiled stores the diffraction patterns as uint8 amplitude (sqrt of intensity)
to halve write volume. ``PtychographyDataset`` expects intensity in ``/dp``
(it Poisson-samples then sqrts at load time), so we square the amplitude back
to intensity before writing. uint16 is sufficient: max 255*255 = 65025 < 65535.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def _open_run(tiled_uri: str, api_key: Optional[str]):
    """Open a Tiled run container with the auth precedence the dataset uses."""
    from tiled.client import from_uri

    resolved = api_key or os.environ.get("TILED_API_KEY") or None
    kwargs = {"api_key": resolved} if resolved else {}
    return from_uri(tiled_uri, **kwargs)


def _resolve_uri(uid: str, base_url: str) -> str:
    """Build the per-run container URI from a UID and a Tiled base URL.

    Accepts a UID alone (joined onto the default catalog path), or a full URI
    (returned unchanged so users can paste paths from the dashboard).
    """
    if uid.startswith(("http://", "https://")):
        return uid.rstrip("/") + "/"
    base = base_url.rstrip("/")
    return f"{base}/api/v1/metadata/hxn/processed/holoptycho/{uid}/"


def _resolve_object_name(run, override: Optional[str]) -> str:
    if override:
        return override
    meta = dict(run.metadata or {})
    name = meta.get("scan_id") or meta.get("scan_num")
    if not name:
        raise ValueError(
            "Could not determine object_name: run metadata has no scan_id or "
            "scan_num. Pass --object-name explicitly."
        )
    return str(name)


def _stream_intensity(dp_node, dp_dataset: h5py.Dataset, chunk: int) -> None:
    """Copy uint8 amplitude → uint16 intensity in chunks, server-side sliced."""
    nz = dp_node.shape[0]
    for start in range(0, nz, chunk):
        stop = min(start + chunk, nz)
        amp_u8 = np.asarray(dp_node[start:stop])
        # Square uint8 amplitude back to intensity. Cast first so the multiply
        # doesn't overflow (uint8 * uint8 wraps); uint16 holds the result.
        intensity = amp_u8.astype(np.uint16) ** 2
        dp_dataset[start:stop] = intensity
        print(
            f"  dp: {stop}/{nz} frames "
            f"({100 * stop / nz:.1f}%)",
            flush=True,
        )


def export_run(
    *,
    tiled_uri: str,
    out_dir: Path,
    api_key: Optional[str] = None,
    object_name: Optional[str] = None,
    chunk: int = 256,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Export a Tiled run to a paired ``*_dp.hdf5`` / ``*_para.hdf5``.

    Returns the (dp_path, para_path) pair on success.
    """
    run = _open_run(tiled_uri, api_key)
    meta = dict(run.metadata or {})
    if not meta.get("fine_tunable", False):
        raise SystemExit(
            f"Run at {tiled_uri} is not marked fine_tunable in its metadata.\n"
            "ptycho-vit training needs reconstructed final/probe and final/object "
            "as supervised targets; only runs created with recon_mode='iterative' "
            "or 'both' produce them. Re-run holoptycho on this scan with one of "
            "those recon modes."
        )
    if not meta.get("complete", False):
        print(
            f"WARNING: run metadata['complete'] is False — pipeline may not have "
            "finished writing. Proceeding, but expect missing or partial frames.",
            file=sys.stderr,
        )

    name = _resolve_object_name(run, object_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dp_path = out_dir / f"{name}_dp.hdf5"
    para_path = out_dir / f"{name}_para.hdf5"

    for p in (dp_path, para_path):
        if p.exists() and not overwrite:
            raise SystemExit(
                f"{p} already exists. Pass --overwrite to replace, or move it aside."
            )

    diffraction = run["diffraction"]
    final = run["final"]
    dp_node = diffraction["dp"]
    nz, h, w = dp_node.shape
    print(f"Exporting {name}: {nz} frames @ {h}x{w}", flush=True)
    print(f"  -> {dp_path}", flush=True)
    print(f"  -> {para_path}", flush=True)

    # Diffraction file (streamed)
    with h5py.File(dp_path, "w") as f:
        dset = f.create_dataset(
            "dp",
            shape=(nz, h, w),
            dtype=np.uint16,
            chunks=(min(chunk, nz), h, w),
        )
        _stream_intensity(dp_node, dset, chunk)

    # Parameters file
    pixel_size_m = meta.get("x_pixel_m")
    if pixel_size_m is None:
        raise SystemExit(
            "Run metadata missing 'x_pixel_m' — required as object/pixel_height_m."
        )
    object_arr = np.asarray(final["object"][...])
    probe_arr = np.asarray(final["probe"][...])
    pos_x_m = np.asarray(diffraction["probe_position_x_m"][...]).astype(np.float64)
    pos_y_m = np.asarray(diffraction["probe_position_y_m"][...]).astype(np.float64)

    with h5py.File(para_path, "w") as f:
        obj_ds = f.create_dataset("object", data=object_arr)
        obj_ds.attrs["pixel_height_m"] = float(pixel_size_m)
        f.create_dataset("probe", data=probe_arr)
        f.create_dataset("probe_position_x_m", data=pos_x_m)
        f.create_dataset("probe_position_y_m", data=pos_y_m)

    print(
        f"Done. Object={object_arr.shape}{object_arr.dtype}, "
        f"probe={probe_arr.shape}{probe_arr.dtype}, "
        f"pixel_height_m={pixel_size_m}",
        flush=True,
    )
    return dp_path, para_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a holoptycho Tiled run to *_dp.hdf5 + *_para.hdf5.",
    )
    parser.add_argument(
        "uid",
        help="Run UID (joined onto --tiled-base-url) or a full Tiled run URI.",
    )
    parser.add_argument(
        "--tiled-base-url",
        default=os.environ.get(
            "TILED_BASE_URL", "https://tiled.nsls2.bnl.gov"
        ),
        help="Tiled server base URL (default: $TILED_BASE_URL or "
             "https://tiled.nsls2.bnl.gov).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Tiled API key. Default: $TILED_API_KEY env var, then `tiled login` "
             "cached credentials.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write the *_dp.hdf5 / *_para.hdf5 pair into.",
    )
    parser.add_argument(
        "--object-name",
        default=None,
        help="Override the object name used as the file prefix. Defaults to "
             "scan_id from the run's metadata.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=256,
        help="Frames per Tiled read (default: 256). Larger = fewer round trips, "
             "more peak memory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_dp.hdf5 / *_para.hdf5 in out-dir.",
    )

    args = parser.parse_args()
    tiled_uri = _resolve_uri(args.uid, args.tiled_base_url)
    print(f"Tiled URI: {tiled_uri}", flush=True)
    export_run(
        tiled_uri=tiled_uri,
        out_dir=args.out_dir,
        api_key=args.api_key,
        object_name=args.object_name,
        chunk=args.chunk,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
