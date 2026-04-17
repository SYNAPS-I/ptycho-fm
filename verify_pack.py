"""Sanity-check packed_*.hdf5 shards produced by pack_hdf5.py.

Run on the server where the packed files live. Tiered checks — cheapest first.
Exits with non-zero status on any failure; prints a summary either way.

Usage:
    python verify_pack.py --pack-dir /flare/SYNAPS-I/simulated_data_cleanedProbe_2_packed

    # Also cross-check contents against the source pair files:
    python verify_pack.py --pack-dir ... --source /flare/SYNAPS-I/simulated_data_cleanedProbe_2

    # Exercise the training dataset loader on N random patterns:
    python verify_pack.py --pack-dir ... --loader-samples 16
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

REQUIRED_DATASETS = {
    "dp",
    "object",
    "probe",
    "probe_position_x_m",
    "probe_position_y_m",
    "n_dp",
    "normalization",
    "pixel_height_m",
    "object_key",
}


def _fail(errs: list, msg: str) -> None:
    print(f"  FAIL: {msg}", flush=True)
    errs.append(msg)


def check_index(pack_dir: Path, errs: list) -> dict:
    idx_path = pack_dir / "pack_index.json"
    if not idx_path.exists():
        _fail(errs, f"missing pack_index.json at {idx_path}")
        return {}
    try:
        idx = json.loads(idx_path.read_text())
    except Exception as e:
        _fail(errs, f"pack_index.json unreadable: {e!r}")
        return {}

    on_disk = {p.name for p in pack_dir.glob("packed_*.hdf5")}
    missing = [n for n in idx if not (pack_dir / n).exists()]
    orphan = sorted(on_disk - set(idx))
    if missing:
        _fail(errs, f"{len(missing)} indexed shard(s) missing on disk, e.g. {missing[:3]}")
    if orphan:
        _fail(errs, f"{len(orphan)} shard(s) on disk not in index, e.g. {orphan[:3]}")

    total = sum(len(v) for v in idx.values())
    print(f"  index: {len(idx)} shard(s), {total} object(s)", flush=True)
    return idx


def check_shard_structure(path: Path, keys_from_index: list, errs: list) -> Optional[dict]:
    try:
        f = h5py.File(path, "r", libver="latest", swmr=True)
    except Exception as e:
        _fail(errs, f"{path.name}: cannot open ({e!r})")
        return None
    with f:
        present = set(f.keys())
        missing = REQUIRED_DATASETS - present
        if missing:
            _fail(errs, f"{path.name}: missing datasets {sorted(missing)}")
            return None

        nob = int(f["dp"].shape[0])
        checks = {
            "object rows":     f["object"].shape[0] == nob,
            "probe rows":      f["probe"].shape[0] == nob,
            "px rows":         f["probe_position_x_m"].shape[0] == nob,
            "py rows":         f["probe_position_y_m"].shape[0] == nob,
            "n_dp len":        f["n_dp"].shape[0] == nob,
            "norm len":        f["normalization"].shape[0] == nob,
            "pixel len":       f["pixel_height_m"].shape[0] == nob,
            "object_key len":  f["object_key"].shape[0] == nob,
            "index key count": len(keys_from_index) == nob,
        }
        for name, ok in checks.items():
            if not ok:
                _fail(errs, f"{path.name}: row-count mismatch — {name}")

        max_modes_attr = int(f.attrs.get("max_probe_modes", -1))
        if max_modes_attr < 0:
            _fail(errs, f"{path.name}: missing attr max_probe_modes")
        elif f["probe"].shape[2] != max_modes_attr:
            _fail(errs, f"{path.name}: probe mode dim {f['probe'].shape[2]} != attr {max_modes_attr}")

        n_dp = f["n_dp"][:]
        if (n_dp <= 0).any():
            _fail(errs, f"{path.name}: {int((n_dp <= 0).sum())} slot(s) have n_dp <= 0")
        if (n_dp > f["dp"].shape[1]).any():
            _fail(errs, f"{path.name}: n_dp exceeds dp max_n_dp")

        pix = f["pixel_height_m"][:]
        if not np.isfinite(pix).all():
            _fail(errs, f"{path.name}: non-finite pixel_height_m ({int((~np.isfinite(pix)).sum())} row(s))")
        if (pix <= 0).any():
            _fail(errs, f"{path.name}: non-positive pixel_height_m")

        keys_hdf5 = [
            (k.decode("utf-8") if isinstance(k, bytes) else str(k))
            for k in f["object_key"][:]
        ]
        if keys_from_index and keys_hdf5 != list(keys_from_index):
            _fail(errs, f"{path.name}: object_key order differs from pack_index.json")

        return {"nob": nob, "n_dp": n_dp, "keys": keys_hdf5}


def spot_check_content(path: Path, meta: dict, rng: random.Random, errs: list) -> None:
    with h5py.File(path, "r", libver="latest", swmr=True) as f:
        i = rng.randrange(meta["nob"])
        ni = int(meta["n_dp"][i])
        pi = rng.randrange(ni)

        dp = f["dp"][i, pi]
        if not np.isfinite(dp).all():
            _fail(errs, f"{path.name}[{i},{pi}]: non-finite dp")
        if dp.max() == 0:
            _fail(errs, f"{path.name}[{i},{pi}]: dp all zeros")

        obj = f["object"][i]
        if not np.count_nonzero(obj):
            _fail(errs, f"{path.name}[{i}]: object all zeros")

        pr = f["probe"][i, 0]
        nonzero_modes = [k for k in range(pr.shape[0]) if np.any(pr[k])]
        if not nonzero_modes:
            _fail(errs, f"{path.name}[{i}]: probe has no nonzero modes")

        px = f["probe_position_x_m"][i, :ni]
        py = f["probe_position_y_m"][i, :ni]
        if px.max() == px.min() and py.max() == py.min():
            _fail(errs, f"{path.name}[{i}]: all probe positions identical")

        key = meta["keys"][i]
        print(
            f"  spot {path.name}[{i}/{meta['nob']}], pattern {pi}/{ni}: "
            f"key={key}, dp[min,max]=[{dp.min():.3g},{dp.max():.3g}], "
            f"obj nz frac={np.count_nonzero(obj)/obj.size:.3f}, "
            f"probe modes with energy={nonzero_modes}",
            flush=True,
        )


def cross_check_source(path: Path, meta: dict, source: Path, rng: random.Random, errs: list) -> None:
    from data_simple import PtychographyDatasetSimple  # local import; optional

    pairs = PtychographyDatasetSimple.find_paired_files(source)
    key_to_dp = {PtychographyDatasetSimple.derive_object_name(p, source): p for p in pairs}

    i = rng.randrange(meta["nob"])
    key = meta["keys"][i]
    src_dp = key_to_dp.get(key)
    if src_dp is None:
        _fail(errs, f"{path.name}[{i}]: key {key} not found under {source}")
        return
    src_para = src_dp.parent / (
        (src_dp.stem[:-3] if src_dp.stem.endswith("_dp") else src_dp.stem) + src_dp.suffix
    )
    src_para = src_para.with_name(src_para.stem + "_para" + src_para.suffix) \
        if not src_para.stem.endswith("_para") else src_para

    with h5py.File(path, "r", libver="latest", swmr=True) as f, \
         h5py.File(src_dp, "r") as sdp:
        ni = int(meta["n_dp"][i])
        a = f["dp"][i, :ni]
        b = sdp["dp"][:ni]
        if a.shape != b.shape:
            _fail(errs, f"{path.name}[{i}] vs source: dp shape {a.shape} != {b.shape}")
            return
        if not np.array_equal(a, b):
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
            _fail(errs, f"{path.name}[{i}] vs source: dp not equal (max |Δ|={diff:.3g})")
        else:
            print(f"  cross-check {path.name}[{i}] key={key}: dp matches source", flush=True)


def exercise_loader(pack_dir: Path, n_samples: int, errs: list) -> None:
    try:
        from data_simple_pack import PtychographyDatasetPacked
    except Exception as e:
        _fail(errs, f"loader import failed: {e!r}")
        return

    try:
        ds = PtychographyDatasetPacked(pack_dir)
    except Exception as e:
        _fail(errs, f"loader init failed: {e!r}")
        return
    try:
        rng = random.Random(0)
        for _ in range(n_samples):
            idx = rng.randrange(len(ds))
            out = ds[idx]
            if not isinstance(out, tuple) or len(out) != 7:
                _fail(errs, f"loader[{idx}]: unexpected return {type(out)}")
                break
            amp, ap, ph, probe, xy, norm, scale = out
            if not (np.isfinite(amp).all() and np.isfinite(ap).all() and np.isfinite(ph).all()):
                _fail(errs, f"loader[{idx}]: non-finite amp/ap/ph")
                break
        else:
            print(f"  loader: {n_samples} random sample(s) decoded OK", flush=True)
    finally:
        ds.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", required=True, type=Path)
    ap.add_argument("--source", type=Path, default=None,
                    help="Optional: original source dir to cross-check one object per sampled shard.")
    ap.add_argument("--shards", type=int, default=0,
                    help="Structural-check this many random shards (0 = all).")
    ap.add_argument("--spot-shards", type=int, default=8,
                    help="Do content spot-checks on this many random shards.")
    ap.add_argument("--loader-samples", type=int, default=0,
                    help="Exercise PtychographyDatasetPacked on N random patterns (0 = skip).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pack_dir: Path = args.pack_dir.resolve()
    if not pack_dir.is_dir():
        print(f"not a directory: {pack_dir}", file=sys.stderr)
        return 2

    errs: list = []
    rng = random.Random(args.seed)
    t0 = time.perf_counter()

    print(f"[1/4] index")
    idx = check_index(pack_dir, errs)
    if not idx:
        print(f"\nFAILED: {len(errs)} error(s)")
        return 1

    names = list(idx.keys())
    structural_names = names if args.shards in (0, len(names)) else rng.sample(names, args.shards)
    print(f"[2/4] structure on {len(structural_names)} shard(s)")
    shard_meta: dict = {}
    for name in structural_names:
        m = check_shard_structure(pack_dir / name, idx[name], errs)
        if m is not None:
            shard_meta[name] = m

    spot_names = rng.sample(list(shard_meta), min(args.spot_shards, len(shard_meta)))
    print(f"[3/4] content spot-check on {len(spot_names)} shard(s)")
    for name in spot_names:
        spot_check_content(pack_dir / name, shard_meta[name], rng, errs)
        if args.source is not None:
            cross_check_source(pack_dir / name, shard_meta[name], args.source.resolve(), rng, errs)

    print(f"[4/4] loader")
    if args.loader_samples > 0:
        exercise_loader(pack_dir, args.loader_samples, errs)
    else:
        print("  skipped (set --loader-samples N to enable)")

    dt = time.perf_counter() - t0
    if errs:
        print(f"\nFAILED in {dt:.1f}s with {len(errs)} error(s); first few:")
        for e in errs[:10]:
            print(f"  - {e}")
        return 1
    print(f"\nOK — all checks passed in {dt:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
