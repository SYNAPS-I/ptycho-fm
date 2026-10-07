"""Spot-check that packed shards match their source *_dp / *_para files.

Usage: python tests/test_pack_hdf5_consistency.py --source PATH --packed PATH
"""

import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Root directory containing the source *_dp.hdf5/*_para.hdf5 pairs",
    )
    parser.add_argument(
        "--packed",
        type=Path,
        required=True,
        help="Directory containing packed_*.hdf5 and pack_index.json",
    )
    parser.add_argument(
        "--num-shards",
        type=positive_int,
        default=10,
        help="Number of randomly selected shards to check (default: 10)",
    )
    parser.add_argument(
        "--tolerance",
        type=nonnegative_float,
        default=1e-5,
        help="Maximum allowed norm difference (default: 1e-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random shard-selection seed (default: 0)",
    )
    return parser


def err(a, b):
    return float(np.linalg.norm(np.asarray(a).ravel() - np.asarray(b).ravel()))


def check_row(pack, row, key, *, source: Path, tolerance: float):
    dp_src = source / f"{key}_dp.hdf5"
    para_src = source / f"{key}_para.hdf5"
    ni = int(pack["n_dp"][row])
    max_modes = int(pack.attrs["max_probe_modes"])

    with h5py.File(dp_src, "r") as dpf, h5py.File(para_src, "r") as pf:
        dh, dw = dpf["dp"].shape[1:3]
        osh = pf["object"].shape
        pr = np.asarray(pf["probe"])[:1, :max_modes]

        diffs = {
            "dp": err(pack["dp"][row, :ni, :dh, :dw], dpf["dp"][:ni]),
            "object": err(
                pack["object"][(row,) + tuple(slice(0, s) for s in osh)],
                pf["object"][...],
            ),
            "probe": err(pack["probe"][row, 0, : pr.shape[1]], pr[0]),
            "px": err(
                pack["probe_position_x_m"][row, :ni], pf["probe_position_x_m"][:ni]
            ),
            "py": err(
                pack["probe_position_y_m"][row, :ni], pf["probe_position_y_m"][:ni]
            ),
        }
        # fmt = "  ".join(f"{n}={d:.3e}" for n, d in diffs.items())
        # print(f"    {key}  {fmt}")
        for n, d in diffs.items():
            assert d <= tolerance, f"{key}: {n} norm_diff={d}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source.expanduser().resolve()
    packed = args.packed.expanduser().resolve()

    with open(packed / "pack_index.json", encoding="utf-8") as f:
        index = json.load(f)

    picked = random.Random(args.seed).sample(
        sorted(index), min(args.num_shards, len(index))
    )
    print(f"Checking {len(picked)} shard(s) in {packed}")

    t0 = time.perf_counter()
    for name in picked:
        t_shard = time.perf_counter()
        with h5py.File(packed / name, "r") as pack:
            for row, key in enumerate(index[name]):
                check_row(
                    pack,
                    row,
                    key,
                    source=source,
                    tolerance=args.tolerance,
                )
        print(
            f"  ok {name} ({len(index[name])} objects, "
            f"{time.perf_counter() - t_shard:.2f}s)"
        )

    print(f"All sampled shards match source. Total: {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
