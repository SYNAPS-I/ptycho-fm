"""Spot-check that packed shards match their source *_dp / *_para files.

Usage: python tests/test_pack_hdf5_consistency.py [num_shards]
"""
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np

SOURCE = Path("/pscratch/sd/s/shas1693/data/ptycho/simulated_data_cleanedProbe_2")
OUT = Path("/pscratch/sd/s/shas1693/data/ptycho/simulated_data_cleanedProbe_2_packed")
NUM_SHARDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TOL = 1e-5


def err(a, b):
    return float(np.linalg.norm(np.asarray(a).ravel() - np.asarray(b).ravel()))


def check_row(pack, row, key):
    dp_src = SOURCE / f"{key}_dp.hdf5"
    para_src = SOURCE / f"{key}_para.hdf5"
    ni = int(pack["n_dp"][row])
    max_modes = int(pack.attrs["max_probe_modes"])

    with h5py.File(dp_src, "r") as dpf, h5py.File(para_src, "r") as pf:
        dh, dw = dpf["dp"].shape[1:3]
        osh = pf["object"].shape
        pr = np.asarray(pf["probe"])[:1, :max_modes]

        diffs = {
            "dp": err(pack["dp"][row, :ni, :dh, :dw], dpf["dp"][:ni]),
            "object": err(pack["object"][(row,) + tuple(slice(0, s) for s in osh)], pf["object"][...]),
            "probe": err(pack["probe"][row, 0, : pr.shape[1]], pr[0]),
            "px": err(pack["probe_position_x_m"][row, :ni], pf["probe_position_x_m"][:ni]),
            "py": err(pack["probe_position_y_m"][row, :ni], pf["probe_position_y_m"][:ni]),
        }
        # fmt = "  ".join(f"{n}={d:.3e}" for n, d in diffs.items())
        # print(f"    {key}  {fmt}")
        for n, d in diffs.items():
            assert d <= TOL, f"{key}: {n} norm_diff={d}"


with open(OUT / "pack_index.json", encoding="utf-8") as f:
    index = json.load(f)

picked = random.Random(0).sample(sorted(index), min(NUM_SHARDS, len(index)))
print(f"Checking {len(picked)} shard(s) in {OUT}")

t0 = time.perf_counter()
for name in picked:
    t_shard = time.perf_counter()
    with h5py.File(OUT / name, "r") as pack:
        for row, key in enumerate(index[name]):
            check_row(pack, row, key)
    print(f"  ok {name} ({len(index[name])} objects, {time.perf_counter() - t_shard:.2f}s)")

print(f"All sampled shards match source. Total: {time.perf_counter() - t0:.2f}s")
