"""Stage a fraction of packed_*.hdf5 shards from Lustre to node-local tmpfs.

Run via MPI before training. Partitions the chosen shards round-robin across
the nodes in the MPI job; on each node only local-rank-0 performs the copy
(other ranks wait on an MPI barrier).

Writes a per-node manifest file ``<local_dir>/local_manifest.json``:
    {
      "node_idx": 0,
      "n_nodes": 32,
      "hostname": "x4418c6s1b0n0",
      "shard_fraction": 0.1,
      "shard_seed": 8,
      "selected_shards": [...],      # global list (same on every node)
      "local_shards": ["packed_00007.hdf5", ...]  # just the ones on this node
    }

Usage (inside an Aurora job):
    mpiexec -np $NTOTRANKS -ppn $NRANKS_PER_NODE \
        python scripts/stage_pack_to_local.py \
            --source /flare/SYNAPS-I/simulated_data_cleanedProbe_2_packed \
            --local-dir $TMPDIR/packed \
            --fraction 0.1 \
            --seed 8 \
            --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from mpi4py import MPI

MANIFEST_NAME = "local_manifest.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path,
                    help="Lustre directory containing packed_*.hdf5")
    ap.add_argument("--local-dir", required=True, type=Path,
                    help="Per-node local directory (e.g. $TMPDIR/packed)")
    ap.add_argument("--fraction", required=True, type=float,
                    help="Fraction of shards to stage: 0 < fraction <= 1")
    ap.add_argument("--seed", type=int, default=8,
                    help="Seed for shard shuffle (same across all ranks)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel copy threads per node")
    ap.add_argument("--verify", action="store_true", default=True,
                    help="Verify byte-size equality after copy")
    return ap.parse_args()


def _copy_one(src: Path, dst: Path) -> tuple[Path, int, float]:
    t0 = time.perf_counter()
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return dst, dst.stat().st_size, time.perf_counter() - t0


def main() -> int:
    args = parse_args()
    if not (0.0 < args.fraction <= 1.0):
        print(f"fraction must be in (0, 1], got {args.fraction}", file=sys.stderr)
        return 2

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    hostname = socket.gethostname()
    all_hosts = comm.allgather(hostname)
    unique_hosts = sorted(set(all_hosts))
    node_idx = unique_hosts.index(hostname)
    n_nodes = len(unique_hosts)
    local_rank = int(os.environ.get("PALS_LOCAL_RANKID", "0"))

    if rank == 0:
        source: Path = args.source.resolve()
        if not source.is_dir():
            print(f"source not a directory: {source}", file=sys.stderr)
            comm.Abort(3)
        all_shards = sorted(p.name for p in source.glob("packed_*.hdf5"))
        if not all_shards:
            print(f"no packed_*.hdf5 under {source}", file=sys.stderr)
            comm.Abort(4)

        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(all_shards))
        n_stage = max(1, math.ceil(args.fraction * len(all_shards)))
        selected = sorted(all_shards[i] for i in perm[:n_stage])
        print(
            f"[stage] source={source} total_shards={len(all_shards)} "
            f"fraction={args.fraction} -> staging {len(selected)} shard(s) "
            f"across {n_nodes} node(s)",
            flush=True,
        )
    else:
        selected = None
        source = None

    selected = comm.bcast(selected, root=0)
    source = comm.bcast(args.source.resolve() if rank == 0 else None, root=0)

    my_shards = selected[node_idx::n_nodes]
    local_dir: Path = args.local_dir.resolve()

    if local_rank == 0:
        local_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        total_bytes = 0
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_copy_one, source / name, local_dir / name): name
                for name in my_shards
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    dst, nbytes, dt = fut.result()
                    total_bytes += nbytes
                    if args.verify:
                        src_size = (source / name).stat().st_size
                        if src_size != nbytes:
                            errors.append(
                                f"{name}: size mismatch src={src_size} dst={nbytes}"
                            )
                except Exception as e:
                    errors.append(f"{name}: {e!r}")

        elapsed = time.perf_counter() - t0
        gib = total_bytes / (1024 ** 3)
        rate = gib / elapsed if elapsed > 0 else 0.0
        print(
            f"[stage node {node_idx}/{n_nodes} {hostname}] "
            f"copied {len(my_shards)} shard(s) = {gib:.1f} GiB "
            f"in {elapsed:.1f}s ({rate:.2f} GiB/s) -> {local_dir}",
            flush=True,
        )
        if errors:
            for e in errors:
                print(f"[stage node {node_idx}] ERROR: {e}", file=sys.stderr, flush=True)

        manifest = {
            "node_idx": node_idx,
            "n_nodes": n_nodes,
            "hostname": hostname,
            "source": str(source),
            "local_dir": str(local_dir),
            "shard_fraction": args.fraction,
            "shard_seed": args.seed,
            "selected_shards": selected,
            "local_shards": my_shards,
        }
        (local_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

        any_err = bool(errors)
    else:
        any_err = False

    any_err = comm.allreduce(1 if any_err else 0, op=MPI.SUM)
    comm.barrier()

    if any_err:
        if rank == 0:
            print(f"[stage] FAILED: {any_err} rank(s) reported errors", file=sys.stderr, flush=True)
        return 1

    if rank == 0:
        print(f"[stage] OK — all {n_nodes} node(s) staged their shards", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
