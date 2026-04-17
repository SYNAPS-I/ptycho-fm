import json
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import h5py
import numpy as np
from mpi4py import MPI

from data_simple import PtychographyDatasetSimple

# ── CONFIGURE THESE FOR YOUR SYSTEM ──────────────────────────────────────────
SOURCE = Path("/flare/SYNAPS-I/simulated_data_cleanedProbe_2")
OUT = Path("/flare/SYNAPS-I/simulated_data_cleanedProbe_2_packed")
MAX_SHARDS = 1563                   # ~32 objects/shard ≈ 7.4 GB each (optimal for NVMe staging)
MAX_OBJECTS: Optional[int] = None   # e.g. 200 for a short test; None = full catalog
MAX_PROBE_MODES = 10
NORMALIZATION_DICT: Optional[Path] = None  # Path to normalization.pkl, or None
# ─────────────────────────────────────────────────────────────────────────────


def _para_path(dp_path: Path) -> Path:
    stem = dp_path.stem[:-3] if dp_path.stem.endswith("_dp") else dp_path.stem
    return dp_path.parent / f"{stem}_para{dp_path.suffix}"


def probe_modes(pr, max_modes):
    pr = np.asarray(pr)[:1]
    m = pr.shape[1]
    if m > max_modes:
        return pr[:, :max_modes]
    if m < max_modes:
        z = np.zeros((1, max_modes - m, pr.shape[2], pr.shape[3]), dtype=pr.dtype)
        pr = np.concatenate([pr, z], axis=1)
    return pr


def fillvalue_for(dtype):
    if np.issubdtype(dtype, np.complexfloating):
        return 0j
    if np.issubdtype(dtype, np.floating):
        return np.float64(0.0).astype(dtype).item()
    return 0


def write_packed_shard(
    shard_idx: int,
    chunk: List[Path],
    *,
    source: Path,
    out_dir: Path,
    norm_map: dict,
    dt,
    mpi_rank: int,
) -> tuple[int, float, List[str]]:
    """Build one packed_{shard_idx:05d}.hdf5 from source pair paths in chunk.

    Returns (n_objects, seconds, object_keys) — keys match row order in the HDF5 ``object_key`` dataset.
    """
    meta = []
    for dp_path in chunk:
        key = PtychographyDatasetSimple.derive_object_name(dp_path, source)
        para = _para_path(dp_path)
        try:
            dpf = h5py.File(dp_path, "r")
        except OSError:
            print(f"[rank {mpi_rank}] open failed: {dp_path}", file=sys.stderr, flush=True)
            raise
        try:
            pf = h5py.File(para, "r")
        except OSError:
            dpf.close()
            print(f"[rank {mpi_rank}] open failed: {para}", file=sys.stderr, flush=True)
            raise
        try:
            with dpf, pf:
                dps = dpf["dp"].shape
                ni, dh_i, dw_i = int(dps[0]), int(dps[1]), int(dps[2])
                obj = pf["object"]
                osh = tuple(int(x) for x in obj.shape)
                prs = pf["probe"].shape
                ph_i, pw_i = int(prs[2]), int(prs[3])
                px = pf["probe_position_x_m"]
                dpx, dpy = px.dtype, pf["probe_position_y_m"].dtype
                meta.append(
                    {
                        "key": key,
                        "dp_path": dp_path,
                        "para_path": para,
                        "n_dp": ni,
                        "dh": dh_i,
                        "dw": dw_i,
                        "obj_shape": osh,
                        "ph": ph_i,
                        "pw": pw_i,
                        "dtype_dp": dpf["dp"].dtype,
                        "dtype_obj": obj.dtype,
                        "dtype_probe": pf["probe"].dtype,
                        "dtype_px": dpx,
                        "dtype_py": dpy,
                        "norm": norm_map.get(key),
                        "pixel_height_m": float(pf["object"].attrs.get("pixel_height_m", np.nan)),
                    }
                )
        except OSError:
            print(
                f"[rank {mpi_rank}] read failed: dp={dp_path} para={para}",
                file=sys.stderr,
                flush=True,
            )
            raise

    nob = len(meta)
    mxp = max(m["n_dp"] for m in meta)
    dh = max(m["dh"] for m in meta)
    dw = max(m["dw"] for m in meta)
    od0 = max(m["obj_shape"][0] for m in meta)
    oh = max(m["obj_shape"][1] for m in meta)
    ow = max(m["obj_shape"][2] for m in meta)
    ph = max(m["ph"] for m in meta)
    pw = max(m["pw"] for m in meta)
    m0 = meta[0]
    out_path = out_dir / f"packed_{shard_idx:05d}.hdf5"

    t_shard0 = time.perf_counter()
    with h5py.File(out_path, "w") as out:
        out.attrs["max_probe_modes"] = MAX_PROBE_MODES
        out.create_dataset(
            "dp",
            shape=(nob, mxp, dh, dw),
            dtype=m0["dtype_dp"],
            fillvalue=fillvalue_for(m0["dtype_dp"])#,
            #chunks=(1, 1, dh, dw),
        )
        out.create_dataset(
            "object",
            shape=(nob, od0, oh, ow),
            dtype=m0["dtype_obj"],
            fillvalue=fillvalue_for(m0["dtype_obj"])#,
            #chunks=(1, od0, oh, ow),
        )
        out.create_dataset(
            "probe",
            shape=(nob, 1, MAX_PROBE_MODES, ph, pw),
            dtype=m0["dtype_probe"],
            fillvalue=fillvalue_for(m0["dtype_probe"])#,
            #chunks=(1, 1, MAX_PROBE_MODES, ph, pw),
        )
        out.create_dataset(
            "probe_position_x_m",
            shape=(nob, mxp),
            dtype=m0["dtype_px"],
            fillvalue=fillvalue_for(m0["dtype_px"])#,
            #chunks=(1, mxp),
        )
        out.create_dataset(
            "probe_position_y_m",
            shape=(nob, mxp),
            dtype=m0["dtype_py"],
            fillvalue=fillvalue_for(m0["dtype_py"])#,
            #chunks=(1, mxp),
        )
        ds_n = out.create_dataset("n_dp", shape=(nob,), dtype=np.int64)
        ds_norm = out.create_dataset("normalization", shape=(nob,), dtype=np.float64, fillvalue=np.nan)
        ds_pix = out.create_dataset("pixel_height_m", shape=(nob,), dtype=np.float64)
        keys = []
        ds_dp = out["dp"]
        ds_obj = out["object"]
        ds_pr = out["probe"]
        ds_px = out["probe_position_x_m"]
        ds_py = out["probe_position_y_m"]

        for i, m in enumerate(meta):
            ds_n[i] = m["n_dp"]
            if m["norm"] is not None:
                ds_norm[i] = m["norm"]
            ds_pix[i] = m["pixel_height_m"]
            keys.append(m["key"])
            dp_p = m["dp_path"]
            para_p = m["para_path"]
            try:
                dpf = h5py.File(dp_p, "r", libver="latest", swmr=True)
            except OSError:
                print(f"[rank {mpi_rank}] open failed: {dp_p}", file=sys.stderr, flush=True)
                raise
            try:
                pf = h5py.File(para_p, "r", libver="latest", swmr=True)
            except OSError:
                dpf.close()
                print(f"[rank {mpi_rank}] open failed: {para_p}", file=sys.stderr, flush=True)
                raise
            try:
                with dpf, pf:
                    ni = m["n_dp"]
                    dh_i, dw_i = m["dh"], m["dw"]
                    ds_dp[i, :ni, :dh_i, :dw_i] = dpf["dp"][:ni, :, :]
                    ds_px[i, :ni] = pf["probe_position_x_m"][:ni]
                    ds_py[i, :ni] = pf["probe_position_y_m"][:ni]
                    idx = (i,) + tuple(slice(0, s) for s in m["obj_shape"])
                    ds_obj[idx] = pf["object"][...]
                    pr = probe_modes(pf["probe"][...], MAX_PROBE_MODES)
                    ds_pr[i, 0, : pr.shape[1], : pr.shape[2], : pr.shape[3]] = pr[0]
            except OSError:
                print(
                    f"[rank {mpi_rank}] read failed: dp={dp_p} para={para_p}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

        out.create_dataset("object_key", data=np.array(keys, dtype=object), dtype=dt)

    return nob, time.perf_counter() - t_shard0, keys


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    t_wall0 = time.perf_counter()
    source = SOURCE.resolve()
    out_dir = OUT.resolve()

    norm_map: dict = {}
    if NORMALIZATION_DICT is not None:
        with open(NORMALIZATION_DICT, "rb") as _f:
            norm_map = pickle.load(_f)

    t_list = 0.0
    if rank == 0:
        t_list0 = time.perf_counter()
        pairs_all = PtychographyDatasetSimple.find_paired_files(source)
        t_list = time.perf_counter() - t_list0
        n_total = len(pairs_all)
        paths_payload = [str(p) for p in pairs_all]
    else:
        n_total = None
        paths_payload = None

    n_total = comm.bcast(n_total, root=0)
    paths_payload = comm.bcast(paths_payload, root=0)
    t_list = comm.bcast(t_list, root=0)

    if n_total == 0:
        if rank == 0:
            print("No paired files to pack.", file=sys.stderr)
        sys.exit(1)

    pairs_all = [Path(s) for s in paths_payload]

    per = max(1, (n_total + MAX_SHARDS - 1) // MAX_SHARDS)
    n_shards_if_full = (n_total + per - 1) // per

    if MAX_OBJECTS is not None:
        if MAX_OBJECTS < 1:
            if rank == 0:
                print(
                    "MAX_OBJECTS must be >= 1 when set, or use None to pack the full catalog.",
                    file=sys.stderr,
                )
            sys.exit(1)
        pairs = pairs_all[:MAX_OBJECTS]
    else:
        pairs = pairs_all

    n = len(pairs)
    n_shards = (n + per - 1) // per

    if rank == 0:
        print(
            f"MPI {size} ranks | catalog {n_total} pair(s), per_shard={per} (MAX_SHARDS={MAX_SHARDS}) "
            f"-> full catalog = {n_shards_if_full} shard(s)."
        )
        print(
            f"Packing {n} pair(s) -> {n_shards} shard(s) (≤{per} pairs/shard) -> {out_dir}"
        )
        print(
            f"  Shards round-robin: indices rank, rank+{size}, … (~{(n_shards + size - 1) // size} file(s)/rank)."
        )
        if NORMALIZATION_DICT is not None:
            print(f"Normalization dict: {NORMALIZATION_DICT}")
        print(f"Max probe modes (padded): {MAX_PROBE_MODES}")
        print(f"Listing pairs: {t_list:.2f}s")

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    dt = h5py.string_dtype("utf-8")
    n_local = 0
    t_pack_local0 = time.perf_counter()
    local_index: List[tuple[int, str, List[str]]] = []

    for shard_idx in range(rank, n_shards, size):
        start = shard_idx * per
        chunk = pairs[start : start + per]
        nob, t_shard, keys = write_packed_shard(
            shard_idx,
            chunk,
            source=source,
            out_dir=out_dir,
            norm_map=norm_map,
            dt=dt,
            mpi_rank=rank,
        )
        pack_name = f"packed_{shard_idx:05d}.hdf5"
        local_index.append((shard_idx, pack_name, keys))
        print(
            f"[rank {rank}] shard {shard_idx}: {pack_name} ({nob} objects) — {t_shard:.2f}s "
            f"({nob / t_shard:.2f} obj/s)",
            flush=True,
        )
        n_local += nob

    t_pack_local = time.perf_counter() - t_pack_local0
    n_tot = np.zeros(1, dtype=np.int64) if rank == 0 else None
    t_slow = np.zeros(1, dtype=np.float64) if rank == 0 else None
    comm.Reduce(np.array([n_local], dtype=np.int64), n_tot, op=MPI.SUM, root=0)
    comm.Reduce(np.array([t_pack_local], dtype=np.float64), t_slow, op=MPI.MAX, root=0)
    comm.barrier()

    gathered = comm.gather(local_index, root=0)

    if rank == 0:
        index_path = out_dir / "pack_index.json"
        rows: List[tuple[int, str, List[str]]] = []
        for part in gathered:
            rows.extend(part)
        rows.sort(key=lambda r: r[0])
        # One key per packed file: basename -> object ids (HDF5 row order).
        payload = {name: keys for _, name, keys in rows}
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote pack index: {index_path}")

    if rank == 0:
        t_wall = time.perf_counter() - t_wall0
        print(
            f"Done: wall {t_wall:.2f}s (list {t_list:.2f}s, slowest pack {t_slow[0]:.2f}s), "
            f"{int(n_tot[0])} pairs in {n_shards} shard(s)."
        )
