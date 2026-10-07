import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np


def find_paired_files(directory: Path) -> list[Path]:
    """Return sorted ``*_dp.hdf5`` paths that have a matching ``*_para.hdf5`` sibling."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    out: list[Path] = []
    for dp in sorted(directory.rglob("*_dp.hdf5")):
        para = dp.with_name(f"{dp.stem[:-3]}_para{dp.suffix}")
        if para.is_file():
            out.append(dp)
    if not out:
        raise ValueError(f"No paired HDF5 files found in {directory}")
    print(f"pack_hdf5: {len(out)} paired file(s) under {directory}", flush=True)
    return out


def derive_object_name(file_path: Path, base_dir: Path) -> str:
    """Stable object id: relative path under ``base_dir`` without ``_dp`` suffix, else stem."""
    fp, bd = file_path.resolve(), Path(base_dir).resolve()
    if not fp.is_relative_to(bd):
        stem = file_path.stem
        return stem.removesuffix("_dp")
    rel = fp.relative_to(bd).with_suffix("")
    name = rel.name
    if name.endswith("_dp"):
        rel = rel.with_name(name[:-3])
    return rel.as_posix()


def positive_int(value: str) -> int:
    """Return a positive integer for argparse options."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pack paired *_dp.hdf5/*_para.hdf5 files into HDF5 shards."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Root directory containing paired *_dp.hdf5 and *_para.hdf5 files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory in which to write packed shards and pack_index.json",
    )
    parser.add_argument(
        "--max-shards",
        type=positive_int,
        default=500,
        help="Maximum number of output shards (default: 500)",
    )
    parser.add_argument(
        "--max-objects",
        type=positive_int,
        default=None,
        help="Pack only the first N objects; omit to pack the full catalog",
    )
    parser.add_argument(
        "--max-probe-modes",
        type=positive_int,
        default=10,
        help="Number of probe modes retained or padded to (default: 10)",
    )
    parser.add_argument(
        "--normalization-dict",
        type=Path,
        default=None,
        help="Optional pickle mapping object keys to normalization values",
    )
    return parser


def _para_path(dp_path: Path) -> Path:
    stem = dp_path.stem.removesuffix("_dp")
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
    chunk: list[Path],
    *,
    source: Path,
    out_dir: Path,
    norm_map: dict,
    dt,
    mpi_rank: int,
    max_probe_modes: int,
) -> tuple[int, float, list[str]]:
    """Build one packed_{shard_idx:05d}.hdf5 from source pair paths in chunk.

    Returns (n_objects, seconds, object_keys) — keys match row order in the HDF5 ``object_key`` dataset.
    """
    meta = []
    for dp_path in chunk:
        key = derive_object_name(dp_path, source)
        para = _para_path(dp_path)
        try:
            dpf = h5py.File(dp_path, "r")
        except OSError:
            print(
                f"[rank {mpi_rank}] open failed: {dp_path}", file=sys.stderr, flush=True
            )
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
                        "pixel_height_m": float(
                            pf["object"].attrs.get("pixel_height_m", np.nan)
                        ),
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
        out.attrs["max_probe_modes"] = max_probe_modes
        out.create_dataset(
            "dp",
            shape=(nob, mxp, dh, dw),
            dtype=m0["dtype_dp"],
            fillvalue=fillvalue_for(m0["dtype_dp"]),
        )
        out.create_dataset(
            "object",
            shape=(nob, od0, oh, ow),
            dtype=m0["dtype_obj"],
            fillvalue=fillvalue_for(m0["dtype_obj"]),
        )
        out.create_dataset(
            "probe",
            shape=(nob, 1, max_probe_modes, ph, pw),
            dtype=m0["dtype_probe"],
            fillvalue=fillvalue_for(m0["dtype_probe"]),
        )
        out.create_dataset(
            "probe_position_x_m",
            shape=(nob, mxp),
            dtype=m0["dtype_px"],
            fillvalue=fillvalue_for(m0["dtype_px"]),
        )
        out.create_dataset(
            "probe_position_y_m",
            shape=(nob, mxp),
            dtype=m0["dtype_py"],
            fillvalue=fillvalue_for(m0["dtype_py"]),
        )
        ds_n = out.create_dataset("n_dp", shape=(nob,), dtype=np.int64)
        ds_norm = out.create_dataset(
            "normalization", shape=(nob,), dtype=np.float64, fillvalue=np.nan
        )
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
                print(
                    f"[rank {mpi_rank}] open failed: {dp_p}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            try:
                pf = h5py.File(para_p, "r", libver="latest", swmr=True)
            except OSError:
                dpf.close()
                print(
                    f"[rank {mpi_rank}] open failed: {para_p}",
                    file=sys.stderr,
                    flush=True,
                )
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
                    pr = probe_modes(pf["probe"][...], max_probe_modes)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise SystemExit(
            "mpi4py is required to pack data; install it with `uv sync --extra mpi`"
        ) from exc

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    t_wall0 = time.perf_counter()
    source = args.source.expanduser().resolve()
    out_dir = args.output.expanduser().resolve()
    normalization_dict = (
        args.normalization_dict.expanduser().resolve()
        if args.normalization_dict is not None
        else None
    )

    norm_map: dict = {}
    if normalization_dict is not None:
        with open(normalization_dict, "rb") as _f:
            norm_map = pickle.load(_f)
        if not isinstance(norm_map, dict):
            raise TypeError("normalization dictionary must contain a dict")

    t_list = 0.0
    if rank == 0:
        t_list0 = time.perf_counter()
        pairs_all = find_paired_files(source)
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

    per = max(1, (n_total + args.max_shards - 1) // args.max_shards)
    n_shards_if_full = (n_total + per - 1) // per

    if args.max_objects is not None:
        pairs = pairs_all[: args.max_objects]
    else:
        pairs = pairs_all

    n = len(pairs)
    n_shards = (n + per - 1) // per

    if rank == 0:
        print(
            f"MPI {size} ranks | catalog {n_total} pair(s), per_shard={per} (max_shards={args.max_shards}) "
            f"-> full catalog = {n_shards_if_full} shard(s)."
        )
        print(
            f"Packing {n} pair(s) -> {n_shards} shard(s) (≤{per} pairs/shard) -> {out_dir}"
        )
        print(
            f"  Shards round-robin: indices rank, rank+{size}, … (~{(n_shards + size - 1) // size} file(s)/rank)."
        )
        if normalization_dict is not None:
            print(f"Normalization dict: {normalization_dict}")
        print(f"Max probe modes (padded): {args.max_probe_modes}")
        print(f"Listing pairs: {t_list:.2f}s")

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    dt = h5py.string_dtype("utf-8")
    n_local = 0
    t_pack_local0 = time.perf_counter()
    local_index: list[tuple[int, str, list[str]]] = []

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
            max_probe_modes=args.max_probe_modes,
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
        rows: list[tuple[int, str, list[str]]] = []
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
