"""Dispatch whole S26 fly scans across GPUs, once per participating node."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ptycho_fm.utils.dispatch import (
    get_available_gpus,
    gpu_environment,
    local_ranks,
    save_launch_script,
    shard_items,
    terminate_processes,
)

SCRIPT_DIR = Path(__file__).resolve().parent
LAZY_INFERENCE_FLYSCAN = SCRIPT_DIR / "run_flyscan_inference.py"
SHARD_PLAN_VERSION = 1


@dataclass(frozen=True)
class FilePair:
    object_name: str
    dp_path: Path
    para_path: Path | None


@dataclass(frozen=True)
class RankJob:
    local_index: int
    global_rank: int
    gpu_id: int
    shard_dir: Path
    files: list[FilePair]


def find_file_pairs(input_dir: Path) -> list[FilePair]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")

    pairs = []
    for dp_path in sorted(input_dir.glob("*_dp.hdf5")):
        object_name = dp_path.name[: -len("_dp.hdf5")]
        para_path = dp_path.with_name(f"{object_name}_para.hdf5")
        if not para_path.is_file():
            raise FileNotFoundError(
                f"Missing paired parameter file for {dp_path}: {para_path}"
            )
        pairs.append(
            FilePair(
                object_name=object_name,
                dp_path=dp_path.resolve(),
                para_path=para_path.resolve(),
            )
        )

    if not pairs:
        raise FileNotFoundError(f"No paired *_dp.hdf5 files found in {input_dir}")
    return pairs


def find_raw_h5_files(input_dir: Path) -> list[FilePair]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")

    files = sorted(
        path
        for pattern in ("*.h5", "*.hdf5")
        for path in input_dir.glob(pattern)
        if not path.stem.endswith(("_dp", "_para"))
    )
    if not files:
        raise FileNotFoundError(f"No raw .h5 files found in {input_dir}")
    return [
        FilePair(object_name=path.stem, dp_path=path.resolve(), para_path=None)
        for path in files
    ]


def verify_global_sharding(
    file_pairs: Sequence[FilePair], world_size: int
) -> list[list[FilePair]]:
    shards = shard_items(file_pairs, world_size)
    assigned = [pair.dp_path for shard in shards for pair in shard]
    unique_assigned = set(assigned)
    if len(assigned) != len(unique_assigned):
        raise RuntimeError(
            "Internal sharding error: at least one file was assigned to multiple ranks"
        )
    if unique_assigned != {pair.dp_path for pair in file_pairs}:
        raise RuntimeError("Internal sharding error: not all files were assigned")
    return shards


def _plan_file_record(pair: FilePair) -> dict:
    record = {
        "object_name": pair.object_name,
        "data_file": pair.dp_path.name,
        "data_size": pair.dp_path.stat().st_size,
        "parameter_file": None,
        "parameter_size": None,
    }
    if pair.para_path is not None:
        record["parameter_file"] = pair.para_path.name
        record["parameter_size"] = pair.para_path.stat().st_size
    return record


def write_shard_plan(
    plan_path: Path,
    data_format: str,
    shards: Sequence[Sequence[FilePair]],
) -> str:
    payload = {
        "version": SHARD_PLAN_VERSION,
        "data_format": data_format,
        "world_size": len(shards),
        "file_count": sum(len(shard) for shard in shards),
        "shards": [
            {
                "rank": rank,
                "files": [_plan_file_record(pair) for pair in shard],
            }
            for rank, shard in enumerate(shards)
        ],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(serialized, encoding="utf-8")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _resolve_plan_file(
    input_dir: Path, name: str, expected_size: int, label: str
) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"Invalid {label} filename in shard plan: {name!r}")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError(
            f"Invalid expected size for {label} file {name!r}: {expected_size!r}"
        )
    path = input_dir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Shard plan {label} file is missing on this node: {path}"
        )
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Shard plan {label} file has the wrong size on this node: {path} "
            f"({actual_size} bytes, expected {expected_size})"
        )
    return path.resolve()


def load_shard_plan(
    plan_path: Path,
    input_dir: Path,
    data_format: str,
    expected_world_size: int,
) -> tuple[list[list[FilePair]], str]:
    serialized = plan_path.read_text(encoding="utf-8")
    payload = json.loads(serialized)
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    if not isinstance(payload, dict):
        raise TypeError("Shard plan root must be a JSON object")
    if payload.get("version") != SHARD_PLAN_VERSION:
        raise ValueError(
            f"Unsupported shard plan version {payload.get('version')!r}; "
            f"expected {SHARD_PLAN_VERSION}"
        )
    if payload.get("data_format") != data_format:
        raise ValueError(
            f"Shard plan data format is {payload.get('data_format')!r}, "
            f"but --data-format is {data_format!r}"
        )
    if payload.get("world_size") != expected_world_size:
        raise ValueError(
            f"Shard plan world size is {payload.get('world_size')}, but this invocation "
            f"configured {expected_world_size}"
        )

    shard_records = payload.get("shards")
    if not isinstance(shard_records, list) or len(shard_records) != expected_world_size:
        raise ValueError(
            "Shard plan does not contain exactly one entry per global rank"
        )

    shards: list[list[FilePair]] = []
    assigned_names = set()
    for expected_rank, shard_record in enumerate(shard_records):
        if not isinstance(shard_record, dict):
            raise TypeError(f"Shard plan entry {expected_rank} must be a JSON object")
        if shard_record.get("rank") != expected_rank:
            raise ValueError(
                f"Shard plan rank entry {expected_rank} is labeled {shard_record.get('rank')!r}"
            )
        records = shard_record.get("files")
        if not isinstance(records, list):
            raise TypeError(
                f"Shard plan rank {expected_rank} has an invalid files list"
            )

        shard = []
        for record in records:
            if not isinstance(record, dict):
                raise TypeError(
                    f"Shard plan rank {expected_rank} contains an invalid file entry"
                )
            object_name = record.get("object_name")
            if not isinstance(object_name, str) or not object_name:
                raise ValueError(
                    f"Shard plan rank {expected_rank} has an invalid object name"
                )
            data_name = record.get("data_file")
            if data_name in assigned_names:
                raise ValueError(
                    f"Shard plan assigns data file more than once: {data_name}"
                )
            assigned_names.add(data_name)
            dp_path = _resolve_plan_file(
                input_dir, data_name, record.get("data_size"), "data"
            )

            para_name = record.get("parameter_file")
            if para_name is None:
                para_path = None
            else:
                para_path = _resolve_plan_file(
                    input_dir,
                    para_name,
                    record.get("parameter_size"),
                    "parameter",
                )
            shard.append(
                FilePair(object_name=object_name, dp_path=dp_path, para_path=para_path)
            )
        shards.append(shard)

    if data_format == "ptychodus" and any(
        pair.para_path is None for shard in shards for pair in shard
    ):
        raise ValueError("Paired input requires parameter files in the shard plan")
    if len(assigned_names) != payload.get("file_count"):
        raise ValueError(
            f"Shard plan contains {len(assigned_names)} unique data files, "
            f"but declares {payload.get('file_count')}"
        )
    return shards, fingerprint


def link_file(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing shard entry: {dst}")
    os.symlink(src, dst)


def create_shard_dir(
    root_dir: Path,
    global_rank: int,
    file_pairs: Sequence[FilePair],
    normalization_files: Sequence[Path],
) -> Path:
    shard_dir = root_dir / f"rank{global_rank:05d}"
    shard_dir.mkdir(parents=True, exist_ok=False)

    for pair in file_pairs:
        link_file(pair.dp_path, shard_dir / pair.dp_path.name)
        if pair.para_path is not None:
            link_file(pair.para_path, shard_dir / pair.para_path.name)

    for pkl_path in normalization_files:
        link_file(pkl_path.resolve(), shard_dir / pkl_path.name)

    return shard_dir


def build_inference_command(args: argparse.Namespace, shard_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(LAZY_INFERENCE_FLYSCAN),
        str(shard_dir),
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
        "--batch-size",
        str(args.batch_size),
        "--crop-size",
        str(args.crop_size),
        "--crop-size-step",
        str(args.crop_size_step),
        "--step-size",
        str(args.step_size),
        "--ny",
        str(args.ny),
        "--nx",
        str(args.nx),
        "--data-format",
        args.data_format,
    ]
    if args.checkpoint is not None:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.apply_noise is not None:
        cmd.append("--apply-noise" if args.apply_noise else "--no-apply-noise")
    cmd.extend(
        [
            "--scan-pattern",
            args.scan_pattern,
            "--bin",
            str(args.binning),
            "--device",
            "cuda:0",
        ]
    )
    cmd.append("--flip-patch-y" if args.flip_patch_y else "--no-flip-patch-y")
    if args.normalization_file is not None:
        cmd.extend(["--normalization-file", args.normalization_file])
    if args.crop_size_range is not None:
        cmd.extend(
            [
                "--crop-size-range",
                str(args.crop_size_range[0]),
                str(args.crop_size_range[1]),
            ]
        )
    if args.data_format == "raw-h5":
        cmd.extend(["--raw-dataset", args.raw_dataset])
        if args.raw_crop is not None:
            cmd.extend(["--raw-crop", *[str(value) for value in args.raw_crop]])
    return cmd


def launch_job(job: RankJob, args: argparse.Namespace) -> subprocess.Popen:
    env = gpu_environment(job.gpu_id, args.omp_num_threads)

    cmd = build_inference_command(args, job.shard_dir)
    save_launch_script(
        Path(args.output_dir) / "launches" / f"rank{job.global_rank:05d}.sh",
        cmd,
        job.gpu_id,
        args.omp_num_threads,
    )
    print(
        f"[rank {job.global_rank}] launching on GPU {job.gpu_id} with "
        f"{len(job.files)} file(s): {' '.join(cmd)}",
        flush=True,
    )
    return subprocess.Popen(cmd, env=env)


def run_jobs(jobs: Sequence[RankJob], args: argparse.Namespace) -> int:
    if args.dry_run:
        for job in jobs:
            names = ", ".join(pair.dp_path.name for pair in job.files)
            print(
                f"[rank {job.global_rank}] GPU {job.gpu_id}: {len(job.files)} file(s): {names}"
            )
        return 0

    active = []
    pending = [job for job in jobs if job.files]
    failures = 0
    try:
        while pending or active:
            still_active = []
            for proc, job in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, job))
                else:
                    status = "OK" if ret == 0 else f"FAILED exit={ret}"
                    print(
                        f"[rank {job.global_rank}] GPU {job.gpu_id}: {status}",
                        flush=True,
                    )
                    failures += int(ret != 0)
            active = still_active
            if pending:
                available = get_available_gpus(
                    set(args.gpus), set(), {job.gpu_id for _, job in active}
                )
                for job in pending[:]:
                    if job.gpu_id in available:
                        active.append((launch_job(job, args), job))
                        pending.remove(job)
            if pending or active:
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        return 130
    finally:
        terminate_processes([proc for proc, _ in active], args.terminate_timeout)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch run_flyscan_inference.py across local GPUs, with explicit multi-node "
            "file sharding and temporary per-rank symlink input directories."
        )
    )
    parser.add_argument(
        "input_path", help="Directory containing paired *_dp.hdf5 and *_para.hdf5 files"
    )
    parser.add_argument("--config", help="Path to config YAML file")
    parser.add_argument("--output-dir", help="Base directory for saving results")
    parser.add_argument(
        "--checkpoint", default=None, help="Path to model checkpoint .pth file"
    )
    parser.add_argument(
        "--apply-noise", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--data-format", choices=["ptychodus", "raw-h5"], default="ptychodus"
    )
    parser.add_argument("--raw-dataset", default="/entry/data/data")
    parser.add_argument(
        "--raw-crop", type=int, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX")
    )
    parser.add_argument(
        "--normalization-file",
        help="Explicit normalization pickle mapping; otherwise compute maxima",
    )
    parser.add_argument("--bin", type=int, default=1, dest="binning")
    parser.add_argument(
        "--scan-pattern", choices=["raster", "zigzag"], default="zigzag"
    )
    parser.add_argument(
        "--flip-patch-y", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--crop-size", type=int, default=104)
    parser.add_argument(
        "--crop-size-range", type=int, nargs=2, metavar=("START", "END")
    )
    parser.add_argument("--crop-size-step", type=int, default=1)
    parser.add_argument("--step-size", type=int, default=14)
    parser.add_argument("--ny", type=int, default=201)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument(
        "--node-rank", type=int, required=True, help="Zero-based index of this node"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=True,
        help="Total number of participating nodes",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        required=True,
        help="Local GPU IDs to use on this node",
    )
    parser.add_argument(
        "--shard-plan",
        help="Canonical JSON shard plan created once and copied unchanged to every node",
    )
    parser.add_argument(
        "--write-shard-plan",
        help="Write a canonical JSON shard plan from this node's input directory",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the shard plan and exit without loading the model or launching inference",
    )
    parser.add_argument(
        "--tmp-root", default=None, help="Parent directory for temporary shard symlinks"
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Do not delete temporary shard directories",
    )
    parser.add_argument("--omp-num-threads", type=int, default=1)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--terminate-timeout", type=float, default=30.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print assignments without launching inference",
    )
    args = parser.parse_args()

    if args.num_nodes <= 0:
        parser.error("--num-nodes must be positive")
    if args.node_rank < 0 or args.node_rank >= args.num_nodes:
        parser.error("--node-rank must be in [0, num_nodes)")
    if args.shard_plan is not None and args.write_shard_plan is not None:
        parser.error("--shard-plan and --write-shard-plan cannot be used together")
    if args.plan_only and args.write_shard_plan is None:
        parser.error("--plan-only requires --write-shard-plan")
    if not args.plan_only:
        if args.config is None:
            parser.error("--config is required unless --plan-only is used")
        if args.output_dir is None:
            parser.error("--output-dir is required unless --plan-only is used")
    if args.crop_size_step <= 0:
        parser.error("--crop-size-step must be positive")
    if (
        args.crop_size_range is not None
        and args.crop_size_range[1] < args.crop_size_range[0]
    ):
        parser.error("--crop-size-range END must be >= START")
    if args.data_format == "raw-h5" and not args.plan_only and args.raw_crop is None:
        parser.error("--raw-crop is required when --data-format raw-h5")

    try:
        local_ranks(args.node_rank, args.num_nodes, args.gpus)
    except ValueError as error:
        parser.error(str(error))
    if args.batch_size < 1 or args.omp_num_threads < 1 or args.binning < 1:
        parser.error("--batch-size, --omp-num-threads and --bin must be positive")
    if args.poll_interval <= 0 or args.terminate_timeout <= 0:
        parser.error("poll and termination intervals must be positive")
    if args.ny < 1 or args.nx < 1 or args.step_size < 1:
        parser.error("--ny, --nx and --step-size must be positive")
    if args.data_format == "ptychodus" and args.binning != 1:
        parser.error("--bin is only supported for raw input")
    if not args.plan_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --plan-only is used")
    if args.normalization_file is not None:
        args.normalization_file = str(Path(args.normalization_file).resolve())
    args.input_path = str(Path(args.input_path).resolve())
    if args.config is not None:
        args.config = str(Path(args.config).resolve())
    if args.output_dir is not None:
        args.output_dir = str(Path(args.output_dir).resolve())
    if args.checkpoint is not None:
        args.checkpoint = str(Path(args.checkpoint).resolve())
    if args.shard_plan is not None:
        args.shard_plan = str(Path(args.shard_plan).resolve())
    if args.write_shard_plan is not None:
        args.write_shard_plan = str(Path(args.write_shard_plan).resolve())
    return args


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_path)

    local_world_size = len(args.gpus)
    global_world_size = args.num_nodes * local_world_size
    if args.num_nodes > 1 and args.shard_plan is None and args.write_shard_plan is None:
        raise ValueError(
            "Multi-node runs require a canonical --shard-plan; create it with --write-shard-plan --plan-only"
        )
    if args.shard_plan is not None:
        global_shards, plan_fingerprint = load_shard_plan(
            Path(args.shard_plan),
            input_dir,
            args.data_format,
            global_world_size,
        )
        file_pairs = [pair for shard in global_shards for pair in shard]
        print(
            f"Loaded shard plan {args.shard_plan} "
            f"(sha256={plan_fingerprint}, files={len(file_pairs)}, ranks={global_world_size})",
            flush=True,
        )
    else:
        if args.data_format == "raw-h5":
            file_pairs = find_raw_h5_files(input_dir)
        else:
            file_pairs = find_file_pairs(input_dir)
        global_shards = verify_global_sharding(file_pairs, global_world_size)
        if args.write_shard_plan is not None:
            plan_fingerprint = write_shard_plan(
                Path(args.write_shard_plan),
                args.data_format,
                global_shards,
            )
            print(
                f"Wrote shard plan {args.write_shard_plan} "
                f"(sha256={plan_fingerprint}, files={len(file_pairs)}, ranks={global_world_size})",
                flush=True,
            )
            if args.plan_only:
                return 0

    normalization_files = []

    local_global_ranks = local_ranks(args.node_rank, args.num_nodes, args.gpus)
    local_assigned = [
        pair.dp_path for rank in local_global_ranks for pair in global_shards[rank]
    ]
    if len(local_assigned) != len(set(local_assigned)):
        raise RuntimeError("A local input file was assigned to more than one process")

    print(
        f"Found {len(file_pairs)} flyscan file(s). "
        f"Global world size: {global_world_size}; this node ranks: {local_global_ranks}",
        flush=True,
    )
    print(
        "Verified complete-file sharding: no individual file is split across processes.",
        flush=True,
    )

    names = [pair.object_name for pair in file_pairs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate object names would overwrite outputs")
    if args.dry_run:
        for rank, gpu in zip(local_global_ranks, args.gpus, strict=True):
            print(
                f"[rank {rank}] GPU {gpu}: {[pair.dp_path.name for pair in global_shards[rank]]}"
            )
        return 0

    tmp_parent = (
        Path(args.tmp_root).resolve() if args.tmp_root else Path(tempfile.gettempdir())
    )
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(
        tempfile.mkdtemp(
            prefix=f"ptycho_flyscan_node{args.node_rank:03d}_",
            dir=str(tmp_parent),
        )
    )

    try:
        jobs = []
        for local_index, gpu_id in enumerate(args.gpus):
            global_rank = args.node_rank * local_world_size + local_index
            files = global_shards[global_rank]
            shard_dir = create_shard_dir(
                tmp_root, global_rank, files, normalization_files
            )
            jobs.append(
                RankJob(
                    local_index=local_index,
                    global_rank=global_rank,
                    gpu_id=gpu_id,
                    shard_dir=shard_dir,
                    files=files,
                )
            )

        failures = run_jobs(jobs, args)
        return failures
    finally:
        if args.keep_shards:
            print(f"Keeping temporary shard directory: {tmp_root}", flush=True)
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)
            print(f"Deleted temporary shard directory: {tmp_root}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
