"""Discover reconstruction inputs and dispatch Pty-Chi workers across GPUs."""

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PTYCHI_RECON = os.path.join(SCRIPT_DIR, "ptychi_recon.py")
TV_SUPPORTED_ENGINES = {"lsqml", "epie", "rpie", "dm"}


@dataclass(frozen=True)
class ReconJob:
    scan: int
    data_path: str
    data_dir: str
    data_format: str
    scan_glob: str
    bin_dir_template: str
    probe_path: str
    positions_path: str
    output_dir: str
    object_name: str
    num_probe_modes: int


@dataclass(frozen=True)
class SkippedScan:
    scan: int
    reason: str


@dataclass(frozen=True)
class ReconstructionOptions:
    crop: tuple[int, ...]
    pixel_size: float
    engine: str
    opr: bool
    num_opr_modes: int
    total_variation: bool
    total_variation_weight: float
    total_variation_start: int
    total_variation_stride: int
    continue_recon: bool
    flip_x_positions: bool
    filter_indices: str | None
    frame_stride: int
    frame_sum: int
    preprocess: bool
    xray_energy_kev: float | None
    zp_diameter: float | None
    zp_outer_zone_width: float | None
    sample_detector_distance: float | None
    defocus_distance: float | None
    detector_pixel_size: float | None
    bin_factor: int | None
    max_position_jitter_m: float | None
    bin_pattern: str
    bin_frame_shape: tuple[int, ...] | None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ReconstructionOptions":
        bin_frame_shape = getattr(args, "bin_frame_shape", None)
        return cls(
            crop=tuple(args.crop),
            pixel_size=args.pixel_size,
            engine=args.engine,
            opr=args.opr,
            num_opr_modes=args.num_opr_modes,
            total_variation=args.total_variation,
            total_variation_weight=args.total_variation_weight,
            total_variation_start=args.total_variation_start,
            total_variation_stride=args.total_variation_stride,
            continue_recon=args.continue_recon,
            flip_x_positions=args.flip_x_positions,
            filter_indices=args.filter_indices,
            frame_stride=args.frame_stride,
            frame_sum=args.frame_sum,
            preprocess=args.preprocess,
            xray_energy_kev=args.xray_energy_kev,
            zp_diameter=args.zp_diameter,
            zp_outer_zone_width=args.zp_outer_zone_width,
            sample_detector_distance=args.sample_detector_distance,
            defocus_distance=args.defocus_distance,
            detector_pixel_size=args.detector_pixel_size,
            bin_factor=args.bin,
            max_position_jitter_m=args.max_position_jitter_m,
            bin_pattern=getattr(args, "bin_pattern", "image_*.bin"),
            bin_frame_shape=(
                tuple(bin_frame_shape) if bin_frame_shape is not None else None
            ),
        )


@dataclass(frozen=True)
class SchedulerOptions:
    allowed_gpus: set[int] | None
    excluded_gpus: set[int]
    omp_num_threads: int
    poll_interval: float
    dry_run: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SchedulerOptions":
        return cls(
            allowed_gpus=set(args.gpus) if args.gpus else None,
            excluded_gpus=set(args.exclude_gpus),
            omp_num_threads=args.omp_num_threads,
            poll_interval=args.poll_interval,
            dry_run=args.dry_run,
        )


@dataclass
class WatchState:
    known_files: set[str]
    previous_file: str | None = None
    queue_previous_on_shutdown: bool = False


ActiveJob = tuple[subprocess.Popen, ReconJob, int]


def parse_boolean(value: str) -> bool:
    """Parse explicit true/false CLI values while also supporting a bare flag."""
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def run_command(args: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout


def query_gpus() -> list[tuple[int, str]]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return []

    gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, uuid = [part.strip() for part in line.split(",", 1)]
        gpus.append((int(index), uuid))
    return gpus


def query_busy_gpus(uuid_to_index: dict[str, int]) -> set[int]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return set()

    busy = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        uuid = line.split(",", 1)[0].strip()
        if uuid in uuid_to_index:
            busy.add(uuid_to_index[uuid])
    return busy


def get_available_gpus(
    allowed_gpus: set[int] | None,
    excluded_gpus: set[int],
    locally_reserved_gpus: set[int],
) -> list[int]:
    gpus = query_gpus()
    if not gpus:
        raise RuntimeError("Could not query GPUs with nvidia-smi")

    uuid_to_index = {uuid: index for index, uuid in gpus}
    busy_gpus = query_busy_gpus(uuid_to_index)

    available = []
    for gpu_id, _uuid in gpus:
        if allowed_gpus is not None and gpu_id not in allowed_gpus:
            continue
        if gpu_id in excluded_gpus:
            continue
        if gpu_id in busy_gpus:
            continue
        if gpu_id in locally_reserved_gpus:
            continue
        available.append(gpu_id)
    return available


def scan_glob(data_root: str, scan_num: int, pattern: str, recursive: bool) -> list[str]:
    scan_token = pattern.format(scan=scan_num, scan_padded=f"{scan_num:04d}")
    search_path = os.path.join(data_root, "**", scan_token) if recursive else os.path.join(data_root, scan_token)
    return sorted(glob.glob(search_path, recursive=recursive))


def resolve_scan_template(template: str, base_dir: str, scan_num: int) -> str:
    path = template.format(scan=scan_num, scan_padded=f"{scan_num:04d}")
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.abspath(path)


def bin_scan_matches(
    data_root: str, scan_num: int, bin_dir_template: str, bin_pattern: str
) -> list[str]:
    bin_dir = resolve_scan_template(bin_dir_template, data_root, scan_num)
    if not os.path.isdir(bin_dir):
        return []
    matches = sorted(glob.glob(os.path.join(bin_dir, bin_pattern)))
    return [bin_dir] if matches else []


def find_scan_files(
    data_root: str,
    scan_start: int,
    scan_end: int,
    data_format: str,
    pattern: str,
    recursive: bool,
    bin_dir_template: str,
    bin_pattern: str,
) -> tuple[list[tuple[int, str, str]], list[SkippedScan]]:
    if scan_end < scan_start:
        raise ValueError(f"scan-end ({scan_end}) must be >= scan-start ({scan_start})")

    scan_files = []
    skipped = []
    for scan_num in range(scan_start, scan_end + 1):
        h5_matches = scan_glob(data_root, scan_num, pattern, recursive) if data_format in ("auto", "h5") else []
        bin_matches = bin_scan_matches(data_root, scan_num, bin_dir_template, bin_pattern) if data_format in ("auto", "bin") else []

        if h5_matches:
            matches = h5_matches
            scan_format = "h5"
        else:
            matches = bin_matches
            scan_format = "bin"

        if len(matches) == 0:
            print(f"Skipping scan {scan_num}: no {data_format} data matched")
            skipped.append(SkippedScan(scan_num, "missing_data_file"))
            continue
        if len(matches) > 1:
            raise ValueError(f"Multiple data files matched scan {scan_num}: {matches}")
        scan_files.append((scan_num, scan_format, os.path.abspath(matches[0])))
    return scan_files, skipped


def build_context(data_root: str, scan_num: int, data_path: str) -> dict[str, str]:
    data_root = os.path.abspath(data_root)
    data_path = os.path.abspath(data_path)
    data_dir = os.path.dirname(data_path)
    rel_path = os.path.relpath(data_path, data_root)
    rel_dir = os.path.dirname(rel_path)
    if rel_dir == "":
        rel_dir = "."
    stem, ext = os.path.splitext(os.path.basename(data_path))
    return {
        "scan": str(scan_num),
        "scan_padded": f"{scan_num:04d}",
        "data_root": data_root,
        "data_path": data_path,
        "data_dir": data_dir,
        "rel_path": rel_path,
        "rel_dir": rel_dir,
        "filename": os.path.basename(data_path),
        "stem": stem,
        "ext": ext,
    }


def resolve_template(template: str, base_dir: str, context: dict[str, str]) -> str:
    path = template.format(**context)
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.abspath(path)


def detect_num_probe_modes(probe_path: str, csv_probe_side: int) -> int:
    if probe_path.endswith(".csv"):
        probe = np.genfromtxt(probe_path, delimiter=",", dtype=np.complex64)
        values = probe.size
        pixels_per_mode = csv_probe_side * csv_probe_side
        if values % pixels_per_mode != 0:
            raise ValueError(
                f"CSV probe {probe_path} has {values} values, not a multiple of "
                f"{csv_probe_side}x{csv_probe_side}"
            )
        return values // pixels_per_mode

    if probe_path.endswith(".npy"):
        probe = np.load(probe_path, mmap_mode="r")
        return probe_modes_from_shape(probe.shape, probe_path)

    if probe_path.endswith(".hdf5"):
        with h5py.File(probe_path, "r") as f:
            if "probe" not in f:
                raise KeyError(f"Probe file {probe_path} does not contain dataset 'probe'")
            return probe_modes_from_shape(f["probe"].shape, probe_path)

    raise ValueError(f"Unsupported probe file format for ptychi_recon.py: {probe_path}")


def probe_modes_from_shape(shape: tuple[int, ...], probe_path: str) -> int:
    if len(shape) == 4:
        return int(shape[1])
    if len(shape) == 3:
        return int(shape[0])
    if len(shape) == 2:
        return 1
    raise ValueError(f"Cannot infer probe modes from shape {shape} in {probe_path}")


def read_frame_count(data_path: str) -> int:
    with h5py.File(data_path, "r") as f:
        return int(f["/entry/data/data"].shape[0])


def make_range_jobs(args: argparse.Namespace) -> tuple[list[ReconJob], list[SkippedScan]]:
    scan_files, skipped = find_scan_files(
        args.data_root,
        args.scan_start,
        args.scan_end,
        args.data_format,
        args.scan_glob,
        args.recursive,
        args.bin_dir_template,
        args.bin_pattern,
    )

    jobs = []
    for scan_num, scan_format, data_path in scan_files:
        if args.expected_frames is not None:
            frame_count = read_frame_count(data_path) if scan_format == "h5" else len(glob.glob(os.path.join(data_path, args.bin_pattern)))
            if frame_count != args.expected_frames:
                print(
                    f"Skipping scan {scan_num}: data shape[0]={frame_count}, "
                    f"expected {args.expected_frames}"
                )
                skipped.append(SkippedScan(scan_num, f"frame_count_{frame_count}_expected_{args.expected_frames}"))
                continue

        context = build_context(args.data_root, scan_num, data_path)
        if scan_format == "bin":
            context["data_dir"] = data_path
        probe_path = resolve_template(args.probe_template, context["data_dir"], context)
        positions_path = resolve_template(args.positions_template, context["data_dir"], context)
        output_dir = resolve_template(args.output_template, args.output_dir, context)
        object_name = args.object_template.format(**context)

        if not os.path.isfile(probe_path):
            raise FileNotFoundError(f"Probe file not found for scan {scan_num}: {probe_path}")
        if not os.path.isfile(positions_path):
            raise FileNotFoundError(f"Positions file not found for scan {scan_num}: {positions_path}")

        jobs.append(
            ReconJob(
                scan=scan_num,
                data_path=data_path,
                data_dir=os.path.dirname(data_path) if scan_format == "h5" else data_path,
                data_format=scan_format,
                scan_glob=os.path.basename(data_path) if scan_format == "h5" else args.scan_glob,
                bin_dir_template="." if scan_format == "bin" else args.bin_dir_template,
                probe_path=probe_path,
                positions_path=positions_path,
                output_dir=output_dir,
                object_name=object_name,
                num_probe_modes=detect_num_probe_modes(probe_path, args.csv_probe_side),
            )
        )
    return jobs, skipped


def get_sorted_h5_files(watch_dir: str, file_glob: str, recursive: bool) -> list[str]:
    search_path = (
        os.path.join(watch_dir, "**", file_glob)
        if recursive
        else os.path.join(watch_dir, file_glob)
    )
    files = [
        (os.path.getmtime(path), os.path.abspath(path))
        for path in glob.glob(search_path, recursive=recursive)
    ]
    files.sort()
    return [path for _mtime, path in files]


def parse_scan_number(data_path: str, scan_regex: str) -> int | None:
    match = re.search(scan_regex, os.path.basename(data_path))
    if match is None:
        return None
    return int(match.group(1))


def make_watch_job(
    data_path: str, args: argparse.Namespace
) -> tuple[ReconJob | None, SkippedScan | None]:
    scan_num = parse_scan_number(data_path, args.scan_regex)
    if scan_num is None:
        print(
            f"Skipping {os.path.basename(data_path)}: filename did not match "
            f"scan regex {args.scan_regex!r}"
        )
        return None, SkippedScan(-1, f"unparsed_scan:{data_path}")

    if args.expected_frames is not None:
        frame_count = read_frame_count(data_path)
        if frame_count != args.expected_frames:
            print(
                f"Skipping scan {scan_num}: data shape[0]={frame_count}, "
                f"expected {args.expected_frames}"
            )
            reason = f"frame_count_{frame_count}_expected_{args.expected_frames}"
            return None, SkippedScan(scan_num, reason)

    context = {
        "scan": str(scan_num),
        "scan_padded": f"{scan_num:04d}",
        "data_path": data_path,
        "data_dir": os.path.dirname(data_path),
        "rel_dir": ".",
        "filename": os.path.basename(data_path),
        "stem": os.path.splitext(os.path.basename(data_path))[0],
        "ext": os.path.splitext(os.path.basename(data_path))[1],
    }
    output_dir = resolve_template(args.output_template, args.output_dir, context)
    object_name = args.object_template.format(**context)

    return (
        ReconJob(
            scan=scan_num,
            data_path=data_path,
            data_dir=os.path.dirname(data_path),
            data_format="h5",
            scan_glob=os.path.basename(data_path),
            bin_dir_template=".",
            probe_path=args.probe_path,
            positions_path=args.positions_path,
            output_dir=output_dir,
            object_name=object_name,
            num_probe_modes=detect_num_probe_modes(
                args.probe_path, args.csv_probe_side
            ),
        ),
        None,
    )


def observe_new_files(state: WatchState, current_files: set[str]) -> list[str]:
    ready_files = []
    new_files = sorted(current_files - state.known_files, key=os.path.getmtime)
    for new_file in new_files:
        print(f"\nNew file detected: {os.path.basename(new_file)}")
        if state.previous_file is not None:
            ready_files.append(state.previous_file)
        state.previous_file = new_file
        state.queue_previous_on_shutdown = True
        state.known_files.add(new_file)
    return ready_files


def add_watch_jobs(
    data_paths: Sequence[str],
    args: argparse.Namespace,
    scheduler: "Scheduler",
    skipped: list[SkippedScan],
) -> None:
    for data_path in data_paths:
        job, skipped_scan = make_watch_job(data_path, args)
        if job is not None:
            scheduler.pending.append(job)
        if skipped_scan is not None:
            skipped.append(skipped_scan)


def save_skipped_scans(skipped: list[SkippedScan], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("scan,reason\n")
        f.writelines(
            f"{skipped_scan.scan},{skipped_scan.reason}\n"
            for skipped_scan in skipped
        )
    print(f"Saved skipped scan list to {output_path}")


def build_reconstruction_command(
    job: ReconJob, options: ReconstructionOptions
) -> list[str]:
    command = [
        sys.executable,
        PTYCHI_RECON,
        "--data-dir",
        job.data_dir,
        "--data-format",
        job.data_format,
        "--scan-start",
        str(job.scan),
        "--scan-end",
        str(job.scan),
        "--scan-glob",
        job.scan_glob,
        "--bin-dir-template",
        job.bin_dir_template,
        "--bin-pattern",
        options.bin_pattern,
        "--crop",
        *[str(value) for value in options.crop],
        "--probe-path",
        job.probe_path,
        "--positions-path",
        job.positions_path,
        "--pixel-size",
        str(options.pixel_size),
        "--output-dir",
        job.output_dir,
        "--num-probe-modes",
        str(job.num_probe_modes),
        "--object-name",
        job.object_name,
        "--engine",
        options.engine,
    ]
    if options.opr:
        command.extend(
            ["--opr", "true", "--num-opr-modes", str(options.num_opr_modes)]
        )
    if options.total_variation:
        command.extend(
            [
                "--total-variation",
                "--total-variation-weight",
                str(options.total_variation_weight),
                "--total-variation-start",
                str(options.total_variation_start),
                "--total-variation-stride",
                str(options.total_variation_stride),
            ]
        )
    if options.bin_frame_shape is not None:
        command.extend(
            [
                "--bin-frame-shape",
                *[str(value) for value in options.bin_frame_shape],
            ]
        )
    if options.continue_recon:
        command.append("--continue-recon")
    if options.flip_x_positions:
        command.append("--flip-x-positions")
    if options.filter_indices is not None:
        command.extend(["--filter-indices", options.filter_indices])
    if options.frame_stride != 1:
        command.extend(["--frame-stride", str(options.frame_stride)])
    if options.frame_sum != 1:
        command.extend(["--frame-sum", str(options.frame_sum)])
    if options.bin_factor is not None:
        command.extend(["--bin", str(options.bin_factor)])
    if options.preprocess:
        command.append("--preprocess")
        preprocess_options = (
            ("--xray-energy-kev", options.xray_energy_kev),
            ("--zp-diameter", options.zp_diameter),
            ("--zp-outer-zone-width", options.zp_outer_zone_width),
            ("--sample-detector-distance", options.sample_detector_distance),
            ("--defocus-distance", options.defocus_distance),
            ("--detector-pixel-size", options.detector_pixel_size),
            ("--max-position-jitter-m", options.max_position_jitter_m),
        )
        for option, value in preprocess_options:
            command.extend([option, str(value)])
    return command


def save_launch_script(
    job: ReconJob,
    command: Sequence[str],
    scheduler: SchedulerOptions,
    gpu_id: int,
) -> str:
    object_name = os.path.basename(job.object_name) or f"scan_{job.scan}"
    script_path = os.path.join(job.output_dir, f"{object_name}_launch.sh")
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"export OMP_NUM_THREADS={shlex.quote(str(scheduler.omp_num_threads))}\n"
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu_id))}\n"
        f"cd -- {shlex.quote(os.getcwd())}\n"
        f"exec {shlex.join(command)}\n"
    )
    with open(script_path, "w", encoding="utf-8") as file:
        file.write(script)
    os.chmod(script_path, 0o755)
    print(f"Saved reconstruction launch command to {script_path}")
    return script_path


def launch_reconstruction(
    job: ReconJob,
    reconstruction: ReconstructionOptions,
    scheduler: SchedulerOptions,
    gpu_id: int,
) -> subprocess.Popen:
    os.makedirs(job.output_dir, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(scheduler.omp_num_threads)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    command = build_reconstruction_command(job, reconstruction)
    save_launch_script(job, command, scheduler, gpu_id)

    print(f"Launching scan {job.scan} on GPU {gpu_id}: {' '.join(command)}")
    return subprocess.Popen(command, env=env)


def describe_job(job: ReconJob, options: ReconstructionOptions) -> str:
    return (
        f"scan={job.scan} data={job.data_path} probe={job.probe_path} "
        f"positions={job.positions_path} modes={job.num_probe_modes} "
        f"output={job.output_dir} object={job.object_name} "
        f"engine={options.engine} opr={options.opr} "
        f"num_opr_modes={options.num_opr_modes} "
        f"preprocess={options.preprocess} "
        f"xray_energy_kev={options.xray_energy_kev} "
        f"zp_diameter={options.zp_diameter} "
        f"zp_outer_zone_width={options.zp_outer_zone_width} "
        f"sample_detector_distance={options.sample_detector_distance} "
        f"defocus_distance={options.defocus_distance} "
        f"detector_pixel_size={options.detector_pixel_size} "
        f"bin={options.bin_factor} "
        f"max_position_jitter_m={options.max_position_jitter_m}"
    )


@dataclass
class Scheduler:
    reconstruction: ReconstructionOptions
    options: SchedulerOptions
    pending: list[ReconJob] = field(default_factory=list)
    active: list[ActiveJob] = field(default_factory=list)
    failures: int = 0

    def reap_finished(self) -> None:
        still_active = []
        for process, job, gpu_id in self.active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((process, job, gpu_id))
                continue
            status = "OK" if return_code == 0 else f"FAILED exit={return_code}"
            print(f"Finished scan {job.scan} on GPU {gpu_id}: {status}")
            if return_code != 0:
                self.failures += 1
        self.active = still_active

    def dispatch_pending(self) -> None:
        if self.options.dry_run:
            while self.pending:
                print(describe_job(self.pending.pop(0), self.reconstruction))
            return

        reserved_gpus = {gpu_id for _process, _job, gpu_id in self.active}
        available_gpus = get_available_gpus(
            self.options.allowed_gpus,
            self.options.excluded_gpus,
            reserved_gpus,
        )
        while self.pending and available_gpus:
            gpu_id = available_gpus.pop(0)
            job = self.pending.pop(0)
            process = launch_reconstruction(
                job, self.reconstruction, self.options, gpu_id
            )
            self.active.append((process, job, gpu_id))

    def step(self) -> None:
        self.reap_finished()
        if self.pending:
            self.dispatch_pending()

    def drain(self) -> int:
        while self.pending or self.active:
            self.step()
            if self.pending or self.active:
                time.sleep(self.options.poll_interval)
        return self.failures


RANGE_DESCRIPTION = (
    "Search for scans in a numeric range and dispatch ptychi_recon.py jobs "
    "onto GPUs with no active compute processes."
)
WATCH_DESCRIPTION = (
    "Monitor a directory for new HDF5 scan files and dispatch ptychi_recon.py "
    "jobs onto GPUs with no active compute processes."
)
DISPATCH_MODES = {"range", "watch"}


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    poll_interval: float,
    expected_frames_help: str,
    skipped_scans_help: str,
) -> None:
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        required=True,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Crop bounds forwarded to ptychi_recon.py",
    )
    parser.add_argument(
        "--csv-probe-side",
        type=int,
        default=256,
        help="Side length used to infer modes from CSV probes",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=6.65e-9, help="Pixel size in meters"
    )
    parser.add_argument(
        "--output-dir", default="./recon_outputs", help="Base output directory"
    )
    parser.add_argument(
        "--object-template",
        default="{stem}",
        help="Object/output file prefix template",
    )
    parser.add_argument(
        "--engine",
        choices=["lsqml", "epie", "rpie", "dm", "raar"],
        default="lsqml",
    )
    parser.add_argument(
        "--opr",
        type=parse_boolean,
        nargs="?",
        const=True,
        default=False,
        metavar="{true,false}",
        help="Enable OPR modes for non-projection engines (default: false)",
    )
    parser.add_argument(
        "--num-opr-modes",
        type=int,
        default=3,
        help="Number of additional OPR probe modes to add when --opr is true",
    )
    parser.add_argument(
        "--total-variation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable object total variation regularization for lsqml, epie, rpie, or dm"
        ),
    )
    parser.add_argument(
        "--total-variation-weight",
        type=float,
        default=1e-2,
        help="Total variation regularization weight",
    )
    parser.add_argument(
        "--total-variation-start",
        type=int,
        default=20,
        help="First epoch at which to apply total variation",
    )
    parser.add_argument(
        "--total-variation-stride",
        type=int,
        default=5,
        help="Apply total variation every N epochs after it starts",
    )
    parser.add_argument(
        "--continue-recon", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--flip-x-positions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward --flip-x-positions to ptychi_recon.py",
    )
    parser.add_argument(
        "--filter-indices", default=None, help="Forwarded to ptychi_recon.py"
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Load every Nth frame; forwarded to ptychi_recon.py",
    )
    parser.add_argument(
        "--frame-sum",
        type=int,
        default=1,
        help="Sum every N consecutive loaded frames; forwarded to ptychi_recon.py",
    )
    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward weakly scattering sample preprocessing to ptychi_recon.py",
    )
    parser.add_argument(
        "--xray-energy-kev",
        type=float,
        default=None,
        help="X-ray photon energy in keV; required with --preprocess",
    )
    parser.add_argument(
        "--zp-diameter",
        type=float,
        default=None,
        help="Zone plate diameter in meters; required with --preprocess",
    )
    parser.add_argument(
        "--zp-outer-zone-width",
        type=float,
        default=None,
        help="Zone plate outermost-zone width in meters; required with --preprocess",
    )
    parser.add_argument(
        "--sample-detector-distance",
        type=float,
        default=None,
        help="Sample-to-detector distance in meters; required with --preprocess",
    )
    parser.add_argument(
        "--defocus-distance",
        type=float,
        default=None,
        help="Signed sample defocus in meters; required with --preprocess",
    )
    parser.add_argument(
        "--detector-pixel-size",
        type=float,
        default=None,
        help="Raw detector pixel pitch in meters; required with --preprocess",
    )
    parser.add_argument(
        "--bin",
        type=int,
        default=None,
        help="Spatial binning factor forwarded to ptychi_recon.py",
    )
    parser.add_argument(
        "--max-position-jitter-m",
        type=float,
        default=None,
        help="Maximum absolute position jitter in meters; required with --preprocess",
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=None,
        help=expected_frames_help,
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=None,
        help="Restrict scheduling to these GPU IDs",
    )
    parser.add_argument(
        "--exclude-gpus",
        type=int,
        nargs="*",
        default=[],
        help="GPU IDs never used",
    )
    parser.add_argument("--omp-num-threads", type=int, default=1)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=poll_interval,
        help="Seconds between process/GPU checks",
    )
    parser.add_argument(
        "--skipped-scans-path",
        default=None,
        help=skipped_scans_help,
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print resolved jobs without launching",
    )


def add_range_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing scan HDF5 files or binary scan directories",
    )
    parser.add_argument(
        "--data-format",
        choices=["auto", "h5", "bin"],
        default="auto",
        help="Input data format. auto checks for HDF5 first, then binary directories.",
    )
    parser.add_argument("--scan-start", type=int, required=True, help="First scan number")
    parser.add_argument(
        "--scan-end", type=int, required=True, help="Last scan number, inclusive"
    )
    parser.add_argument(
        "--scan-glob",
        default="scan_{scan}_*.h5",
        help="Glob pattern under data-root; placeholders: {scan}, {scan_padded}",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Search recursively under data-root",
    )
    parser.add_argument(
        "--bin-dir-template",
        default="{scan}/raw",
        help="Binary frame directory template under data-root",
    )
    parser.add_argument(
        "--bin-pattern",
        default="image_*.bin",
        help="Binary frame filename glob inside each binary scan directory",
    )
    parser.add_argument(
        "--bin-frame-shape",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Shape of each binary frame before crop",
    )
    parser.add_argument(
        "--probe-template",
        required=True,
        help="Per-scan probe path template, resolved from the scan data directory",
    )
    parser.add_argument(
        "--positions-template",
        required=True,
        help="Per-scan positions path template, resolved from the scan data directory",
    )
    parser.add_argument(
        "--output-template",
        default="{rel_dir}",
        help="Per-scan output subdirectory template, resolved under --output-dir",
    )
    add_common_arguments(
        parser,
        poll_interval=10.0,
        expected_frames_help="Skip scans with a different frame count",
        skipped_scans_help=(
            "CSV file for skipped scan numbers; defaults to "
            "output-dir/skipped_scans.csv"
        ),
    )


def add_watch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--watch-dir", required=True, help="Directory to monitor for new HDF5 files"
    )
    parser.add_argument(
        "--file-glob", default="*.h5", help="File glob to monitor under watch-dir"
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Monitor recursively under watch-dir",
    )
    parser.add_argument(
        "--scan-regex",
        default=r"scan_(\d+)_.*\.h5$",
        help="Regex used to extract the scan number from each filename",
    )
    parser.add_argument(
        "--probe-path",
        required=True,
        help="Single probe file used for all monitored reconstructions",
    )
    parser.add_argument(
        "--positions-path",
        required=True,
        help="Single positions file used for all monitored reconstructions",
    )
    parser.add_argument(
        "--output-template",
        default=".",
        help="Per-file output subdirectory template, resolved under --output-dir",
    )
    parser.add_argument(
        "--process-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Queue files already present at startup",
    )
    add_common_arguments(
        parser,
        poll_interval=5.0,
        expected_frames_help="Skip files with a different frame count",
        skipped_scans_help=(
            "CSV file for skipped scans/files; defaults to "
            "output-dir/skipped_scans.csv"
        ),
    )


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "YAML file containing mode and CLI option values. Explicit CLI "
            "options override values from the file."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover reconstruction inputs and dispatch workers across GPUs."
    )
    add_config_argument(parser)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    range_parser = subparsers.add_parser("range", help=RANGE_DESCRIPTION)
    add_range_arguments(range_parser)
    watch_parser = subparsers.add_parser("watch", help=WATCH_DESCRIPTION)
    add_watch_arguments(watch_parser)
    return parser


def build_mode_parser(mode: str) -> argparse.ArgumentParser:
    if mode not in DISPATCH_MODES:
        raise ValueError(f"Unknown reconstruction dispatch mode: {mode}")
    description = RANGE_DESCRIPTION if mode == "range" else WATCH_DESCRIPTION
    parser = argparse.ArgumentParser(
        prog=f"{os.path.basename(sys.argv[0])} {mode}",
        description=description,
    )
    add_config_argument(parser)
    if mode == "range":
        add_range_arguments(parser)
    else:
        add_watch_arguments(parser)
    return parser


def extract_config_path(
    argv: Sequence[str], parser: argparse.ArgumentParser
) -> tuple[str | None, list[str]]:
    config_path = None
    remaining = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--config":
            if config_path is not None:
                parser.error("--config may only be specified once")
            if index + 1 >= len(argv):
                parser.error("--config requires a path")
            config_path = argv[index + 1]
            index += 2
            continue
        if argument.startswith("--config="):
            if config_path is not None:
                parser.error("--config may only be specified once")
            config_path = argument.split("=", 1)[1]
            if not config_path:
                parser.error("--config requires a path")
            index += 1
            continue
        remaining.append(argument)
        index += 1
    return config_path, remaining


def load_yaml_config(
    config_path: str, parser: argparse.ArgumentParser
) -> dict[str, object]:
    try:
        with open(config_path, encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as error:
        parser.error(f"could not read config file {config_path}: {error}")
    except yaml.YAMLError as error:
        parser.error(f"invalid YAML in config file {config_path}: {error}")

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        parser.error("the YAML config root must be a mapping")

    normalized = {}
    for raw_key, value in loaded.items():
        if not isinstance(raw_key, str):
            parser.error("all YAML config keys must be strings")
        key = raw_key.replace("-", "_")
        if key in normalized:
            parser.error(f"duplicate YAML config key after normalization: {key}")
        normalized[key] = value
    return normalized


def config_to_cli_args(
    config: dict[str, object], parser: argparse.ArgumentParser
) -> list[str]:
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    unknown = sorted(set(config) - set(actions) - {"mode"})
    if unknown:
        parser.error("unknown YAML config option(s): " + ", ".join(unknown))

    arguments = []
    for key, value in config.items():
        if key == "mode" or value is None:
            continue
        action = actions[key]
        long_options = [
            option for option in action.option_strings if option.startswith("--")
        ]
        if not long_options:
            parser.error(f"YAML config option {key} has no corresponding CLI option")

        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                parser.error(f"YAML config option {key} must be true or false")
            prefix = "--no-" if not value else "--"
            option = next(
                option for option in long_options if option.startswith(prefix)
            )
            arguments.append(option)
            continue

        option = next(
            (option for option in long_options if not option.startswith("--no-")),
            long_options[0],
        )
        if action.nargs in {"*", "+"} or isinstance(action.nargs, int):
            if not isinstance(value, (list, tuple)):
                parser.error(f"YAML config option {key} must be a list")
            arguments.append(option)
            arguments.extend(str(item) for item in value)
            continue
        if isinstance(value, (dict, list, tuple)):
            parser.error(f"YAML config option {key} must be a scalar value")
        serialized = str(value).lower() if isinstance(value, bool) else str(value)
        arguments.extend([option, serialized])
    return arguments


def validate_and_normalize_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    if args.opr and args.engine in {"dm", "raar"}:
        parser.error("--opr true is only supported with lsqml, epie, and rpie")
    if args.opr and args.num_opr_modes < 1:
        parser.error("--num-opr-modes must be >= 1 when --opr is true")
    if args.total_variation and args.engine not in TV_SUPPORTED_ENGINES:
        supported = ", ".join(sorted(TV_SUPPORTED_ENGINES))
        parser.error(f"--total-variation is only supported with {supported}")
    if args.total_variation and (
        not np.isfinite(args.total_variation_weight)
        or args.total_variation_weight <= 0
    ):
        parser.error("--total-variation-weight must be finite and > 0")
    if args.total_variation_start < 0:
        parser.error("--total-variation-start must be >= 0")
    if args.total_variation_stride < 1:
        parser.error("--total-variation-stride must be >= 1")
    if args.bin is not None and args.bin < 1:
        parser.error("--bin must be >= 1")
    if args.preprocess:
        required_preprocess_args = {
            "--xray-energy-kev": args.xray_energy_kev,
            "--zp-diameter": args.zp_diameter,
            "--zp-outer-zone-width": args.zp_outer_zone_width,
            "--sample-detector-distance": args.sample_detector_distance,
            "--defocus-distance": args.defocus_distance,
            "--detector-pixel-size": args.detector_pixel_size,
            "--bin": args.bin,
            "--max-position-jitter-m": args.max_position_jitter_m,
        }
        missing = [
            name for name, value in required_preprocess_args.items() if value is None
        ]
        if missing:
            parser.error("--preprocess requires " + " and ".join(missing))

        positive_preprocess_args = {
            "--xray-energy-kev": args.xray_energy_kev,
            "--zp-diameter": args.zp_diameter,
            "--zp-outer-zone-width": args.zp_outer_zone_width,
            "--sample-detector-distance": args.sample_detector_distance,
            "--detector-pixel-size": args.detector_pixel_size,
            "--max-position-jitter-m": args.max_position_jitter_m,
        }
        for name, value in positive_preprocess_args.items():
            if not np.isfinite(value) or value <= 0:
                parser.error(f"{name} must be finite and > 0")
        if not np.isfinite(args.defocus_distance):
            parser.error("--defocus-distance must be finite")
    elif args.max_position_jitter_m is not None:
        parser.error("--max-position-jitter-m is only valid with --preprocess")

    args.output_dir = os.path.abspath(args.output_dir)
    if args.skipped_scans_path is None:
        args.skipped_scans_path = os.path.join(args.output_dir, "skipped_scans.csv")
    else:
        args.skipped_scans_path = os.path.abspath(args.skipped_scans_path)

    if args.mode == "range":
        args.data_root = os.path.abspath(args.data_root)
    else:
        args.watch_dir = os.path.abspath(args.watch_dir)
        args.probe_path = os.path.abspath(args.probe_path)
        args.positions_path = os.path.abspath(args.positions_path)
        if not os.path.isfile(args.probe_path):
            raise FileNotFoundError(f"Probe file not found: {args.probe_path}")
        if not os.path.isfile(args.positions_path):
            raise FileNotFoundError(f"Positions file not found: {args.positions_path}")
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root_parser = build_parser()
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    config_path, cli_arguments = extract_config_path(raw_arguments, root_parser)
    config = (
        load_yaml_config(config_path, root_parser)
        if config_path is not None
        else {}
    )

    explicit_mode = None
    if cli_arguments and cli_arguments[0] in DISPATCH_MODES:
        explicit_mode = cli_arguments.pop(0)
    config_mode = config.get("mode")
    if config_mode is not None and (
        not isinstance(config_mode, str) or config_mode not in DISPATCH_MODES
    ):
        root_parser.error("YAML config option mode must be 'range' or 'watch'")
    if explicit_mode is not None and config_mode not in {None, explicit_mode}:
        root_parser.error(
            f"CLI mode {explicit_mode!r} conflicts with YAML mode {config_mode!r}"
        )

    mode = explicit_mode or config_mode
    if mode is None:
        if cli_arguments == ["--help"]:
            root_parser.parse_args(cli_arguments)
        root_parser.error("a mode is required: use 'range', 'watch', or YAML mode")

    parser = build_mode_parser(mode)
    config_arguments = config_to_cli_args(config, parser)
    args = parser.parse_args([*config_arguments, *cli_arguments])
    args.mode = mode
    args.config = os.path.abspath(config_path) if config_path is not None else None
    return validate_and_normalize_args(parser, args)


def create_scheduler(args: argparse.Namespace) -> Scheduler:
    return Scheduler(
        reconstruction=ReconstructionOptions.from_args(args),
        options=SchedulerOptions.from_args(args),
    )


def run_range(args: argparse.Namespace) -> int:
    jobs, skipped = make_range_jobs(args)
    print(f"Prepared {len(jobs)} reconstruction job(s)")
    scheduler = create_scheduler(args)
    scheduler.pending.extend(jobs)
    if not jobs:
        print("No reconstruction jobs found")
    exit_code = scheduler.drain()
    save_skipped_scans(skipped, args.skipped_scans_path)
    return exit_code


def run_watch(args: argparse.Namespace) -> int:
    if not os.path.isdir(args.watch_dir):
        print(f"Error: watch directory does not exist: {args.watch_dir}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    known_files = set(
        get_sorted_h5_files(args.watch_dir, args.file_glob, args.recursive)
    )
    state = WatchState(known_files=known_files)
    scheduler = create_scheduler(args)
    skipped: list[SkippedScan] = []

    if args.process_existing:
        files_to_queue = sorted(known_files, key=os.path.getmtime)
        print(f"Queuing {len(files_to_queue)} existing file(s)")
        add_watch_jobs(files_to_queue, args, scheduler, skipped)
    elif known_files:
        state.previous_file = max(known_files, key=os.path.getmtime)
        print(
            f"Found {len(known_files)} existing file(s), "
            "will start processing on next new file"
        )
        print(f"  Most recent existing file: {os.path.basename(state.previous_file)}")

    print(
        f"Watching {args.watch_dir} for {args.file_glob!r} files "
        f"(poll every {args.poll_interval}s)"
    )

    try:
        while True:
            scheduler.step()
            current_files = set(
                get_sorted_h5_files(args.watch_dir, args.file_glob, args.recursive)
            )
            ready_files = observe_new_files(state, current_files)
            add_watch_jobs(ready_files, args, scheduler, skipped)
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        final_file = state.previous_file
        if (
            state.queue_previous_on_shutdown
            and final_file is not None
            and final_file in state.known_files
        ):
            already_pending = any(job.data_path == final_file for job in scheduler.pending)
            already_active = any(
                job.data_path == final_file
                for _process, job, _gpu_id in scheduler.active
            )
            if not already_pending and not already_active:
                print(
                    "Queueing last observed file before shutdown: "
                    f"{os.path.basename(final_file)}"
                )
                add_watch_jobs([final_file], args, scheduler, skipped)

        scheduler.drain()
        save_skipped_scans(skipped, args.skipped_scans_path)
        print("Done.")
        return scheduler.failures


def run(args: argparse.Namespace) -> int:
    if args.mode == "range":
        return run_range(args)
    if args.mode == "watch":
        return run_watch(args)
    raise ValueError(f"Unknown reconstruction dispatch mode: {args.mode}")


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
