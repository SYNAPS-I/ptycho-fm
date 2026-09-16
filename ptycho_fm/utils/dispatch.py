"""GPU discovery and subprocess helpers shared by batch dispatchers."""

import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path


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


def gpu_environment(gpu_id: int, omp_num_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_num_threads)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def save_launch_script(path, command, gpu_id, omp_num_threads):
    """Save a shell-quoted, executable record of a worker launch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        f"export OMP_NUM_THREADS={shlex.quote(str(omp_num_threads))}\n"
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu_id))}\n"
        f"cd -- {shlex.quote(os.getcwd())}\n"
        f"exec {shlex.join(command)}\n"
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def terminate_processes(processes, timeout=30):
    """Terminate and reap children before their temporary inputs are removed."""
    processes = list(processes)
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def shard_items(items, world_size):
    """Assign each whole item to exactly one global rank in stable order."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    return [list(items[rank::world_size]) for rank in range(world_size)]


def local_ranks(node_rank, num_nodes, gpus):
    if num_nodes < 1 or not 0 <= node_rank < num_nodes:
        raise ValueError("node_rank must be in [0, num_nodes)")
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("GPU IDs must be non-negative and unique")
    return [node_rank * len(gpus) + index for index in range(len(gpus))]
