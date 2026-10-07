"""Oracle Cloud torchrun entry point with GPU-local CPU affinity.

This wrapper runs once per local torchrun rank. It binds the rank to its CUDA
device before the training code initializes NCCL, then divides the CPU cores
closest to that GPU among ranks that share the same GPU/NUMA affinity mask.
DataLoader workers inherit the resulting CPU affinity.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path


def _expand_cpu_list(spec: str) -> set[int]:
    """Expand Linux CPU-list syntax such as ``48-63,176-191``."""
    cpus: set[int] = set()
    for part in spec.split(","):
        start_end = part.strip().split("-", maxsplit=1)
        start = int(start_end[0])
        end = int(start_end[-1])
        cpus.update(range(start, end + 1))
    return cpus


def _gpu_cpu_affinities() -> dict[int, set[int]]:
    """Return the CPU affinity reported by nvidia-smi for every physical GPU."""
    output = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    affinities: dict[int, set[int]] = {}
    for line in output.splitlines():
        fields = line.split()
        if not fields or re.fullmatch(r"GPU\d+", fields[0]) is None:
            continue
        # The final columns are CPU Affinity, NUMA Affinity, and GPU NUMA ID.
        affinities[int(fields[0][3:])] = _expand_cpu_list(fields[-3])
    return affinities


def _physical_core_groups(cpus: set[int]) -> list[set[int]]:
    """Group SMT siblings so a physical core is never split between ranks."""
    groups: dict[tuple[int, int], set[int]] = defaultdict(set)
    for cpu in sorted(cpus):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = int((topology / "physical_package_id").read_text().strip())
        core = int((topology / "core_id").read_text().strip())
        groups[(package, core)].add(cpu)
    return sorted(groups.values(), key=min)


def _bind_cpu_affinity(local_rank: int) -> set[int]:
    """Give this rank a disjoint share of its GPU-local physical CPU cores."""
    gpu_affinities = _gpu_cpu_affinities()
    if local_rank not in gpu_affinities:
        raise RuntimeError(
            f"nvidia-smi did not report a CPU affinity for local GPU {local_rank}"
        )

    target = gpu_affinities[local_rank]
    peers = sorted(
        gpu for gpu, affinity in gpu_affinities.items() if affinity == target
    )
    permitted = set(os.sched_getaffinity(0))
    physical_cores = _physical_core_groups(target & permitted)
    if len(physical_cores) < len(peers):
        raise RuntimeError(
            f"GPU {local_rank} has only {len(physical_cores)} permitted physical "
            f"CPU cores to divide among local GPU ranks {peers}"
        )

    peer_index = peers.index(local_rank)
    start = len(physical_cores) * peer_index // len(peers)
    end = len(physical_cores) * (peer_index + 1) // len(peers)
    assigned = set().union(*physical_cores[start:end])
    os.sched_setaffinity(0, assigned)
    return assigned


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    cpu_affinity = _bind_cpu_affinity(local_rank)

    # Select CUDA before importing the normal training entry point, whose
    # distributed setup initializes the NCCL process group.
    import torch

    visible_gpus = torch.cuda.device_count()
    if visible_gpus == 0:
        raise RuntimeError("The Oracle multinode launcher requires CUDA GPUs")
    device_index = local_rank % visible_gpus
    torch.cuda.set_device(device_index)

    print(
        f"[oracle rank setup] LOCAL_RANK={local_rank} cuda:{device_index} "
        f"CPUs={','.join(map(str, sorted(cpu_affinity)))}",
        flush=True,
    )

    from ptycho_fm.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
