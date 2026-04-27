"""SLURM / Polaris / torchrun-style process group setup."""

from __future__ import annotations

import os
import socket
import subprocess
import torch
import torch.distributed as dist

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def ensure_env_from_polaris():
    """Populate torchrun-style env vars from SLURM or MPI if missing."""
    if MPI is None:
        raise ImportError("MPI is not installed. Please install MPI to use this function.")
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    local_rank = os.environ["PMI_LOCAL_RANK"] if "PMI_LOCAL_RANK" in os.environ else rank % 4
    os.environ["LOCAL_RANK"] = str(local_rank)

    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(2345)


def ensure_env_from_slurm():
    """Populate torchrun-style env vars from SLURM if missing."""
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]


def init_distributed(platform: str = "slurm"):
    if platform == "polaris":
        return init_distributed_polaris()
    if platform == "slurm":
        return init_distributed_slurm()
    raise ValueError(f"Invalid platform: {platform}. Must be 'polaris' or 'slurm'")


def init_distributed_polaris():
    """
    Initialize torch.distributed if WORLD_SIZE>1 and bind CUDA to a local device
    respecting CUDA_VISIBLE_DEVICES. Returns (rank, world_size, local_rank, device).
    """
    ensure_env_from_polaris()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank_env = int(os.environ.get("LOCAL_RANK", os.environ.get("PMI_LOCAL_RANK", "0")))
    rank_env = int(os.environ.get("RANK", "0"))

    dist.init_process_group("nccl", init_method="env://")

    if torch.cuda.is_available():
        nvis = torch.cuda.device_count()
        if world_size == 1:
            mapped_local = 0
        else:
            mapped_local = 0 if nvis == 1 else (local_rank_env % nvis)
        torch.cuda.set_device(mapped_local)
        device = torch.device(f"cuda:{mapped_local}")
        os.environ["LOCAL_RANK"] = str(mapped_local)
    else:
        mapped_local = 0
        device = torch.device("cpu")

    return rank_env, world_size, mapped_local, device


def init_distributed_slurm():
    """
    Initialize torch.distributed if WORLD_SIZE>1 and bind CUDA to a local device
    respecting CUDA_VISIBLE_DEVICES. Returns (rank, world_size, local_rank, device).
    """
    ensure_env_from_slurm()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank_env = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    rank_env = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))

    if "MASTER_ADDR" not in os.environ:
        if "SLURM_JOB_NODELIST" in os.environ:
            nodelist = os.environ["SLURM_JOB_NODELIST"]
            try:
                result = subprocess.run(
                    ["scontrol", "show", "hostnames", nodelist],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    first_node = result.stdout.strip().split("\n")[0]
                    os.environ["MASTER_ADDR"] = first_node
                else:
                    os.environ["MASTER_ADDR"] = socket.gethostname()
            except Exception:
                os.environ["MASTER_ADDR"] = socket.gethostname()
        else:
            os.environ["MASTER_ADDR"] = socket.gethostname()

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    if world_size > 1 and not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="nccl", init_method="env://")

    if torch.cuda.is_available():
        nvis = torch.cuda.device_count()
        if world_size == 1:
            mapped_local = 0
        else:
            mapped_local = 0 if nvis == 1 else (local_rank_env % nvis)
        torch.cuda.set_device(mapped_local)
        torch.backends.cudnn.benchmark = True
        device = torch.device(f"cuda:{mapped_local}")
        os.environ["LOCAL_RANK"] = str(mapped_local)
    else:
        mapped_local = 0
        device = torch.device("cpu")

    return rank_env, world_size, mapped_local, device


def ddp_barrier(world_size: int, device: torch.device) -> None:
    """Keep ranks in lockstep before collectives. Use after rank-only work (e.g. wandb on rank 0)."""
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return
    if device.type == "cuda" and device.index is not None:
        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
