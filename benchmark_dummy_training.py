import argparse
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_dummy import DummyPtychographyDataset
from model.model import PtychoViT


def parse_args():
    parser = argparse.ArgumentParser(description="Dummy zero-data PtychoViT training benchmark")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--normalization", type=float, default=100000.0)
    parser.add_argument("--scale", type=float, default=10000.0)
    parser.add_argument("--log-average-window", type=int, default=100)
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu, cuda, cuda:0")
    return parser.parse_args()


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    return rank, world_size, local_rank


def resolve_device(device_arg, local_rank):
    if device_arg is not None:
        device = torch.device(device_arg)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        if device.index is None:
            device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    return device


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_since(start, device):
    sync_if_needed(device)
    return time.perf_counter() - start


def maybe_average_across_ranks(value, device, world_size):
    if world_size <= 1:
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return float(tensor.item())


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.log_average_window <= 0:
        raise ValueError("--log-average-window must be positive")

    rank, world_size, local_rank = init_distributed()
    is_main_process = rank == 0
    device = resolve_device(args.device, local_rank)

    dataset_length = args.iterations * args.batch_size * max(world_size, 1)
    dataset = DummyPtychographyDataset(
        length=dataset_length,
        image_size=args.image_size,
        normalization=args.normalization,
        scale=args.scale,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (not args.no_pin_memory) and device.type == "cuda",
        "shuffle": sampler is None,
        "drop_last": True,
        "sampler": sampler,
    }
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
        dataloader_kwargs["persistent_workers"] = True

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    data_iter = iter(dataloader)

    model = PtychoViT().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    if is_main_process:
        print(
            f"[{timestamp()}] Starting dummy benchmark: iterations={args.iterations} "
            f"batch_size={args.batch_size} world_size={world_size} device={device}",
            flush=True,
        )
        print(
            f"[{timestamp()}] Dummy tensors per sample: input=(1,{args.image_size},{args.image_size}) "
            f"labels=(1,{args.image_size},{args.image_size}) probe=(1,1,{args.image_size},{args.image_size})",
            flush=True,
        )

    window_data = 0.0
    window_forward = 0.0
    window_backward = 0.0
    window_total = 0.0
    total_data = 0.0
    total_forward = 0.0
    total_backward = 0.0
    total_time = 0.0

    for iteration in range(1, args.iterations + 1):
        sync_if_needed(device)
        data_start = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        data_time = elapsed_since(data_start, device)

        diff_amp, amp_patch, ph_patch, probe, _probe_pos, normalization, scale = batch
        diff_amp = diff_amp.to(device, non_blocking=True)
        amp_patch = amp_patch.to(device, non_blocking=True)
        ph_patch = ph_patch.to(device, non_blocking=True)
        probe = probe.to(device, non_blocking=True)
        normalization = normalization.to(device, non_blocking=True)
        scale = scale.to(device, non_blocking=True)
        input_probe = torch.view_as_real(probe)

        sync_if_needed(device)
        forward_start = time.perf_counter()
        _output_diff, output_amp, output_ph = model(diff_amp, input_probe, normalization, scale)
        loss = criterion(output_amp, amp_patch) + criterion(output_ph, ph_patch)
        forward_time = elapsed_since(forward_start, device)

        sync_if_needed(device)
        backward_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        backward_time = elapsed_since(backward_start, device)

        data_time = maybe_average_across_ranks(data_time, device, world_size)
        forward_time = maybe_average_across_ranks(forward_time, device, world_size)
        backward_time = maybe_average_across_ranks(backward_time, device, world_size)
        iteration_total_time = data_time + forward_time + backward_time

        window_data += data_time
        window_forward += forward_time
        window_backward += backward_time
        window_total += iteration_total_time
        total_data += data_time
        total_forward += forward_time
        total_backward += backward_time
        total_time += iteration_total_time

        if is_main_process:
            print(
                f"[{timestamp()}] Iteration {iteration}/{args.iterations} "
                f"data_loading={data_time:.6f}s forward={forward_time:.6f}s "
                f"backward={backward_time:.6f}s total={iteration_total_time:.6f}s "
                f"loss={loss.detach().item():.6e}",
                flush=True,
            )

        if iteration % args.log_average_window == 0 and is_main_process:
            window = args.log_average_window
            print(
                f"[{timestamp()}] Average over iterations "
                f"{iteration - window + 1}-{iteration}: "
                f"data_loading={window_data / window:.6f}s "
                f"forward={window_forward / window:.6f}s "
                f"backward={window_backward / window:.6f}s "
                f"total={window_total / window:.6f}s",
                flush=True,
            )
            window_data = 0.0
            window_forward = 0.0
            window_backward = 0.0
            window_total = 0.0

    if is_main_process:
        print(
            f"[{timestamp()}] Overall average over {args.iterations} iterations: "
            f"data_loading={total_data / args.iterations:.6f}s "
            f"forward={total_forward / args.iterations:.6f}s "
            f"backward={total_backward / args.iterations:.6f}s "
            f"total={total_time / args.iterations:.6f}s",
            flush=True,
        )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
