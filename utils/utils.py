import hashlib
import math
from typing import Any, Callable, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from custom_loss import WeightedLoss


def make_warmup_stable_lr_lambda(warmup_steps: int) -> Callable[[int], float]:
    """Linear warmup then constant LR=1 (no cooldown)."""
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0
    return lr_lambda


def make_cooldown_lr_lambda(cooldown_steps: int) -> Callable[[int], float]:
    """Decay LR from 1 → 0 over cooldown_steps using 1 - sqrt(progress)."""
    if cooldown_steps <= 0:
        raise ValueError("cooldown_steps must be a positive integer.")

    def lr_lambda(step: int) -> float:
        if step >= cooldown_steps:
            return 0.0
        progress = (step + 1) / cooldown_steps
        return 1.0 - math.sqrt(progress)

    return lr_lambda


def compute_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for a file without loading it all into memory."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Total L2 norm of gradients (sqrt of sum of per-parameter gradient norms squared)."""
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach().float().norm(2)
        total_norm_sq += float(g) ** 2
    return total_norm_sq ** 0.5


def build_criterion(config: dict) -> nn.Module:
    """Build the training loss from ``config['training']['loss_function']``."""
    training = config["training"]
    name = training["loss_function"]
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "poisson_nll":
        return nn.PoissonNLLLoss(log_input=False, full=False)
    if name == "weighted":
        w = training["weighted_loss"]
        return WeightedLoss(
            loss_type=w["loss_type"],
            threshold=w["threshold"],
            alpha=w["alpha"],
        )
    raise ValueError(f"Unknown loss function: {name}")


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: dict,
    total_train_steps: Optional[int] = None,
) -> Tuple[optim.Optimizer, Optional[Any], float, float, float]:
    """
    Adam with per-module learning rates and optional LR scheduler.

    Returns ``(optimizer, scheduler_or_none, encoder_lr, amp_decoder_lr, ph_decoder_lr)``.
    Unwraps DataParallel / DDP for parameter groups.
    """
    training = config["training"]
    lr = training["learning_rate"]
    encoder_lr = training.get("encoder_lr", lr)
    amp_decoder_lr = training.get("amp_decoder_lr", lr)
    ph_decoder_lr = training.get("ph_decoder_lr", lr)

    core = model.module if isinstance(model, (nn.DataParallel, DDP)) else model
    param_groups = [
        {"params": core.encoder.parameters(), "lr": encoder_lr, "name": "encoder"},
        {"params": core.amp_decoder.parameters(), "lr": amp_decoder_lr, "name": "amp_decoder"},
        {"params": core.ph_decoder.parameters(), "lr": ph_decoder_lr, "name": "ph_decoder"},
    ]
    optimizer = optim.AdamW(param_groups, fused=True, betas=(0.9, 0.95))

    scheduler = None
    lr_sched_cfg = training.get("lr_scheduler", {})
    if lr_sched_cfg.get("enabled", False):
        sched_name = lr_sched_cfg.get("scheduler_class")
        if not sched_name:
            raise ValueError(
                "training.lr_scheduler.scheduler_class must be provided when lr_scheduler.enabled is True."
            )
        sched_kwargs = lr_sched_cfg.get("kwargs", {})
        if not isinstance(sched_kwargs, dict):
            raise ValueError("training.lr_scheduler.kwargs must be a dictionary.")
        sched_kwargs = dict(sched_kwargs)

        sched_name_lower = str(sched_name).lower()
        if sched_name_lower == "warmup-stable":
            warmup_steps = sched_kwargs.get("warmup_steps", 0)
            lr_fn = make_warmup_stable_lr_lambda(warmup_steps)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
            if torch.distributed.get_rank() == 0:
                print(f"warmup-stable scheduler: warmup_steps={warmup_steps}", flush=True)
        elif sched_name_lower == "cooldown":
            cooldown_steps = sched_kwargs.get("cooldown_steps")
            if cooldown_steps is None:
                raise ValueError("cooldown scheduler requires 'cooldown_steps' in lr_scheduler.kwargs.")
            lr_fn = make_cooldown_lr_lambda(cooldown_steps)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
            if torch.distributed.get_rank() == 0:
                print(f"cooldown scheduler: cooldown_steps={cooldown_steps}", flush=True)
        else:
            sched_cls = getattr(optim.lr_scheduler, sched_name, None)
            if sched_cls is None:
                raise ValueError(f"Unknown lr scheduler class: {sched_name}")
            scheduler = sched_cls(optimizer=optimizer, **sched_kwargs)

    return optimizer, scheduler, encoder_lr, amp_decoder_lr, ph_decoder_lr


def computedistancematrix(patch_size: int,
                          num_patches: int,
                          length: int)->np.array:
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j: 
                continue 
            xi, yi = i // length, i % length
            xj, yj = j // length, j % length
            distance_matrix[i, j] = patch_size * np.linalg.norm([xj - xi, yj - yi])
    return distance_matrix

def computemeanattentiondistance(patch_size: int, 
                                 attention_weights: np.array)->np.array:
    attention_weights = attention_weights[..., 1:, 1:]
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    distance_matrix = computedistancematrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(mean_distances, axis=-1)
    mean_distances = np.mean(mean_distances, axis=-1)
    return mean_distances

def computemads(attention_scores: torch.tensor, 
                patch_size: int)->list:
    all_mads = [computemeanattentiondistance(patch_size, attention_weight.numpy()) for attention_weight in attention_scores]
    return all_mads

def visualize_mads(all_mads: np.array, 
                   save_dir:str="./")->None:
    fpath = save_dir + "mads.pdf"
    num_heads = len(all_mads)
    plt.figure(figsize=(6, 6))
    for idx in range(len(all_mads)):
        mean_distance = all_mads[idx]
        x = [idx] * num_heads
        y = mean_distance[0, :]
        plt.scatter(x=x, y=y, label=f"attention_head_{idx}")
        plt.xlabel("Block Index")
        plt.ylabel("Mean Attention Distance")
        plt.legend(loc="lower right")
        plt.savefig(fpath, bbox_inches='tight')
