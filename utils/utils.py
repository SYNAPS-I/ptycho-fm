import hashlib
from typing import Any, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from custom_loss import WeightedLoss


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
    optimizer = optim.AdamW(param_groups, fused=True, betas=(0.9, 0.95), weight_decay=1E-4)

    scheduler = None
    lr_sched_cfg = training.get("lr_scheduler", {})
    if lr_sched_cfg.get("enabled", False):
        sched_name = lr_sched_cfg.get("scheduler_class")
        if not sched_name:
            raise ValueError(
                "training.lr_scheduler.scheduler_class must be provided when lr_scheduler.enabled is True."
            )
        sched_cls = getattr(torch.optim.lr_scheduler, sched_name, None)
        if sched_cls is None:
            raise ValueError(f"Unknown lr scheduler class: {sched_name}")
        sched_kwargs = lr_sched_cfg.get("kwargs", {})
        if not isinstance(sched_kwargs, dict):
            raise ValueError("training.lr_scheduler.kwargs must be a dictionary.")
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
