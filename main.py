import os
from typing import Literal
import argparse
import pickle
import yaml
import socket
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.distributed as dist
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from data import PtychographyDataset, CombinedDataset, RankShardedSubset
from model.model import PtychoViT
from custom_loss import WeightedLoss
from training import Trainer
from torch.utils.data import DataLoader, random_split, DistributedSampler, Subset
from prefetcher import CUDAPrefetcher
from utils.utils import compute_sha256

import wandb
from mlflow_logger import MLflowLogger

def resolve_config_path(config_path):
    """Resolve configuration path (relative to script directory)."""
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    return config_path

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    config_path = resolve_config_path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if config.get('trainer', {}).get('run_num') is None:
        # Ensure run folder name is "run_yyyymmdd_hhmmss"
        config.setdefault('trainer', {})['run_num'] = datetime.now().strftime("_%Y%m%d_%H%M%S")
    return config

# Parse command-line arguments
parser = argparse.ArgumentParser(description='PtychoViT Training Script')
parser.add_argument(
    '--config',
    default='config.yaml',
    help='Path to config YAML file (default: config.yaml relative to script)',
)
parser.add_argument('--debug', action='store_true', 
                    help='Enable debug logging to verify CSV usage and shuffling')
args = parser.parse_args()
DEBUG_MODE = args.debug

# ────────────────────────────────────────────────────────────────────────────────
# Data subsetting schedule helpers
# ────────────────────────────────────────────────────────────────────────────────
def _validate_data_subsetting_schedule(schedule_cfg):
    if schedule_cfg is None:
        return None
    enabled = bool(schedule_cfg.get('enabled', False))
    if not enabled:
        return None

    epochs = schedule_cfg.get('epochs')
    fractions = schedule_cfg.get('fractions')
    if not isinstance(epochs, list) or not isinstance(fractions, list):
        raise ValueError("training.data_subsetting_schedule.epochs and fractions must be lists.")
    if len(epochs) == 0 or len(fractions) == 0 or len(epochs) != len(fractions):
        raise ValueError("training.data_subsetting_schedule.epochs and fractions must be non-empty and the same length.")
    if epochs[0] != 0:
        raise ValueError("training.data_subsetting_schedule.epochs must start with 0.")
    if any(not isinstance(e, int) or e < 0 for e in epochs):
        raise ValueError("training.data_subsetting_schedule.epochs must be a list of non-negative integers.")
    if any(epochs[i] >= epochs[i + 1] for i in range(len(epochs) - 1)):
        raise ValueError("training.data_subsetting_schedule.epochs must be strictly increasing.")
    if any(not isinstance(f, (int, float)) or not (0.0 < float(f) <= 1.0) for f in fractions):
        raise ValueError("training.data_subsetting_schedule.fractions must be floats in (0, 1].")

    return {
        'epochs': epochs,
        'fractions': [float(f) for f in fractions],
    }

def _fraction_for_epoch(schedule, epoch):
    # schedule is validated; pick the last fraction whose epoch <= current epoch
    idx = 0
    for i, start_epoch in enumerate(schedule['epochs']):
        if start_epoch <= epoch:
            idx = i
        else:
            break
    return schedule['fractions'][idx]

def _subset_training_subset(train_subset_base, fraction):
    total = len(train_subset_base)
    subset_size = int(total * fraction)
    if subset_size < 1:
        raise ValueError(
            f"training.data_subsetting_schedule fraction {fraction} results in 0 samples. "
            "Increase the fraction or use a larger dataset."
        )
    if hasattr(train_subset_base, 'indices'):
        base_indices = train_subset_base.indices
        if isinstance(base_indices, torch.Tensor):
            base_indices = base_indices.tolist()
        return Subset(train_subset_base.dataset, base_indices[:subset_size])
    return Subset(train_subset_base, list(range(subset_size)))

def _build_train_loader(
    fraction: float,
    train_subset_base: Subset,
    sharding_strategy: str,
    rank: int,
    world_size: int,
    debug_mode: bool,
    train_dataloader_kwargs_base: dict,
    random_seed: int,
    device: torch.device,
    use_cuda_prefetcher: bool,
    drop_last: bool,
):
    train_subset_epoch = _subset_training_subset(train_subset_base, fraction)

    if sharding_strategy == 'static':
        train_dataset = RankShardedSubset(
            train_subset_epoch,
            rank,
            world_size,
            debug=debug_mode,
            subset_type='train'
        )
        train_dataloader_kwargs = train_dataloader_kwargs_base.copy()
        train_loader = DataLoader(train_dataset, **train_dataloader_kwargs)
        train_sampler = None
    elif sharding_strategy == 'dynamic':
        train_dataset = train_subset_epoch
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=random_seed,
            drop_last=drop_last,
        )
        train_dataloader_kwargs = train_dataloader_kwargs_base.copy()
        train_dataloader_kwargs['sampler'] = train_sampler
        train_dataloader_kwargs['shuffle'] = False
        train_loader = DataLoader(train_dataset, **train_dataloader_kwargs)
    else:
        raise ValueError(
            f"Invalid sharding_strategy: {sharding_strategy}. Must be 'static' or 'dynamic'"
        )

    if torch.cuda.is_available() and use_cuda_prefetcher:
        train_prefetcher = CUDAPrefetcher(train_loader, device)
    else:
        train_prefetcher = train_loader

    return (
        train_dataset,
        train_loader,
        train_sampler,
        train_prefetcher,
        train_subset_epoch,
        train_dataloader_kwargs,
    )

def _build_val_loader(
    val_subset: Subset,
    sharding_strategy: str,
    rank: int,
    world_size: int,
    debug_mode: bool,
    val_dataloader_kwargs_base: dict,
    random_seed: int,
    device: torch.device,
    use_cuda_prefetcher: bool,
    drop_last: bool,
):
    if sharding_strategy == 'static':
        val_dataset = RankShardedSubset(
            val_subset,
            rank,
            world_size,
            debug=debug_mode,
            subset_type='val'
        )
        val_sampler = None
        val_dataloader_kwargs = val_dataloader_kwargs_base.copy()
        val_loader = DataLoader(val_dataset, **val_dataloader_kwargs)
    elif sharding_strategy == 'dynamic':
        val_dataset = val_subset
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=random_seed,
            drop_last=drop_last,
        )
        val_dataloader_kwargs = val_dataloader_kwargs_base.copy()
        val_dataloader_kwargs['sampler'] = val_sampler
        val_dataloader_kwargs['shuffle'] = False
        val_loader = DataLoader(val_dataset, **val_dataloader_kwargs)
    else:
        raise ValueError(
            f"Invalid sharding_strategy: {sharding_strategy}. Must be 'static' or 'dynamic'"
        )

    if torch.cuda.is_available() and use_cuda_prefetcher:
        val_prefetcher = CUDAPrefetcher(val_loader, device)
    else:
        val_prefetcher = val_loader

    return val_dataset, val_loader, val_sampler, val_prefetcher, val_dataloader_kwargs

# ────────────────────────────────────────────────────────────────────────────────
# Minimal, robust distributed setup for SLURM or torchrun
# ────────────────────────────────────────────────────────────────────────────────
def ensure_env_from_polaris():
    """Populate torchrun-style env vars from SLURM or MPI if missing."""
    if MPI is None:
        raise ImportError("MPI is not installed. Please install MPI to use this function.")
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    local_rank = os.environ['PMI_LOCAL_RANK'] if 'PMI_LOCAL_RANK' in os.environ else rank % 4
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
        

def init_distributed(platform: str = Literal['polaris', 'slurm']):
    if platform == 'polaris':
        return init_distributed_polaris()
    elif platform == 'slurm':
        return init_distributed_slurm()
    else:
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

    dist.init_process_group('nccl', init_method='env://')
    
    if torch.cuda.is_available():
        nvis = torch.cuda.device_count()
        # For single-process (world_size=1), always use GPU 0
        # For distributed training, map local_rank to available GPUs
        if world_size == 1:
            mapped_local = 0
        else:
            mapped_local = 0 if nvis == 1 else (local_rank_env % nvis)
        torch.cuda.set_device(mapped_local)
        device = torch.device(f"cuda:{mapped_local}")
        os.environ["LOCAL_RANK"] = str(mapped_local)  # keep downstream code consistent
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

    # Set MASTER_ADDR and MASTER_PORT if not already set
    if "MASTER_ADDR" not in os.environ:
        if "SLURM_JOB_NODELIST" in os.environ:
            import subprocess
            import socket
            nodelist = os.environ["SLURM_JOB_NODELIST"]
            try:
                result = subprocess.run(
                    ["scontrol", "show", "hostnames", nodelist],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    first_node = result.stdout.strip().split('\n')[0]
                    os.environ["MASTER_ADDR"] = first_node
                else:
                    os.environ["MASTER_ADDR"] = socket.gethostname()
            except Exception:
                os.environ["MASTER_ADDR"] = socket.gethostname()
        else:
            import socket
            os.environ["MASTER_ADDR"] = socket.gethostname()
    
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    if world_size > 1 and not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="nccl", init_method="env://")

    # Map LOCAL_RANK to a valid CUDA device index after any device masking.
    # This properly handles CUDA_VISIBLE_DEVICES set by SLURM
    if torch.cuda.is_available():
        nvis = torch.cuda.device_count()
        # For single-process (world_size=1), always use GPU 0
        # For distributed training, map local_rank to available GPUs
        if world_size == 1:
            mapped_local = 0
        else:
            mapped_local = 0 if nvis == 1 else (local_rank_env % nvis)
        torch.cuda.set_device(mapped_local)
        device = torch.device(f"cuda:{mapped_local}")
        os.environ["LOCAL_RANK"] = str(mapped_local)  # keep downstream code consistent
    else:
        mapped_local = 0
        device = torch.device("cpu")

    return rank_env, world_size, mapped_local, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()

# Load configuration
config_path = resolve_config_path(args.config)
config = load_config(config_path)

# Training parameters
MODE = config['training']['mode']
BATCH_SIZE = config['training']['batch_size']
LR = config['training']['learning_rate']
EPOCHS = config['training']['epochs']
MODEL_SAVE_PATH = config['paths']['model_save_path']
FINETUNE_PATH = config['training'].get('finetune_from_model')
data_subsetting_schedule = _validate_data_subsetting_schedule(
    config['training'].get('data_subsetting_schedule')
)
skip_batch_if_grad_norm_greater_than = config['training'].get('skip_batch_if_grad_norm_greater_than')
if skip_batch_if_grad_norm_greater_than is not None:
    skip_batch_if_grad_norm_greater_than = float(skip_batch_if_grad_norm_greater_than)
    if skip_batch_if_grad_norm_greater_than <= 0:
        raise ValueError("training.skip_batch_if_grad_norm_greater_than must be > 0 or null.")

# Saving / checkpointing
save_epoch_models = bool(config['training'].get('save_epoch_models', False))
checkpoint_freq = int(config['training'].get('checkpoint_freq', 0))  # 0 disables per-epoch checkpoint.state saving


# ────────────────────────────────────────────────────────────────────────────────
# Distributed init (Code A style)
# ────────────────────────────────────────────────────────────────────────────────
rank, world_size, local_rank, DEVICE = init_distributed(config['training'].get('platform', 'slurm'))
is_main_process = rank == 0

# Normalize DataLoader pinned-memory usage.
# Pinned memory speeds up non_blocking H2D copies, but it uses CUDA's caching host allocator
# and can trigger `CUDACachingHostAllocatorImpl::record_stream` crashes on some systems.
pin_memory = bool(config['data'].get('pin_memory', True))
if pin_memory and not (torch.cuda.is_available() and config['data'].get('use_cuda_prefetcher', True)):
    pin_memory = False
    config['data']['pin_memory'] = False
    if is_main_process:
        print("Disabling DataLoader pin_memory because use_cuda_prefetcher is False (avoids CUDA pinned-host allocator crashes).", flush=True)

print(
    f"[{socket.gethostname()}] WORLD_SIZE={world_size} RANK={rank} "
    f"LOCAL_RANK={local_rank} device={DEVICE}",
    flush=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# Dataset & Dataloaders
# ────────────────────────────────────────────────────────────────────────────────
if 'data_path' not in config['data']:
    raise ValueError("Config must specify 'data_path' (directory containing paired HDF5 files)")

data_dir = config['data']['data_path']

# Create full dataset with sequential indices
# Shuffling is handled by random_split with a deterministic seed
base_dataset = CombinedDataset(
    file_paths=data_dir,
    rank=rank,
    world_size=world_size,
    scale=config['data']['scale'],
    normalization_dict_path=config['data'].get('normalization_dict_path'),
    default_normalization=config['data'].get('default_normalization', 100000.0),
    apply_noise=config['data'].get('apply_noise', True),
    cache_object=config['data'].get('cache_object', False),
    max_probe_modes=config['data'].get('max_probe_modes', 8),
    target_size=config['data'].get('target_size', 256),
    max_files=config['data'].get('max_files'),
    debug=DEBUG_MODE
)

# Optionally take only the first N samples from the full dataset
full_dataset = base_dataset
total_size = len(base_dataset)
subset_fraction = config['data'].get('subset_fraction')
if subset_fraction is not None:
    if not (0.0 < subset_fraction <= 1.0):
        raise ValueError("Config 'data.subset_fraction' must be in the range (0, 1].")
    subset_size = int(total_size * subset_fraction)
    if subset_size < 1:
        raise ValueError(
            f"Config 'data.subset_fraction'={subset_fraction} results in 0 samples. "
            "Increase subset_fraction or use a larger dataset."
        )
    full_dataset = Subset(base_dataset, list(range(subset_size)))
    total_size = subset_size
    if is_main_process:
        print(
            f"Using subset_fraction={subset_fraction} -> {subset_size}/{len(base_dataset)} samples",
            flush=True,
        )

# Split into train and validation using PyTorch's random_split
# This ensures mutually exclusive splits and follows PyTorch best practices
train_split = config['data']['train_split']
train_size = int(total_size * train_split)
val_size = total_size - train_size

generator = torch.Generator().manual_seed(config['data']['random_seed'])
train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
train_subset_base = train_subset

# ────────────────────────────────────────────────────────────────────────────────
# Distributed Data Loading Strategy
# ────────────────────────────────────────────────────────────────────────────────
sharding_strategy = config['data'].get('sharding_strategy', 'static')
drop_last = config['data'].get('drop_last', False)
if sharding_strategy == 'static':
    if is_main_process:
        print("\nUsing STATIC sharding (RankShardedSubset)", flush=True)
        print("  - Each rank processes fixed samples across all epochs", flush=True)
elif sharding_strategy == 'dynamic':
    if is_main_process:
        print("\nUsing DYNAMIC sharding (DistributedSampler)", flush=True)
        print("  - Each rank sees different samples each epoch", flush=True)
else:
    raise ValueError(f"Invalid sharding_strategy: {sharding_strategy}. Must be 'static' or 'dynamic'")

if (
    is_main_process
    and config['training'].get('platform', 'slurm') == 'polaris'
    and not drop_last
):
    print(
        "WARNING: data.drop_last is False on Polaris. Distributed training may crash during synchronization when ranks have uneven batch counts.",
        flush=True,
    )

# Base DataLoader kwargs
train_dataloader_kwargs_base = {
    'batch_size': BATCH_SIZE,
    'num_workers': config['data'].get('num_workers', 0),
    'pin_memory': pin_memory,
    'shuffle': True,
    'drop_last': drop_last,
}

val_dataloader_kwargs_base = {
    'batch_size': BATCH_SIZE,
    'num_workers': config['data'].get('num_workers', 0),
    'pin_memory': pin_memory,
    'shuffle': False,
    'drop_last': drop_last,
}

# Add prefetch settings if using workers
if train_dataloader_kwargs_base['num_workers'] > 0:
    train_dataloader_kwargs_base['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
    train_dataloader_kwargs_base['persistent_workers'] = config['data'].get('persistent_workers', False)

if val_dataloader_kwargs_base['num_workers'] > 0:
    val_dataloader_kwargs_base['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
    val_dataloader_kwargs_base['persistent_workers'] = config['data'].get('persistent_workers', False)

# Only use prefetcher if CUDA is available AND enabled in config
use_cuda_prefetcher = config['data'].get('use_cuda_prefetcher', True)

# Build initial train/val loaders (full training set)
(
    train_dataset,
    train_loader,
    train_sampler,
    train_prefetcher,
    _train_subset_epoch,
    train_dataloader_kwargs,
) = _build_train_loader(
    fraction=1.0,
    train_subset_base=train_subset_base,
    sharding_strategy=sharding_strategy,
    rank=rank,
    world_size=world_size,
    debug_mode=DEBUG_MODE,
    train_dataloader_kwargs_base=train_dataloader_kwargs_base,
    random_seed=config['data']['random_seed'],
    device=DEVICE,
    use_cuda_prefetcher=use_cuda_prefetcher,
    drop_last=drop_last,
)

(
    val_dataset,
    val_loader,
    val_sampler,
    val_prefetcher,
    val_dataloader_kwargs,
) = _build_val_loader(
    val_subset=val_subset,
    sharding_strategy=sharding_strategy,
    rank=rank,
    world_size=world_size,
    debug_mode=DEBUG_MODE,
    val_dataloader_kwargs_base=val_dataloader_kwargs_base,
    random_seed=config['data']['random_seed'],
    device=DEVICE,
    use_cuda_prefetcher=use_cuda_prefetcher,
    drop_last=drop_last,
)

if is_main_process:
    if torch.cuda.is_available() and use_cuda_prefetcher:
        print("Using CUDAPrefetcher for async data transfer", flush=True)
    elif torch.cuda.is_available():
        print("Using standard DataLoader (CUDAPrefetcher disabled, manual device transfers)", flush=True)
    else:
        print("Using standard DataLoader on CPU", flush=True)

# Create test dataset and loader only on main process when test_path is configured
if is_main_process and config['data'].get('test_path') is not None:
    test_dataset = PtychographyDataset(
        file_path=config['data'].get('test_path'),
        scale=config['data']['scale'],
        normalization_dict_path=config['data'].get('test_normalization'),
        apply_noise=config['data'].get('apply_noise', False),  # Don't add noise to test data
        default_normalization=config['data'].get('default_normalization', 100000.0),
        max_probe_modes=config['data'].get('max_probe_modes', 8),
        target_size=config['data'].get('target_size', 256),
        object_name=config['data'].get('test_dataset_object_name', None),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=pin_memory
    )

    print(f"Test dataset: {len(test_dataset)} patterns", flush=True)
else:
    test_loader = None
    if is_main_process:
        print("Test plotting disabled: data.test_path is null", flush=True)

# Print configuration only on main process
if is_main_process:
    print("=" * 50, flush=True)
    print("Training Configuration", flush=True)
    print("=" * 50, flush=True)
    print(f"Mode: {MODE}")
    print(f"Batch size: {BATCH_SIZE} | Learning rate: {LR}", flush=True)
    print(f"Epochs: {EPOCHS} | World Size (GPUs): {world_size}", flush=True)
    print(f"Loss function: {config['training']['loss_function']}", flush=True)
    print(f"Data directory: {data_dir}", flush=True)
    print(f"Number of paired files: {len(base_dataset.file_paths)}", flush=True)
    print(f"Total patterns (after subset_fraction): {len(full_dataset)}", flush=True)
    print(f"Train patterns (this rank): {len(train_dataset)} | Val patterns (this rank): {len(val_dataset)}", flush=True)
    print(f"Total batches/epoch (train): {len(train_loader)}", flush=True)
    print(f"Total batches/epoch (val): {len(val_loader)}", flush=True)
    print("\nDataLoader Settings:", flush=True)
    print(f"  num_workers: {train_dataloader_kwargs['num_workers']}", flush=True)
    print(f"  pin_memory: {train_dataloader_kwargs['pin_memory']}", flush=True)
    print(f"  train shuffle: {train_dataloader_kwargs['shuffle']} (per-epoch local shuffling)", flush=True)
    print(f"  val shuffle: {val_dataloader_kwargs['shuffle']}", flush=True)
    print(f"  drop_last: {train_dataloader_kwargs['drop_last']}", flush=True)
    if train_dataloader_kwargs['num_workers'] > 0:
        print(f"  prefetch_factor: {train_dataloader_kwargs.get('prefetch_factor', 'N/A')}", flush=True)
        print(f"  persistent_workers: {train_dataloader_kwargs.get('persistent_workers', 'N/A')}", flush=True)
    use_prefetcher_status = torch.cuda.is_available() and config['data'].get('use_cuda_prefetcher', True)
    print(f"  Using CUDAPrefetcher: {use_prefetcher_status}", flush=True)
    print(f"\nDevice: {DEVICE}", flush=True)
    print(f"Model save path: {MODEL_SAVE_PATH}", flush=True)
    print("=" * 50, flush=True)

# ────────────────────────────────────────────────────────────────────────────────
# Model setup - All models are 256x256
# ────────────────────────────────────────────────────────────────────────────────
img_size = 256

# Use unified PtychoViT model with encoder_type selection
model = PtychoViT(config=config['model'])

if is_main_process:
    encoder_type = config['model'].get('encoder_type', 'custom')
    print(f"Using PtychoViT with {encoder_type.upper()} encoder", flush=True)
    dummy_data = torch.randn((1, 1, img_size, img_size))
    dummy_probe = torch.randn((1, 1, 8, img_size, img_size, 2))
    try:
        summary(
            model,
            input_data={
                'x': dummy_data,
                'probe': dummy_probe,
                'normalization': torch.randn((1, 1)),
                'scale': torch.randn((1, 1)),
            },
            device='cpu',
        )
    except Exception as e:
        print(f"[Warning] torchinfo summary failed and will be skipped: {e}", flush=True)

# Move model to device and wrap with DDP (multinode.py approach)
model = model.to(DEVICE)
if world_size > 1:
    # Use torch.cuda.current_device() like multinode.py does
    # When CUDA_VISIBLE_DEVICES is set by SLURM, don't pass device_ids to avoid NCCL PCI bus ID lookup
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        # SLURM sets CUDA_VISIBLE_DEVICES - let DDP auto-detect to avoid PCI bus ID issues
        model = DDP(model, find_unused_parameters=False)
    else:
        # For torchrun or other launchers, explicitly specify device
        dev_index = torch.cuda.current_device()
        model = DDP(model, device_ids=[dev_index], output_device=dev_index, find_unused_parameters=False, gradient_as_bucket_view=False)

# Load pretrained weights for finetuning (fresh optimizer state)
finetune_checkpoint_sha256 = None
if FINETUNE_PATH:
    if is_main_process:
        finetune_checkpoint_sha256 = compute_sha256(FINETUNE_PATH)
    state = torch.load(FINETUNE_PATH, map_location=DEVICE)
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    if is_main_process:
        print(f"Loaded finetune weights from {FINETUNE_PATH}; optimizer will start fresh.", flush=True)        

# ────────────────────────────────────────────────────────────────────────────────
# Loss, optimizer, metrics, trainer
# ────────────────────────────────────────────────────────────────────────────────
if config['training']['loss_function'] == 'smooth_l1':
    criterion = nn.SmoothL1Loss()
elif config['training']['loss_function'] == 'mse':
    criterion = nn.MSELoss()
elif config['training']['loss_function'] == 'l1':
    criterion = nn.L1Loss()
elif config['training']['loss_function'] == 'poisson_nll':
    criterion = nn.PoissonNLLLoss(log_input=False, full=False)
elif config['training']['loss_function'] == 'weighted':
    weighted_loss_config = config['training']['weighted_loss']
    criterion = WeightedLoss(loss_type=weighted_loss_config['loss_type'], threshold=weighted_loss_config['threshold'], alpha=weighted_loss_config['alpha'])
else:
    raise ValueError(f"Unknown loss function: {config['training']['loss_function']}")

# Get individual learning rates from config (fallback to default LR if not specified)
encoder_lr = config['training'].get('encoder_lr', LR)
amp_decoder_lr = config['training'].get('amp_decoder_lr', LR)
ph_decoder_lr = config['training'].get('ph_decoder_lr', LR)

# Get the actual model (unwrap DDP if needed)
actual_model = model.module if isinstance(model, DDP) else model

# Create parameter groups with individual learning rates
param_groups = [
    {'params': actual_model.encoder.parameters(), 'lr': encoder_lr, 'name': 'encoder'},
    {'params': actual_model.amp_decoder.parameters(), 'lr': amp_decoder_lr, 'name': 'amp_decoder'},
    {'params': actual_model.ph_decoder.parameters(), 'lr': ph_decoder_lr, 'name': 'ph_decoder'}
]

optimizer = optim.Adam(param_groups)

# Optional learning rate scheduler
scheduler = None
lr_sched_cfg = config['training'].get('lr_scheduler', {})
if lr_sched_cfg.get('enabled', False):
    sched_name = lr_sched_cfg.get('scheduler_class')
    if not sched_name:
        raise ValueError("training.lr_scheduler.scheduler_class must be provided when lr_scheduler.enabled is True.")
    sched_cls = getattr(torch.optim.lr_scheduler, sched_name, None)
    if sched_cls is None:
        raise ValueError(f"Unknown lr scheduler class: {sched_name}")
    sched_kwargs = lr_sched_cfg.get('kwargs', {})
    if not isinstance(sched_kwargs, dict):
        raise ValueError("training.lr_scheduler.kwargs must be a dictionary.")
    scheduler = sched_cls(optimizer=optimizer, **sched_kwargs)

if is_main_process:
    print("\nOptimizer learning rates:", flush=True)
    print(f"  Encoder: {encoder_lr}", flush=True)
    print(f"  Amplitude Decoder: {amp_decoder_lr}", flush=True)
    print(f"  Phase Decoder: {ph_decoder_lr}", flush=True)

metrics = {'training_loss': [], 'train_amp_loss': [], 'train_ph_loss': [], 'validation_loss': [],
           'val_amp_loss': [], 'val_ph_loss': [], 'best_val_loss': np.inf}

# Track starting epoch for checkpoint resumption
start_epoch = 0
wandb_run_id = None

# If finetuning weights are provided, skip optimizer state resume to keep optimizer fresh
if FINETUNE_PATH:
    if is_main_process:
        print("finetune_from_model is set; ignoring resume_from_checkpoint to keep optimizer state fresh.", flush=True)
    
mlflow_logger = MLflowLogger(config, is_main_process)

trainer = Trainer(
    model,
    MODE,
    config['trainer']['run_num'],
    DEVICE,
    MODEL_SAVE_PATH,
    is_main_process=is_main_process,
    use_ddp=(world_size > 1),
    wandb_enabled=config['wandb']['enabled'],
    debug_mode=DEBUG_MODE,
    skip_batch_if_grad_norm_greater_than=skip_batch_if_grad_norm_greater_than,
    mlflow_logger=mlflow_logger,
)

# ────────────────────────────────────────────────────────────────────────────────
# Resume from checkpoint if requested
# ────────────────────────────────────────────────────────────────────────────────
if config['training'].get('resume_from_checkpoint', False):
    if is_main_process:
        print('\nResuming from checkpoint...', flush=True)

    # Load model weights
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, 'run' + str(config['trainer']['run_num']))
    model_checkpoint = os.path.join(checkpoint_path, 'checkpoint_model.pth')

    if os.path.exists(model_checkpoint):
        if world_size > 1:
            model.module.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
        else:
            model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))

        # Load optimizer, metrics, and wandb run ID
        start_epoch, metrics, optimizer, wandb_run_id, scheduler = trainer.load_state_checkpoint(optimizer, scheduler=scheduler)

        # Use manually specified run ID if checkpoint doesn't have one (for old checkpoints)
        if wandb_run_id is None and config['wandb'].get('resume_run_id') is not None:
            wandb_run_id = config['wandb']['resume_run_id']
            if is_main_process:
                print(f'Using manually specified wandb run ID: {wandb_run_id}', flush=True)

        if is_main_process:
            print(f'Loaded checkpoint from epoch {start_epoch - 1}', flush=True)
            print(f'Resuming training from epoch {start_epoch}', flush=True)
            if wandb_run_id:
                print(f'Will resume wandb run: {wandb_run_id}', flush=True)
            else:
                print('No wandb run ID found - will create new wandb run', flush=True)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {model_checkpoint}")

# ────────────────────────────────────────────────────────────────────────────────
# Initialize wandb only on main process
# ────────────────────────────────────────────────────────────────────────────────
if is_main_process and config['wandb']['enabled']:
    wandb.login()
    if wandb_run_id is not None:
        # Resume existing wandb run
        run = wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project'],
            id=wandb_run_id,
            resume='must'
        )
        print(f'Resumed wandb run: {wandb_run_id}', flush=True)
    else:
        # Create new wandb run
        run = wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project'],
            config={
                "learning_rate": LR,
                "encoder_lr": encoder_lr,
                "amp_decoder_lr": amp_decoder_lr,
                "ph_decoder_lr": ph_decoder_lr,
                "batch_size": BATCH_SIZE,
                "dataset": config['wandb']['dataset_name'],
                "epochs": EPOCHS,
                "notes": config['wandb']['notes'],
                "encoder_type": config['model'].get('encoder_type', 'custom'),
                "model_config": config['model'],
                "data_config": config['data'],
                "trainer_config": config['trainer'],
                "training_config": config['training'],
                "wandb_config": config['wandb'],
                "model_save_path": config['paths'].get('model_save_path', 'N/A')
            }
        )
        wandb_run_id = run.id
        print(f'Created new wandb run: {wandb_run_id}', flush=True)

    # Upload config.yaml to wandb as artifact at the start of training
    import shutil
    config_copy_path = './config_copy.yaml'
    shutil.copy(config_path, config_copy_path)
    artifact = wandb.Artifact(name="config", type="file")
    artifact.add_file(local_path="config_copy.yaml", name="training_config")
    artifact.save()
    # Delete the copy after wandb saves it
    if os.path.exists(config_copy_path):
        os.remove(config_copy_path)
    print('Uploaded config to wandb as artifact', flush=True)

    if FINETUNE_PATH and finetune_checkpoint_sha256 is not None:
        wandb.run.summary["finetune_checkpoint_sha256"] = finetune_checkpoint_sha256
        print("Logged finetune checkpoint SHA256 to wandb", flush=True)

# ────────────────────────────────────────────────────────────────────────────────
# Log run params and config snapshot to MLflow (rank 0, no-op when disabled)
# ────────────────────────────────────────────────────────────────────────────────
mlflow_logger.log_params({
    "learning_rate": LR,
    "encoder_lr": encoder_lr,
    "amp_decoder_lr": amp_decoder_lr,
    "ph_decoder_lr": ph_decoder_lr,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "loss_function": config['training']['loss_function'],
    "encoder_type": config['model'].get('encoder_type', 'custom'),
    "model": config['model'],
    "data": config['data'],
    "training": config['training'],
    "trainer": config['trainer'],
    "model_save_path": config['paths'].get('model_save_path', 'N/A'),
})
mlflow_logger.log_artifact(config_path)
if FINETUNE_PATH and finetune_checkpoint_sha256 is not None:
    mlflow_logger.log_params({"finetune_checkpoint_sha256": finetune_checkpoint_sha256})

if is_main_process:
    print('\nStarting Training...\n', flush=True)

# ────────────────────────────────────────────────────────────────────────────────
# Train / Validate
# ────────────────────────────────────────────────────────────────────────────────
try:
    for epoch in range(start_epoch, EPOCHS):
        # Optionally apply per-epoch training subsetting schedule
        if data_subsetting_schedule is not None:
            fraction = _fraction_for_epoch(data_subsetting_schedule, epoch)
            (
                train_dataset,
                train_loader,
                train_sampler,
                train_prefetcher,
                train_subset_epoch,
                train_dataloader_kwargs,
            ) = _build_train_loader(
                fraction=fraction,
                train_subset_base=train_subset_base,
                sharding_strategy=sharding_strategy,
                rank=rank,
                world_size=world_size,
                debug_mode=DEBUG_MODE,
                train_dataloader_kwargs_base=train_dataloader_kwargs_base,
                random_seed=config['data']['random_seed'],
                device=DEVICE,
                use_cuda_prefetcher=use_cuda_prefetcher,
                drop_last=drop_last,
            )

            if is_main_process:
                print(
                    f"Data subsetting schedule: epoch {epoch} -> "
                    f"{fraction * 100:.2f}% of training set "
                    f"({len(train_subset_epoch)} samples before sharding)",
                    flush=True,
                )

        # Set epoch for DistributedSampler (dynamic sharding only)
        if sharding_strategy == 'dynamic':
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

        # Save config to run path at epoch 0
        if epoch == 0 and is_main_process:
            trainer.save_config(config_path)
            print('Saved config to run path', flush=True)

        # Log epoch start with timestamp
        if is_main_process:
            from datetime import datetime
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========== Starting Epoch {epoch + 1}/{EPOCHS} ==========", flush=True)
        
        # Debug logging for first epoch
        if DEBUG_MODE and epoch < 3:
            # Select ranks to debug (rank 0, rank 1, and middle rank)
            debug_ranks = [0]
            if world_size > 1:
                debug_ranks.append(1)
            if world_size > 2:
                debug_ranks.append(world_size // 2)
            
            if rank in debug_ranks:
                print(f"[DEBUG Rank {rank}] ========== Epoch {epoch + 1} Debug Mode Active ==========", flush=True)
                print(f"[DEBUG Rank {rank}] Will log first 15 data accesses (covers first ~3-5 batches)", flush=True)
                print(f"[DEBUG Rank {rank}] Training DataLoader: shuffle={train_dataloader_kwargs['shuffle']} (per-epoch local shuffling)", flush=True)
                print(f"[DEBUG Rank {rank}] Validation DataLoader: shuffle={val_dataloader_kwargs['shuffle']}", flush=True)
                # Reset debug counters for new epoch (RankShardedSubset has debug_call_count)
                if hasattr(train_dataset, 'debug_call_count'):
                    train_dataset.debug_call_count = 0
                if hasattr(val_dataset, 'debug_call_count'):
                    val_dataset.debug_call_count = 0
                # Also reset the underlying CombinedDataset debug counter
                if hasattr(base_dataset, 'debug_call_count'):
                    base_dataset.debug_call_count = 0

        # Training loop
        model.train()
        trainer.train(train_prefetcher, criterion, optimizer, metrics, epoch=epoch)

        # Validation loop
        model.eval()
        plot = (epoch % config['training']['validation_plot_freq'] == 0)
        
        # Debug logging for validation in first epoch
        if DEBUG_MODE and epoch == 0:
            debug_ranks = [0]
            if world_size > 1:
                debug_ranks.append(1)
            if world_size > 2:
                debug_ranks.append(world_size // 2)
            if rank in debug_ranks:
                print(f"[DEBUG Rank {rank}] ========== Starting Validation (Epoch {epoch + 1}) ==========", flush=True)
                if hasattr(val_dataset, 'debug_call_count'):
                    val_dataset.debug_call_count = 0  # Reset counter for validation
        
        if is_main_process:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running validation...", flush=True)
        trainer.validate(val_prefetcher, criterion, optimizer, metrics, plot=plot, epoch=epoch, scheduler=scheduler)

        # Generate test plot only on main process
        if epoch % config['training']['test_plot_freq'] == 0 and is_main_process and test_loader is not None:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating test plot...", flush=True)
            trainer.generate_test_plot(
                test_loader,
                epoch,
                'test_epoch' + str(epoch) + '.png',
                central_crop=config['training'].get('test_plot_central_crop', 64),
                ph_crop=config['training'].get('test_plot_ph_crop', 180),
            )

        # Optional per-epoch saving
        do_checkpoint = checkpoint_freq > 0 and ((epoch + 1) % checkpoint_freq == 0)
        if do_checkpoint:
            if dist.is_initialized():
                dist.barrier()
            trainer.save_model_and_states_checkpoint(epoch, metrics, optimizer, wandb_run_id, scheduler=scheduler)
            if dist.is_initialized():
                dist.barrier()
        elif is_main_process and save_epoch_models:
            trainer.update_saved_model(f'model_epoch_{epoch + 1:03d}')

        if is_main_process:
            from datetime import datetime
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========== Completed Epoch {epoch + 1}/{EPOCHS} ==========", flush=True)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch + 1} | Train Loss: {metrics['training_loss'][-1]:.4f} | Val. Loss: {metrics['validation_loss'][-1]:.4f} | Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}", flush=True)
finally:
    cleanup_distributed()

# ────────────────────────────────────────────────────────────────────────────────
# Save final checkpoint only on main process
# ────────────────────────────────────────────────────────────────────────────────
if is_main_process:
    trainer.save_model_and_states_checkpoint(epoch, metrics, optimizer, wandb_run_id, scheduler=scheduler)
    run_path = os.path.join(MODEL_SAVE_PATH, 'run' + str(config['trainer']['run_num']))
    with open(os.path.join(run_path, 'metrics.pickle'), 'wb') as file:
        pickle.dump(metrics, file)
    print('\nFinished Training!', flush=True)

if is_main_process and config['wandb']['enabled']:
    wandb.finish()

# ────────────────────────────────────────────────────────────────────────────────
# Push best model to MLflow registry and close the run (rank 0, no-op when disabled)
# ────────────────────────────────────────────────────────────────────────────────
if is_main_process:
    best_model_path = os.path.join(
        MODEL_SAVE_PATH, 'run' + str(config['trainer']['run_num']), 'best_model.pth'
    )
    if config.get('mlflow', {}).get('register_model', True):
        mlflow_logger.register_best_model(
            best_model_path,
            model_name=config.get('mlflow', {}).get('registered_model_name', 'ptycho-vit'),
        )
    mlflow_logger.finish()

