import os
from typing import Literal
import argparse
import pickle
import yaml
import socket

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
from torch.utils.data import DataLoader, random_split, DistributedSampler
from prefetcher import CUDAPrefetcher

import wandb

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
full_dataset = CombinedDataset(
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

# Split into train and validation using PyTorch's random_split
# This ensures mutually exclusive splits and follows PyTorch best practices
train_split = config['data']['train_split']
total_size = len(full_dataset)
train_size = int(total_size * train_split)
val_size = total_size - train_size

generator = torch.Generator().manual_seed(config['data']['random_seed'])
train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

# ────────────────────────────────────────────────────────────────────────────────
# Distributed Data Loading Strategy
# ────────────────────────────────────────────────────────────────────────────────
sharding_strategy = config['data'].get('sharding_strategy', 'static')

if sharding_strategy == 'static':
    # Static sharding: Use RankShardedSubset, each rank gets fixed samples across all epochs
    if is_main_process:
        print("\nUsing STATIC sharding (RankShardedSubset)", flush=True)
        print("  - Each rank processes fixed samples across all epochs", flush=True)

    train_dataset = RankShardedSubset(train_subset, rank, world_size, debug=DEBUG_MODE, subset_type='train')
    val_dataset = RankShardedSubset(val_subset, rank, world_size, debug=DEBUG_MODE, subset_type='val')

    # Base DataLoader kwargs
    train_dataloader_kwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': config['data'].get('num_workers', 0),
        'pin_memory': pin_memory,
        'shuffle': True,  # Per-epoch local shuffling within each rank's shard
    }

    val_dataloader_kwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': config['data'].get('num_workers', 0),
        'pin_memory': pin_memory,
        'shuffle': False,
    }

    # Add prefetch settings if using workers
    if train_dataloader_kwargs['num_workers'] > 0:
        train_dataloader_kwargs['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
        train_dataloader_kwargs['persistent_workers'] = config['data'].get('persistent_workers', False)

    if val_dataloader_kwargs['num_workers'] > 0:
        val_dataloader_kwargs['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
        val_dataloader_kwargs['persistent_workers'] = config['data'].get('persistent_workers', False)

    train_loader = DataLoader(train_dataset, **train_dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **val_dataloader_kwargs)

    # No samplers needed - set to None for later reference
    train_sampler = None
    val_sampler = None

elif sharding_strategy == 'dynamic':
    # Dynamic sharding: Use DistributedSampler, each rank gets different samples each epoch (better diversity)
    if is_main_process:
        print("\nUsing DYNAMIC sharding (DistributedSampler)", flush=True)
        print("  - Each rank sees different samples each epoch", flush=True)

    # Use the subsets directly (no RankShardedSubset wrapper)
    train_dataset = train_subset
    val_dataset = val_subset

    # Create DistributedSamplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,  # Global shuffle + dynamic sharding
        seed=config['data']['random_seed'],
        drop_last=False
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,  # No shuffle for validation
        seed=config['data']['random_seed'],
        drop_last=False
    )

    # Base DataLoader kwargs (NO shuffle when using sampler!)
    train_dataloader_kwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': config['data'].get('num_workers', 0),
        'pin_memory': pin_memory,
        'sampler': train_sampler,  # Sampler handles sharding and shuffling
        'shuffle': False,  # MUST be False when sampler is provided
    }

    val_dataloader_kwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': config['data'].get('num_workers', 0),
        'pin_memory': pin_memory,
        'sampler': val_sampler,
        'shuffle': False,
    }

    # Add prefetch settings if using workers
    if train_dataloader_kwargs['num_workers'] > 0:
        train_dataloader_kwargs['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
        train_dataloader_kwargs['persistent_workers'] = config['data'].get('persistent_workers', False)

    if val_dataloader_kwargs['num_workers'] > 0:
        val_dataloader_kwargs['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
        val_dataloader_kwargs['persistent_workers'] = config['data'].get('persistent_workers', False)

    train_loader = DataLoader(train_dataset, **train_dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **val_dataloader_kwargs)

else:
    raise ValueError(f"Invalid sharding_strategy: {sharding_strategy}. Must be 'static' or 'dynamic'")

# Wrap loaders with CUDAPrefetcher for async data transfer
# Only use prefetcher if CUDA is available AND enabled in config
use_cuda_prefetcher = config['data'].get('use_cuda_prefetcher', True)
if torch.cuda.is_available() and use_cuda_prefetcher:
    train_prefetcher = CUDAPrefetcher(train_loader, DEVICE)
    val_prefetcher = CUDAPrefetcher(val_loader, DEVICE)
    if is_main_process:
        print("Using CUDAPrefetcher for async data transfer", flush=True)
else:
    # Fallback to regular loaders (training.py handles device transfers)
    train_prefetcher = train_loader
    val_prefetcher = val_loader
    if is_main_process:
        if torch.cuda.is_available():
            print("Using standard DataLoader (CUDAPrefetcher disabled, manual device transfers)", flush=True)
        else:
            print("Using standard DataLoader on CPU", flush=True)

# Create test dataset and loader only on main process
if is_main_process:
    test_dataset = PtychographyDataset(
        file_path=config['data']['test_path'],
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
    print(f"Number of paired files: {len(full_dataset.file_paths)}", flush=True)
    print(f"Train patterns (this rank): {len(train_dataset)} | Val patterns (this rank): {len(val_dataset)}", flush=True)
    print(f"Total batches/epoch (train): {len(train_loader)}", flush=True)
    print(f"Total batches/epoch (val): {len(val_loader)}", flush=True)
    print("\nDataLoader Settings:", flush=True)
    print(f"  num_workers: {train_dataloader_kwargs['num_workers']}", flush=True)
    print(f"  pin_memory: {train_dataloader_kwargs['pin_memory']}", flush=True)
    print(f"  train shuffle: {train_dataloader_kwargs['shuffle']} (per-epoch local shuffling)", flush=True)
    print(f"  val shuffle: {val_dataloader_kwargs['shuffle']}", flush=True)
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
    summary(model, input_data={'x': dummy_data, 'probe': dummy_probe,
            'normalization': torch.randn((1, 1)), 'scale': torch.randn((1, 1))}, device='cpu')

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
if FINETUNE_PATH:
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

if is_main_process:
    print(f"\nOptimizer learning rates:", flush=True)
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
    
trainer = Trainer(
    model,
    MODE,
    config['trainer']['run_num'],
    DEVICE,
    MODEL_SAVE_PATH,
    is_main_process=is_main_process,
    use_ddp=(world_size > 1),
    wandb_enabled=config['wandb']['enabled']
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
        start_epoch, metrics, optimizer, wandb_run_id, _ = trainer.load_state_checkpoint(optimizer, scheduler=None)

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
                "model_config": config['model']
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

if is_main_process:
    print('\nStarting Training...\n', flush=True)

# ────────────────────────────────────────────────────────────────────────────────
# Train / Validate
# ────────────────────────────────────────────────────────────────────────────────
try:
    for epoch in range(start_epoch, EPOCHS):
        # Set epoch for DistributedSampler (dynamic sharding only)
        if sharding_strategy == 'dynamic' and train_sampler is not None:
            train_sampler.set_epoch(epoch)
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
                if hasattr(full_dataset, 'debug_call_count'):
                    full_dataset.debug_call_count = 0

        # Training loop
        model.train()
        trainer.train(train_prefetcher, criterion, optimizer, metrics)

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
        trainer.validate(val_prefetcher, criterion, optimizer, metrics, plot=plot, epoch=epoch)

        # Generate test plot only on main process
        if epoch % config['training']['test_plot_freq'] == 0 and is_main_process:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating test plot...", flush=True)
            trainer.generate_test_plot(test_loader, epoch, 'test_epoch' + str(epoch) + '.png')

        # Optional per-epoch saving
        do_checkpoint = checkpoint_freq > 0 and ((epoch + 1) % checkpoint_freq == 0)
        if do_checkpoint:
            if dist.is_initialized():
                dist.barrier()
            trainer.save_model_and_states_checkpoint(epoch, metrics, optimizer, wandb_run_id, scheduler=None)
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
    trainer.save_model_and_states_checkpoint(epoch, metrics, optimizer, wandb_run_id, scheduler=None)
    run_path = os.path.join(MODEL_SAVE_PATH, 'run' + str(config['trainer']['run_num']))
    with open(os.path.join(run_path, 'metrics.pickle'), 'wb') as file:
        pickle.dump(metrics, file)
    print('\nFinished Training!', flush=True)

if is_main_process and config['wandb']['enabled']:
    wandb.finish() 
