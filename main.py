import os
# CUDA_VISIBLE_DEVICES should be set by SLURM or environment, not hardcoded
# Uncomment and set if needed for local testing: os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.distributed as dist
import pickle
import yaml

from data import CombinedDataset
from model import PtychoViT, PtychoViT256
from model_cnn import PtychoCNN, PtychoCNN256
from training import Trainer
from torch.utils.data import DataLoader
from prefetcher import CUDAPrefetcher

import wandb

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# ────────────────────────────────────────────────────────────────────────────────
# Minimal, robust distributed setup for SLURM or torchrun
# Based on multinode.py approach
# ────────────────────────────────────────────────────────────────────────────────
def ensure_env_from_slurm():
    """Populate torchrun-style env vars from SLURM if missing."""
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]


def init_distributed():
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
        mapped_local = 0 if nvis == 1 else (local_rank_env % nvis)
        torch.cuda.set_device(mapped_local)
        device = torch.device(f"cuda:{mapped_local}")
        os.environ["LOCAL_RANK"] = str(mapped_local)  # keep downstream code consistent
    else:
        mapped_local = 0
        device = torch.device("cpu")

    return rank_env, world_size, mapped_local, device

# Load configuration
config = load_config()

# Training parameters
MODE = config['training']['mode']
NGPUS = config['training']['ngpus']
BATCH_SIZE = config['training']['batch_size']
LR = config['training']['learning_rate']
EPOCHS = config['training']['epochs']
MODEL_SAVE_PATH = config['paths']['model_save_path']

# Initialize distributed training (if using multiple GPUs)
# Use the robust multinode.py approach for proper device mapping
if NGPUS > 1:
    rank, world_size, local_rank, DEVICE = init_distributed()
    is_main_process = rank == 0
    
    # Keep this print for all ranks - useful for debugging distributed setup
    print(f"Rank: {rank}, Local Rank: {local_rank}, CUDA Device: {DEVICE}, World Size: {world_size}", flush=True)
else:
    rank = 0
    world_size = 1
    local_rank = 0
    is_main_process = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create CombinedDataset with built-in rank-based sharding (Phase 1: Global Indexing)
# Support both 'datafiles' (list) and 'data_path' (directory)
if 'datafiles' in config['data'] and config['data']['datafiles'] is not None:
    data_source = config['data']['datafiles']
elif 'data_path' in config['data']:
    data_source = config['data']['data_path']
else:
    raise ValueError("Config must specify either 'datafiles' (list) or 'data_path' (directory)")

# Create train and validation datasets with built-in sharding
# No need for random_split or DistributedSampler - sharding is handled by dataset
train_dataset = CombinedDataset(
    file_paths=data_source,
    rank=rank,
    world_size=world_size,
    train_split=config['data']['train_split'],
    shuffle=True,
    random_seed=config['data']['random_seed'],
    mode='train',
    scale=config['data']['scale'],
    normalization_dict_path=config['data'].get('normalization_dict_path'),
    apply_noise=True
)

val_dataset = CombinedDataset(
    file_paths=data_source,
    rank=rank,
    world_size=world_size,
    train_split=config['data']['train_split'],
    shuffle=False,  # No shuffling for validation
    random_seed=config['data']['random_seed'],
    mode='val',
    scale=config['data']['scale'],
    normalization_dict_path=config['data'].get('normalization_dict_path'),
    apply_noise=True
)

# Create DataLoaders
# No sampler needed - sharding is built into dataset (Phase 1)
# shuffle=False because shuffling is done in dataset global indexing
dataloader_kwargs = {
    'batch_size': BATCH_SIZE,
    'num_workers': config['data'].get('num_workers', 0),
    'pin_memory': config['data'].get('pin_memory', True),
    'shuffle': False,  # Shuffling handled by dataset
}

# Add prefetch_factor and persistent_workers only if num_workers > 0
if dataloader_kwargs['num_workers'] > 0:
    dataloader_kwargs['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
    dataloader_kwargs['persistent_workers'] = config['data'].get('persistent_workers', False)

train_loader = DataLoader(train_dataset, **dataloader_kwargs)
val_loader = DataLoader(val_dataset, **dataloader_kwargs)

# Wrap loaders with CUDAPrefetcher for async data transfer (Phase 3)
# Only use prefetcher if CUDA is available
if torch.cuda.is_available():
    train_prefetcher = CUDAPrefetcher(train_loader, DEVICE)
    val_prefetcher = CUDAPrefetcher(val_loader, DEVICE)
else:
    # Fallback to regular loaders on CPU
    train_prefetcher = train_loader
    val_prefetcher = val_loader

# Print configuration only on main process
if is_main_process:
    print("=" * 50, flush=True)
    print("Training Configuration", flush=True)
    print("=" * 50, flush=True)
    print(f"Mode: {MODE}")
    print(f"Batch size: {BATCH_SIZE} | Learning rate: {LR}", flush=True)
    print(f"Epochs: {EPOCHS} | GPUs: {NGPUS}", flush=True)
    print(f"Loss function: {config['training']['loss_function']}", flush=True)
    if isinstance(data_source, list):
        print(f"Data source: List of {len(data_source)} file(s)", flush=True)
    else:
        print(f"Data source: {data_source}", flush=True)
    print(f"Number of files: {len(train_dataset.file_paths)}", flush=True)
    print(f"Train patterns (this rank): {len(train_dataset)} | Val patterns (this rank): {len(val_dataset)}", flush=True)
    print(f"Total batches/epoch (train): {len(train_loader)}", flush=True)
    print(f"Total batches/epoch (val): {len(val_loader)}", flush=True)
    print(f"\nDataLoader Settings:", flush=True)
    print(f"  num_workers: {dataloader_kwargs['num_workers']}", flush=True)
    print(f"  pin_memory: {dataloader_kwargs['pin_memory']}", flush=True)
    if dataloader_kwargs['num_workers'] > 0:
        print(f"  prefetch_factor: {dataloader_kwargs.get('prefetch_factor', 'N/A')}", flush=True)
        print(f"  persistent_workers: {dataloader_kwargs.get('persistent_workers', 'N/A')}", flush=True)
    print(f"  Using CUDAPrefetcher: {torch.cuda.is_available()}", flush=True)
    print(f"\nDevice: {DEVICE}", flush=True)
    print(f"Model save path: {MODEL_SAVE_PATH}", flush=True)
    print("=" * 50, flush=True)

# Model setup with config
model_type = config['model'].get('model_type', 'vit')  # Default to 'vit' if not specified
if model_type == 'vit':
    model = PtychoViT(config=config['model']['vit'])
    img_size = config['model']['vit']['encoder']['img_size']
elif model_type == 'cnn':
    model = PtychoCNN(config=config['model']['cnn'])
    img_size = 512  # CNN models are fixed at 512x512
elif model_type == 'cnn256':
    model = PtychoCNN256(config=config['model']['cnn256'])
    img_size = 256  # CNN256 models are fixed at 256x256
elif model_type == 'vit256':
    model = PtychoViT256(config=config['model']['vit256'])
    img_size = 256  # ViT256 models are fixed at 256x256
else:
    raise ValueError(f"Unknown model type: {model_type}. Choose 'vit', 'cnn', 'cnn256', or 'vit256'")

if is_main_process:
    print(f"Using model type: {model_type.upper()}", flush=True)
    dummy_data = torch.randn((1, 1, img_size, img_size))
    dummy_probe = torch.randn((1, 1, 8, img_size, img_size, 2))
    summary(model, input_data={'x': dummy_data, 'probe': dummy_probe,
            'normalization': torch.randn((1, 1)), 'scale': torch.randn((1, 1))}, device='cpu')

# Move model to device and wrap with DDP (multinode.py approach)
model = model.to(DEVICE)
if NGPUS > 1:
    # Use torch.cuda.current_device() like multinode.py does
    # When CUDA_VISIBLE_DEVICES is set by SLURM, don't pass device_ids to avoid NCCL PCI bus ID lookup
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        # SLURM sets CUDA_VISIBLE_DEVICES - let DDP auto-detect to avoid PCI bus ID issues
        model = DDP(model, find_unused_parameters=False)
    else:
        # For torchrun or other launchers, explicitly specify device
        dev_index = torch.cuda.current_device()
        model = DDP(model, device_ids=[dev_index], output_device=dev_index, find_unused_parameters=False)

# Loss and optimizer
if config['training']['loss_function'] == 'smooth_l1':
    criterion = nn.SmoothL1Loss()
elif config['training']['loss_function'] == 'mse':
    criterion = nn.MSELoss()
elif config['training']['loss_function'] == 'l1':
    criterion = nn.L1Loss()
elif config['training']['loss_function'] == 'poisson_nll':
    criterion = nn.PoissonNLLLoss(log_input=False, full=False)
else:
    raise ValueError(f"Unknown loss function: {config['training']['loss_function']}")

optimizer = optim.Adam(model.parameters(), lr=LR)

metrics = {'training_loss': [], 'train_amp_loss': [], 'train_ph_loss': [], 'validation_loss': [],
           'val_amp_loss': [], 'val_ph_loss': [], 'best_val_loss': np.inf}

# Track starting epoch for checkpoint resumption
start_epoch = 0
wandb_run_id = None

trainer = Trainer(
    model,
    MODE,
    config['trainer']['run_num'],
    DEVICE,
    MODEL_SAVE_PATH,
    is_main_process=is_main_process,
    use_ddp=(NGPUS > 1),
    wandb_enabled=config['wandb']['enabled']
)

# Resume from checkpoint if requested
if config['training'].get('resume_from_checkpoint', False):
    if is_main_process:
        print('\nResuming from checkpoint...', flush=True)

    # Load model weights
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, 'run' + str(config['trainer']['run_num']))
    model_checkpoint = os.path.join(checkpoint_path, 'checkpoint_model.pth')

    if os.path.exists(model_checkpoint):
        if NGPUS > 1:
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

# Initialize wandb only on main process
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
                "batch_size": BATCH_SIZE,
                "dataset": config['wandb']['dataset_name'],
                "epochs": EPOCHS,
                "notes": config['wandb']['notes'],
                "model_type": model_type,
                "model_config": config['model']
            }
        )
        wandb_run_id = run.id
        print(f'Created new wandb run: {wandb_run_id}', flush=True)

    # Upload config.yaml to wandb as artifact at the start of training
    import shutil
    config_copy_path = './config_copy.yaml'
    shutil.copy('config.yaml', config_copy_path)
    artifact = wandb.Artifact(name="config", type="file")
    artifact.add_file(local_path="config_copy.yaml", name="training_config")
    artifact.save()
    # Delete the copy after wandb saves it
    if os.path.exists(config_copy_path):
        os.remove(config_copy_path)
    print(f'Uploaded config to wandb as artifact', flush=True)

if is_main_process:
    print('\nStarting Training...\n', flush=True)

for epoch in range(start_epoch, EPOCHS):
    # Save config to run path at epoch 0
    if epoch == 0 and is_main_process:
        trainer.save_config('config.yaml')
        print('Saved config to run path', flush=True)

    # Log epoch start with timestamp
    if is_main_process:
        from datetime import datetime
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========== Starting Epoch {epoch + 1}/{EPOCHS} ==========", flush=True)

    # Training loop
    model.train()
    trainer.train(train_prefetcher, criterion, optimizer, metrics)

    # Validation loop
    model.eval()
    plot = (epoch % config['training']['validation_plot_freq'] == 0)
    trainer.validate(val_prefetcher, criterion, optimizer, metrics, plot=plot, epoch=epoch)

    if is_main_process:
        from datetime import datetime
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========== Completed Epoch {epoch + 1}/{EPOCHS} ==========", flush=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch + 1} | Train Loss: {metrics['training_loss'][-1]:.4f} | Val. Loss: {metrics['validation_loss'][-1]:.4f} | Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}", flush=True)

# Save final checkpoint only on main process
if is_main_process:
    trainer.save_model_and_states_checkpoint(epoch, metrics, optimizer, wandb_run_id, scheduler=None)
    run_path = os.path.join(MODEL_SAVE_PATH, 'run' + str(config['trainer']['run_num']))
    with open(os.path.join(run_path, 'metrics.pickle'), 'wb') as file:
        pickle.dump(metrics, file)
    print('\nFinished Training!', flush=True)

if is_main_process and config['wandb']['enabled']:
    wandb.finish() 