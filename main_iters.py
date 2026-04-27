import argparse
import os
import pickle
import socket
from datetime import datetime
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from data import CombinedDataset, PtychographyDataset
from data_simple_pack import PtychographyDatasetPacked
from model.model import PtychoViT
from training_iters import Trainer
from utils.flops_utils import AnalyticalFlopsForTrainer, PtychoViTFlopsCalculator
from utils.data_utils import build_train_loader, build_val_loader
from utils.distributed import cleanup_distributed, ddp_barrier, init_distributed
from utils.utils import (
    build_criterion,
    build_optimizer_and_scheduler,
    compute_sha256,
)


def resolve_config_path(config_path: str) -> str:
    """Resolve configuration path (relative to script directory)."""
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    return config_path


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
config_path = resolve_config_path(args.config)
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Training parameters
MODE = config['training']['mode']
BATCH_SIZE = config['training']['batch_size']
VALID_BATCH_SIZE = config['training']['valid_batch_size']
LR = config['training']['learning_rate']
EPOCHS = config['training']['epochs']
MODEL_SAVE_PATH = config['paths']['model_save_path']
FINETUNE_PATH = config['training'].get('finetune_from_model')

# Distributed init
rank, world_size, local_rank, DEVICE = init_distributed(config['training'].get('platform', 'slurm'))
is_main_process = rank == 0

ddp_barrier(world_size, DEVICE)


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
    dummy_probe = torch.randn((1, 1, 10, img_size, img_size, 2))
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
    dev_index = DEVICE.index
    model = DDP(model, device_ids=[dev_index], output_device=dev_index, find_unused_parameters=False, gradient_as_bucket_view=True)

ddp_barrier(world_size, DEVICE)

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
criterion = build_criterion(config)
optimizer, scheduler, encoder_lr, amp_decoder_lr, ph_decoder_lr = build_optimizer_and_scheduler(
    model, config
)

if is_main_process:
    print("\nOptimizer learning rates:", flush=True)
    print(f"  Encoder: {encoder_lr}", flush=True)
    print(f"  Amplitude Decoder: {amp_decoder_lr}", flush=True)
    print(f"  Phase Decoder: {ph_decoder_lr}", flush=True)


# Track starting epoch for checkpoint resumption
start_epoch = 0
wandb_run_id = None

if FINETUNE_PATH and is_main_process:
    print("finetune_from_model is set; ignoring resume_from_checkpoint to keep optimizer state fresh.", flush=True)

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
is_packed = config['data'].get('packed', False)

# Create full dataset with sequential indices
# Shuffling is handled by random_split with a deterministic seed
if is_packed:
    base_dataset = PtychographyDatasetPacked(
        pack_dir=data_dir,
        rank=rank,
        world_size=world_size,
        scale=config['data']['scale'],
        normalization_dict_path=config['data'].get('normalization_dict_path'),
        default_normalization=config['data'].get('default_normalization', 100000.0),
        apply_noise=config['data'].get('apply_noise', True),
        max_probe_modes=config['data'].get('max_probe_modes', 8),
        target_size=config['data'].get('target_size', 256),
        max_shards=config['data'].get('max_files'),
        debug=DEBUG_MODE
)
else:
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

total_size = len(base_dataset)

# Split into train and validation using PyTorch's random_split
# This ensures mutually exclusive splits and follows PyTorch best practices
train_split = config['data']['train_split']
train_size = int(total_size * train_split)
val_size = total_size - train_size

generator = torch.Generator().manual_seed(config['data']['random_seed'])
train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)
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

train_dataloader_kwargs_base = {
    'batch_size': BATCH_SIZE,
    'num_workers': config['data'].get('num_workers', 0),
    'pin_memory': pin_memory,
    'shuffle': True,
    'drop_last': drop_last,
}

val_dataloader_kwargs_base = {
    'batch_size': VALID_BATCH_SIZE,
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

checkpoint_path = os.path.join(MODEL_SAVE_PATH, 'run' + str(config['trainer']['run_num']))
model_checkpoint = os.path.join(checkpoint_path, 'checkpoint_model.pth')
resume_training = (
    config['training'].get('resume_from_checkpoint', False) and not FINETUNE_PATH
)
if resume_training and not os.path.exists(model_checkpoint):
    if is_main_process:
        print(
            f"resume_from_checkpoint is set but no model checkpoint at {model_checkpoint}; "
            "starting from scratch.",
            flush=True,
        )
    resume_training = False

_n_train_samples = len(train_subset_base)
if drop_last:
    _batches_per_epoch = _n_train_samples // BATCH_SIZE
else:
    _batches_per_epoch = (_n_train_samples + BATCH_SIZE - 1) // BATCH_SIZE
# Per-rank step budget: global batches×epochs split across GPUs (matches one process's share).
_train_max_iters = (_batches_per_epoch * EPOCHS) // world_size

flops_calcs = None
if config["model"].get("encoder_type", "custom") == "custom":
    flops_calcs = AnalyticalFlopsForTrainer(
        PtychoViTFlopsCalculator(
            config["model"],
            batch_size=1,
            probe_modes=config["data"].get("max_probe_modes", 10),
        ),
        BATCH_SIZE,
        world_size,
    )

trainer = Trainer(
    model,
    MODE,
    config['trainer']['run_num'],
    DEVICE,
    MODEL_SAVE_PATH,
    criterion=criterion,
    optimizer=optimizer,
    is_main_process=is_main_process,
    use_ddp=(world_size > 1),
    wandb_enabled=config['wandb']['enabled'],
    wandb_run_id=wandb_run_id,
    scheduler=scheduler,
    debug_mode=DEBUG_MODE,
    log_every=config['training'].get('log_every', 100),
    max_iters=_train_max_iters,
    flops_calcs=flops_calcs,
)

# Restore checkpoint first when resuming, then build train loader once (with resume sampler state).
if resume_training:
    if is_main_process:
        print('\nResuming from checkpoint...', flush=True)
    if world_size > 1:
        model.module.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    else:
        model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    start_epoch, wandb_run_id = trainer.load_state_checkpoint()
    trainer.wandb_run_id = wandb_run_id
    if wandb_run_id is None and config['wandb'].get('resume_run_id') is not None:
        wandb_run_id = config['wandb']['resume_run_id']
        trainer.wandb_run_id = wandb_run_id
        if is_main_process:
            print(f'Using manually specified wandb run ID: {wandb_run_id}', flush=True)
    if is_main_process:
        print(
            f'Loaded checkpoint: resume epoch index {start_epoch}, '
            f'global iters {trainer.iters}, '
            f'samples this epoch {trainer.samples_seen_in_epoch}',
            flush=True,
        )
        if wandb_run_id:
            print(f'Will resume wandb run: {wandb_run_id}', flush=True)
        else:
            print('No wandb run ID found - will create new wandb run', flush=True)

(
    train_dataset,
    train_loader,
    train_sampler,
    train_prefetcher,
    _,
    train_dataloader_kwargs,
) = build_train_loader(
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
    resume_epoch=start_epoch,
    resume_num_samples=trainer.samples_seen_in_epoch,
    is_main=is_main_process,
)

(
    val_dataset,
    val_loader,
    val_sampler,
    val_prefetcher,
    val_dataloader_kwargs,
) = build_val_loader(
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

# Create test dataset and loader only on main process
# if is_main_process:
if config['data'].get('test_path') is not None:
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
    ddp_barrier(world_size, DEVICE)
else:
    test_loader = None
    if is_main_process:
        print("Test plotting disabled: data.test_path is null", flush=True)

trainer.set_dataloaders(train_prefetcher, val_prefetcher, test_loader)

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
#    print(f"Number of paired files: {len(base_dataset.file_paths)}", flush=True)
    print(f"Total patterns: {len(base_dataset)}", flush=True)
    print(f"Train patterns (this rank): {len(train_dataset)} | Val patterns (this rank): {len(val_dataset)}", flush=True)
    print(f"Total batches/epoch (train): {len(train_loader)}", flush=True)
    print(f"Total batches/epoch (val): {len(val_loader)}", flush=True)
    print(f"Valid batch size: {VALID_BATCH_SIZE}", flush=True)
    print(f"Max iterations: {_train_max_iters}", flush=True)
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
# Initialize wandb only on main process
# ────────────────────────────────────────────────────────────────────────────────
if is_main_process and config['wandb']['enabled']:
    wandb.login()
    run_name = config['wandb']['run_name']
    if wandb_run_id is not None:
        # Resume existing wandb run
        run = wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project'],
            id=wandb_run_id,
            resume='must',
            name=run_name,
        )
        print(f'Resumed wandb run: {wandb_run_id}', flush=True)
    else:
        # Create new wandb run
        run = wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project'],
            name=run_name,
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

    if FINETUNE_PATH and finetune_checkpoint_sha256 is not None:
        wandb.run.summary["finetune_checkpoint_sha256"] = finetune_checkpoint_sha256
        print("Logged finetune checkpoint SHA256 to wandb", flush=True)

    trainer.wandb_run_id = wandb_run_id


ddp_barrier(world_size, DEVICE)

if is_main_process:
    print('\nStarting Training...\n', flush=True)

# ────────────────────────────────────────────────────────────────────────────────
# Train / Validate
# ────────────────────────────────────────────────────────────────────────────────
for epoch in range(start_epoch, EPOCHS):
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

    ddp_barrier(world_size, DEVICE)

    # Log epoch start with timestamp
    if is_main_process:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========== Starting Epoch {epoch + 1}/{EPOCHS} ==========", flush=True)
    
    # Training loop
    model.train()
    profile = config['training'].get('profile', False)
    trainer.train(profile=profile, epoch=epoch) 

    ddp_barrier(world_size, DEVICE)


if is_main_process:
    trainer.save_final(wandb_run_id=wandb_run_id)
    run_path = os.path.join(MODEL_SAVE_PATH, 'run' + str(config['trainer']['run_num']))
    os.makedirs(run_path, exist_ok=True)
    with open(os.path.join(run_path, 'metrics_iters.pickle'), 'wb') as file:
        pickle.dump(trainer.metrics, file)
    print('\nFinished Training!', flush=True)

ddp_barrier(world_size, DEVICE)
if is_main_process and config['wandb']['enabled']:
    wandb.finish()
cleanup_distributed()

