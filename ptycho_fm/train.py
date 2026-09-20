import argparse
import hashlib
import math
import os
import pickle
import socket
from datetime import UTC, datetime

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from torch import nn, optim
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, random_split
from torchinfo import summary

from ptycho_fm.custom_loss import (
    CombinedLoss,
    ProbeAwareLoss,
    QDependentLoss,
    TotalFluxLoss,
    WeightedLoss,
)
from ptycho_fm.data import CombinedDataset, PtychographyDataset
from ptycho_fm.data_simple_pack import PtychographyDatasetPacked
from ptycho_fm.mlflow_logger import MLflowLogger
from ptycho_fm.model.model import PtychoFM, PtychoFMCoupledAmpPh, PtychoFMReIm
from ptycho_fm.training import Trainer
from ptycho_fm.utils.config import load_config as load_yaml_config
from ptycho_fm.utils.data import build_train_loader as _build_train_loader
from ptycho_fm.utils.data import build_val_loader as _build_val_loader
from ptycho_fm.utils.data import dataset_fingerprint, validate_loader_lengths
from ptycho_fm.utils.distributed import cleanup_distributed, init_distributed
from ptycho_fm.utils.flops import ACCOUNTING_VERSION, PtychoFMFlopsCalculator
from ptycho_fm.utils.progress import TrainingProgress, append_logs
from ptycho_fm.utils.schedulers import build_scheduler


def override_output_norm_from_config(model, model_config):
    """Restore explicitly configured output normalization after weight loading."""
    output_norm = model_config.get('output_norm')
    if not output_norm:
        return {}

    scale_keys = ('amp_scale', 'ph_scale', 'real_scale', 'imag_scale')
    offset_keys = ('amp_offset', 'real_offset', 'imag_offset')
    overridden = {}

    with torch.no_grad():
        for key in scale_keys:
            if key not in output_norm or not hasattr(model, key):
                continue
            value = float(output_norm[key])
            if value <= 0:
                raise ValueError(
                    f"Output norm scale '{key}' must be positive, got {value}"
                )
            getattr(model, key).fill_(math.log(value))
            overridden[key] = value

        for key in offset_keys:
            if key not in output_norm or not hasattr(model, key):
                continue
            value = float(output_norm[key])
            getattr(model, key).fill_(value)
            overridden[key] = value

    return overridden


def main() -> None:
    """Entry point for `ptycho-fm-train` and `python -m ptycho_fm.train`."""
    def resolve_config_path(config_path):
        """Resolve configuration paths relative to the caller's working directory."""
        return os.path.abspath(os.path.expanduser(config_path))

    def load_config(config_path='config.yaml'):
        """Load configuration from YAML file."""
        config_path = resolve_config_path(config_path)
        config = load_yaml_config(config_path)
        if config.get('trainer', {}).get('run_num') is None:
            # Ensure run folder name is "run_yyyymmdd_hhmmss"
            config.setdefault('trainer', {})['run_num'] = datetime.now(tz=UTC).astimezone().strftime("_%Y%m%d_%H%M%S")
        return config

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='PtychoFM Training Script')
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config YAML file (default: config.yaml in the working directory)',
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
            raise TypeError("training.data_subsetting_schedule.epochs and fractions must be lists.")
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




    # Load configuration
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)

    # Training parameters
    MODE = config['training']['mode']
    BATCH_SIZE = config['training']['batch_size']
    LR = config['training']['learning_rate']
    EPOCHS = config['training']['epochs']
    MODEL_SAVE_PATH = config['paths']['model_save_path']
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


    def _ddp_barrier():
        """Keep all ranks in lockstep before collectives. Rank-only work (e.g. wandb.init on rank 0)
        must finish before any rank hits the first DDP all_reduce or peers will block forever."""
        if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier(device_ids=[DEVICE.index] if DEVICE.type == 'cuda' else None)

    _ddp_barrier()

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
            deterministic_noise=config['data'].get('deterministic_noise', False),
            noise_seed=config['data'].get('noise_seed', 0),
            max_probe_modes=config['data'].get('max_probe_modes', 8),
            max_shards=config['data'].get('max_shards'),
            debug=DEBUG_MODE
    )
    else:
        base_dataset = CombinedDataset(
            file_paths=data_dir,
            rank=rank,
            world_size=world_size,
            scale=config['data']['scale'],
            normalization_dict_path=config['data'].get('normalization_dict_path'),
            default_normalization=config['data'].get('default_normalization', 10000.0),
            apply_noise=config['data'].get('apply_noise', True),
            deterministic_noise=config['data'].get('deterministic_noise', False),
            noise_seed=config['data'].get('noise_seed', 0),
            cache_object=config['data'].get('cache_object', False),
            max_probe_modes=config['data'].get('max_probe_modes', 8),
            max_OPR_modes=config['data'].get('max_OPR_modes', 1),
            cache_memory_budget_mb=config['data'].get('cache_memory_budget_mb', 512),
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
        'generator': torch.Generator().manual_seed(config['data']['random_seed'] + rank),
    }

    val_dataloader_kwargs_base = {
        'batch_size': config['training'].get('valid_batch_size', BATCH_SIZE),
        'num_workers': config['data'].get('num_workers', 0),
        'pin_memory': pin_memory,
        'shuffle': False,
        'drop_last': drop_last,
        'generator': torch.Generator().manual_seed(config['data']['random_seed'] + rank),
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

    # Create test dataset and loader only on main process
    # if is_main_process:
    if config['data'].get('test_path') is not None:
        test_dataset = PtychographyDataset(
            file_path=config['data']['test_path'],
            scale=config['data']['scale'],
            normalization_dict_path=config['data'].get('test_normalization'),
            apply_noise=config['data'].get('test_apply_noise', False),
            default_normalization=config['data'].get('default_normalization', 100000.0),
            max_probe_modes=config['data'].get('max_probe_modes', 8),
            max_OPR_modes=config['data'].get('max_OPR_modes', 1),
            cache_object=config['data'].get('cache_object', True),
            cache_memory_budget_mb=config['data'].get('cache_memory_budget_mb', 512),
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
        _ddp_barrier()
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
    #    print(f"Number of paired files: {len(base_dataset.file_paths)}", flush=True)
        print(f"Total patterns (after subset_fraction): {len(full_dataset)}", flush=True)
        train_patterns_this_rank = len(train_sampler) if train_sampler is not None else len(train_dataset)
        val_patterns_this_rank = len(val_sampler) if val_sampler is not None else len(val_dataset)
        print(
            f"Train patterns (this rank): {train_patterns_this_rank} | "
            f"Val patterns (this rank): {val_patterns_this_rank}",
            flush=True,
        )
        print(f"Total batches/epoch (train): {len(train_loader)}", flush=True)
        print(f"Total batches/epoch (val): {len(val_loader)}", flush=True)
        print("\nDataLoader Settings:", flush=True)
        print(f"  num_workers: {train_dataloader_kwargs['num_workers']}", flush=True)
        print(f"  pin_memory: {train_dataloader_kwargs['pin_memory']}", flush=True)
        print(f"  train shuffle: {train_dataloader_kwargs['shuffle']} (per-epoch local shuffling)", flush=True)
        print(f"  val shuffle: {val_dataloader_kwargs['shuffle']}", flush=True)
        print(f"  drop_last: {train_dataloader_kwargs['drop_last']}", flush=True)
        print(
            "  training/validation synthetic noise: "
            f"{config['data'].get('apply_noise', True)}",
            flush=True,
        )
        print(
            "  test synthetic noise: "
            f"{config['data'].get('test_apply_noise', False)}",
            flush=True,
        )
        if train_dataloader_kwargs['num_workers'] > 0:
            print(f"  prefetch_factor: {train_dataloader_kwargs.get('prefetch_factor', 'N/A')}", flush=True)
            print(f"  persistent_workers: {train_dataloader_kwargs.get('persistent_workers', 'N/A')}", flush=True)
        use_prefetcher_status = torch.cuda.is_available() and config['data'].get('use_cuda_prefetcher', True)
        print(f"  Using CUDAPrefetcher: {use_prefetcher_status}", flush=True)
        print(f"\nDevice: {DEVICE}", flush=True)
        print(f"Model save path: {MODEL_SAVE_PATH}", flush=True)
        print("=" * 50, flush=True)

    # ────────────────────────────────────────────────────────────────────────────────
    # Model setup
    # ────────────────────────────────────────────────────────────────────────────────
    img_size = config['model'].get('encoder', {}).get('img_size', 256)

    # Select model based on coupled_decoder_mode config
    coupled_decoder_mode = config['model'].get('coupled_decoder_mode', None)

    if coupled_decoder_mode == 'real_imag':
        model = PtychoFMReIm(config=config['model'])
        model_name = "PtychoFMReIm"
        decoder_info = "coupled (real/imag)"
    elif coupled_decoder_mode == 'amp_phase':
        model = PtychoFMCoupledAmpPh(config=config['model'])
        model_name = "PtychoFMCoupledAmpPh"
        decoder_info = "coupled (amp/phase)"
    else:
        model = PtychoFM(config=config['model'])
        model_name = "PtychoFM"
        decoder_info = config['model'].get('decoder_type', 'standard')

    if is_main_process:
        encoder_type = config['model'].get('encoder_type', 'custom')
        print(
            f"Using {model_name} with {encoder_type.upper()} encoder and "
            f"{decoder_info} decoder",
            flush=True,
        )
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
        except (RuntimeError, TypeError, AttributeError) as e:
            print(f"[Warning] torchinfo summary failed and will be skipped: {e}", flush=True)

    # Move model to device and wrap with DDP (multinode.py approach)
    model = model.to(DEVICE)
    if world_size > 1:
        # Use torch.cuda.current_device() like multinode.py does
        # When CUDA_VISIBLE_DEVICES is set by SLURM, don't pass device_ids to avoid NCCL PCI bus ID lookup
        dev_index = DEVICE.index
        model = DDP(model, device_ids=[dev_index] if DEVICE.type == 'cuda' else None, output_device=dev_index if DEVICE.type == 'cuda' else None, find_unused_parameters=False, gradient_as_bucket_view=True)

    _ddp_barrier()

    # Fine-tuning loads model weights here, before creating a fresh optimizer.
    FINETUNE_PATH = config['training'].get('finetune_from_model')
    resume_from_checkpoint = config['training'].get('resume_from_checkpoint', False)
    if FINETUNE_PATH is not None and resume_from_checkpoint:
        raise ValueError(
            "Set only one of training.finetune_from_model or "
            "training.resume_from_checkpoint"
        )

    finetune_checkpoint_sha256 = None
    if FINETUNE_PATH is not None:
        if is_main_process:
            print(f"\nFinetuning from model weights: {FINETUNE_PATH}", flush=True)
            print(
                "Using fresh optimizer, metrics, epoch counter, and tracking run.",
                flush=True,
            )
            with open(FINETUNE_PATH, 'rb') as checkpoint_file:
                finetune_checkpoint_sha256 = hashlib.file_digest(checkpoint_file, 'sha256').hexdigest()
        state = torch.load(FINETUNE_PATH, map_location=DEVICE, weights_only=True)
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            finetune_model = model.module
        else:
            finetune_model = model
        finetune_model.load_state_dict(state)

        overridden_output_norm = override_output_norm_from_config(
            finetune_model, config['model']
        )
        if is_main_process:
            print(
                f"Loaded finetune weights from {FINETUNE_PATH}; "
                "optimizer will start fresh.",
                flush=True,
            )
            if overridden_output_norm:
                values = ", ".join(
                    f"{key}={value}" for key, value in overridden_output_norm.items()
                )
                print(
                    "Overrode checkpoint output normalization from active config: "
                    f"{values}",
                    flush=True,
                )

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
    elif config['training']['loss_function'] == 'total_flux':
        criterion = TotalFluxLoss()
    elif config['training']['loss_function'] == 'weighted':
        weighted_loss_config = config['training']['weighted_loss']
        criterion = WeightedLoss(loss_type=weighted_loss_config['loss_type'], threshold=weighted_loss_config['threshold'], alpha=weighted_loss_config['alpha'])
    elif config['training']['loss_function'] == 'probe_aware':
        probe_aware_loss_config = config['training']['probe_aware_loss']
        criterion = ProbeAwareLoss(
            loss_type=probe_aware_loss_config.get('loss_type', 'mse'),
            threshold=probe_aware_loss_config.get('threshold', 0.0),
            alpha=probe_aware_loss_config.get('alpha', 1.0),
            envelope_floor=probe_aware_loss_config.get('envelope_floor', 0.05),
            eps=probe_aware_loss_config.get('eps', 1e-6),
        )
    elif config['training']['loss_function'] == 'q_dependent':
        q_dependent_loss_config = config['training']['q_dependent_loss']
        criterion = QDependentLoss(
            loss_type=q_dependent_loss_config.get('loss_type', 'mse'),
            alpha=q_dependent_loss_config.get('alpha', 1.0),
            q_beta=q_dependent_loss_config.get('q_beta', 1.0),
            q_floor=q_dependent_loss_config.get('q_floor', 0.05),
            threshold=q_dependent_loss_config.get('threshold', 0.0),
            log_error=q_dependent_loss_config.get('log_error', True),
            normalize_q=q_dependent_loss_config.get('normalize_q', True),
            eps=q_dependent_loss_config.get('eps', 1e-6),
        )
    elif config['training']['loss_function'] == 'combined':
        criterion = CombinedLoss(
            config['training']['combined_loss'], config['training']
        )
    else:
        raise ValueError(f"Unknown loss function: {config['training']['loss_function']}")

    # Get individual learning rates from config (fallback to default LR if not specified)
    encoder_lr = config['training'].get('encoder_lr', LR)
    amp_decoder_lr = config['training'].get('amp_decoder_lr', LR)
    ph_decoder_lr = config['training'].get('ph_decoder_lr', LR)
    coupled_decoder_lr = config['training'].get('coupled_decoder_lr', LR)

    # Get the actual model (unwrap DDP if needed)
    actual_model = model.module if isinstance(model, DDP) else model

    if coupled_decoder_mode in ('real_imag', 'amp_phase'):
        if coupled_decoder_mode == 'real_imag':
            output_norm_params = [
                actual_model.real_scale,
                actual_model.imag_scale,
                actual_model.real_offset,
                actual_model.imag_offset,
            ]
        else:
            output_norm_params = [
                actual_model.amp_scale,
                actual_model.ph_scale,
                actual_model.amp_offset,
            ]

        param_groups = [
            {'params': actual_model.encoder.parameters(), 'lr': encoder_lr, 'name': 'encoder'},
            {'params': actual_model.coupled_decoder.parameters(), 'lr': coupled_decoder_lr, 'name': 'coupled_decoder'},
            {'params': output_norm_params, 'lr': coupled_decoder_lr, 'name': 'output_norm_params'},
        ]
    else:
        output_norm_params = [
            actual_model.amp_scale,
            actual_model.amp_offset,
            actual_model.ph_scale,
        ]
        param_groups = [
            {'params': actual_model.encoder.parameters(), 'lr': encoder_lr, 'name': 'encoder'},
            {'params': actual_model.amp_decoder.parameters(), 'lr': amp_decoder_lr, 'name': 'amp_decoder'},
            {'params': actual_model.ph_decoder.parameters(), 'lr': ph_decoder_lr, 'name': 'ph_decoder'},
            {'params': output_norm_params, 'lr': amp_decoder_lr, 'name': 'output_norm_params'},
        ]

    optimizer = optim.Adam(param_groups, fused=(DEVICE.type == 'cuda'))

    lr_sched_cfg = config['training'].get('lr_scheduler', {})
    scheduler, scheduler_unit = build_scheduler(optimizer, lr_sched_cfg)
    cooldown = lr_sched_cfg.get('enabled', False) and lr_sched_cfg.get('scheduler_class') == 'cooldown'
    branch_from = lr_sched_cfg.get('kwargs', {}).get('branch_from') if cooldown and not resume_from_checkpoint else None
    if cooldown and not resume_from_checkpoint and not branch_from:
        raise ValueError('A new cooldown run requires lr_scheduler.kwargs.branch_from')
    if branch_from and FINETUNE_PATH:
        raise ValueError('Cooldown branching and weights-only fine-tuning are mutually exclusive')
    flop_calculator = None
    if config['training'].get('track_flops', False) or config['training'].get('log_at_flops'):
        flop_calculator = PtychoFMFlopsCalculator(config['model'], batch_size=BATCH_SIZE)
    progress = TrainingProgress(config['training'], calculator=flop_calculator)
    if cooldown:
        progress.enabled = True
    resume_context = {
        'world_size': world_size, 'batch_size': BATCH_SIZE, 'model': config['model'],
        'data': config['data'], 'train_size': train_size, 'val_size': val_size,
        'subsetting': config['training'].get('data_subsetting_schedule'),
        'dataset_fingerprint': dataset_fingerprint(base_dataset, config['data'].get('normalization_dict_path')),
        'objective': {key: config['training'].get(key) for key in
                      ('mode', 'loss_function', 'weighted_loss', 'combined_loss',
                       'probe_aware_loss', 'q_dependent_loss', 'skip_batch_if_grad_norm_greater_than')},
        'scheduler': lr_sched_cfg,
    }

    if is_main_process:
        print("\nOptimizer learning rates:", flush=True)
        print(f"  Encoder: {encoder_lr}", flush=True)
        if coupled_decoder_mode in ('real_imag', 'amp_phase'):
            print(
                f"  Coupled Decoder ({coupled_decoder_mode}): "
                f"{coupled_decoder_lr}",
                flush=True,
            )
        else:
            print(f"  Amplitude Decoder: {amp_decoder_lr}", flush=True)
            print(f"  Phase Decoder: {ph_decoder_lr}", flush=True)

    metrics = {'training_loss': [], 'train_amp_loss': [], 'train_ph_loss': [], 'validation_loss': [],
               'val_amp_loss': [], 'val_ph_loss': [], 'best_val_loss': np.inf}

    # Track starting epoch for checkpoint resumption
    start_epoch = 0
    wandb_run_id = None

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
        progress=progress,
        resume_context=resume_context,
        mlflow_log_every_n_batches=config.get('mlflow', {}).get('log_every_n_batches', 50),
    )

    if is_main_process:
        mlflow_logger.log_params({
            "learning_rate": LR,
            "encoder_lr": encoder_lr,
            "amp_decoder_lr": amp_decoder_lr,
            "ph_decoder_lr": ph_decoder_lr,
            "coupled_decoder_lr": coupled_decoder_lr,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "loss_function": config['training']['loss_function'],
            "encoder_type": config['model'].get('encoder_type', 'custom'),
            "coupled_decoder_mode": coupled_decoder_mode,
            "world_size": world_size,
            "model": config['model'],
            "data": config['data'],
            "training": config['training'],
            "trainer": config['trainer'],
        })
        mlflow_logger.log_artifact(config_path)
        if FINETUNE_PATH and finetune_checkpoint_sha256 is not None:
            mlflow_logger.log_params({"finetune_checkpoint_sha256": finetune_checkpoint_sha256})


    # ────────────────────────────────────────────────────────────────────────────────
    # Resume from checkpoint if requested
    # ────────────────────────────────────────────────────────────────────────────────
    if resume_from_checkpoint:
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

    if branch_from:
        branch_path = os.path.abspath(os.path.expanduser(branch_from))
        branch_name = os.path.basename(branch_path)
        if not branch_name.startswith('state_iters_'):
            raise ValueError('Cooldown requires a specific state_iters_N.pth checkpoint')
        run_path = os.path.abspath(os.path.join(MODEL_SAVE_PATH, 'run' + str(config['trainer']['run_num'])))
        if run_path == os.path.dirname(branch_path):
            raise ValueError('Cooldown must use a new run directory')
        branch_model = os.path.join(os.path.dirname(branch_path), branch_name.replace('state_iters_', 'model_iters_', 1))
        actual_model.load_state_dict(torch.load(branch_model, map_location=DEVICE, weights_only=True))
        branch_state = torch.load(branch_path, map_location='cpu', weights_only=True)
        parent_context = branch_state.get('resume_context', {})
        for key, value in resume_context.items():
            if key != 'scheduler' and parent_context.get(key) != value:
                raise ValueError(f'Cooldown parent {key} differs; explicit conversion is required')
        trainer.resume_context = parent_context
        start_epoch, metrics, optimizer, _, _ = trainer.load_state_checkpoint(
            optimizer, scheduler=scheduler, checkpoint_file=branch_path, skip_scheduler=True)
        trainer.resume_context = resume_context
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']
        scheduler, scheduler_unit = build_scheduler(optimizer, lr_sched_cfg)
        progress.max_updates = progress.optimizer_steps + lr_sched_cfg['kwargs']['cooldown_steps']
        wandb_run_id = None

    if progress.samples_in_epoch:
        if sharding_strategy != 'dynamic':
            raise ValueError('Mid-epoch resume requires dynamic sharding')
        if config['data'].get('apply_noise', True) and not config['data'].get('deterministic_noise', False):
            raise ValueError('Exact mid-epoch resume with Poisson noise requires deterministic_noise=True in the original run')

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
                    "coupled_decoder_lr": coupled_decoder_lr,
                    "batch_size": BATCH_SIZE,
                    "dataset": config['wandb']['dataset_name'],
                    "epochs": EPOCHS,
                    "notes": config['wandb']['notes'],
                    "encoder_type": config['model'].get('encoder_type', 'custom'),
                    "coupled_decoder_mode": coupled_decoder_mode,
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


    _ddp_barrier()

    if is_main_process:
        print('\nStarting Training...\n', flush=True)

    def validate_iteration(epoch, crossed, at_boundary):
        if progress.last_validation_iter == progress.iters:
            return
        train_metrics = {key: trainer.synchronize_loss(value)
                         for key, value in progress.interval_metrics().items()}
        if not train_metrics:
            return
        model.eval()
        val_metrics = trainer.validate(
            val_prefetcher, criterion, optimizer, metrics,
            epoch=progress.iters, profile=config['training'].get('profile', False),
            plot=(at_boundary and epoch % config['training']['validation_plot_freq'] == 0),
            scheduler=scheduler if scheduler_unit == 'epoch' and at_boundary else None,
            emit_scalars=False,
        )
        for source, dest in [('train_loss', 'training_loss'), ('train_amp_loss', 'train_amp_loss'), ('train_ph_loss', 'train_ph_loss')]:
            metrics[dest].append(train_metrics[source])
        payload = {
            **train_metrics, **val_metrics, 'iter': progress.iters,
            'optimizer_steps': progress.optimizer_steps,
            'samples_seen': progress.samples_seen,
            'tflops_consumed': progress.tflops_consumed if flop_calculator else None,
            'lr': optimizer.param_groups[0]['lr'],
            'interval_start_iter': progress.last_log_iter,
            'interval_batches': progress.interval_batches,
            'world_size': world_size, 'batch_size_per_rank': BATCH_SIZE,
            'targets_crossed': len(crossed),
            'last_target_flops': max(crossed) if crossed else None,
        }
        if flop_calculator:
            payload['tokens_seen'] = progress.samples_seen * flop_calculator.grid_after_vit() ** 2
            payload['params_m'] = flop_calculator.param_count()
        if is_main_process:
            run_path = os.path.join(MODEL_SAVE_PATH, 'run' + str(config['trainer']['run_num']))
            append_logs(os.path.join(run_path, 'logs.txt'), {
                **payload, 'accounting_version': ACCOUNTING_VERSION if flop_calculator else '',
                'target_flops': ';'.join(str(v) for v in crossed),
            })
            if config['wandb']['enabled']:
                wandb.log({key: value for key, value in payload.items() if value is not None}, step=progress.iters)
            mlflow_logger.log_metrics(payload, step=progress.iters)
        if (at_boundary and epoch % config['training']['test_plot_freq'] == 0
                and is_main_process and test_loader is not None
                and not config['training'].get('profile', False)):
            trainer.generate_test_plot(
                test_loader, progress.iters, f'test_iters{progress.iters}.png',
                central_crop=config['training'].get('test_plot_central_crop', 64),
                ph_crop=config['training'].get('test_plot_ph_crop', 180),
            )
        progress.mark_logged()
        trainer.save_model_and_states_checkpoint(epoch, metrics, optimizer, wandb_run_id, scheduler)
        model.train()

    # ────────────────────────────────────────────────────────────────────────────────
    # Train / Validate
    # ────────────────────────────────────────────────────────────────────────────────
    epoch = max(start_epoch - 1, 0)
    try:
        for epoch in range(start_epoch, EPOCHS):
            if progress.stopped:
                break
            # Optionally apply per-epoch training subsetting schedule
            if data_subsetting_schedule is not None or progress.enabled:
                fraction = _fraction_for_epoch(data_subsetting_schedule, epoch) if data_subsetting_schedule else 1.0
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
                    resume_epoch=epoch,
                    resume_num_samples=progress.samples_in_epoch if epoch == start_epoch else 0,
                )

                if is_main_process:
                    print(
                        f"Data subsetting schedule: epoch {epoch} -> "
                        f"{fraction * 100:.2f}% of training set "
                        f"({len(train_subset_epoch)} samples before sharding)",
                        flush=True,
                    )

            validate_loader_lengths(train_loader, val_loader, DEVICE, distributed=world_size > 1)

            # Set epoch for DistributedSampler (dynamic sharding only)
            if sharding_strategy == 'dynamic':
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                if val_sampler is not None:
                    val_sampler.set_epoch(epoch)

            # Save config to run path at epoch 0
            if epoch == start_epoch and is_main_process:
                run_path = os.path.join(MODEL_SAVE_PATH, 'run' + str(config['trainer']['run_num']))
                os.makedirs(run_path, exist_ok=True)
                with open(os.path.join(run_path, 'config.yaml'), 'w') as stream:
                    yaml.safe_dump(config, stream, sort_keys=False)
                print('Saved config to run path', flush=True)

            _ddp_barrier()

            # Log epoch start with timestamp
            if is_main_process:
                print(f"\n[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] ========== Starting Epoch {epoch + 1}/{EPOCHS} ==========", flush=True)
            
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
            profile = config['training'].get('profile', False)
            trainer.train(train_prefetcher, criterion, optimizer, metrics, profile=profile, epoch=epoch,
                          scheduler=scheduler, scheduler_unit=scheduler_unit,
                          on_iteration=validate_iteration if progress.enabled else None)
            if progress.enabled:
                if progress.stopped:
                    break
                continue


            # Validation loop
            model.eval()
            plot = (epoch % config['training']['validation_plot_freq'] == 0) and (not profile)
            
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
                print(f"\n[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] Running validation...", flush=True)
            trainer.validate(val_prefetcher, criterion, optimizer, metrics, 
                plot=plot, epoch=epoch, scheduler=scheduler if scheduler_unit == "epoch" else None, profile=profile)

            # Generate test plot only on main process
            if epoch % config['training']['test_plot_freq'] == 0 and is_main_process and (not profile) and test_loader is not None:
                print(f"\n[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] Generating test plot...", flush=True)
                trainer.generate_test_plot(
                    test_loader,
                    epoch,
                    'test_epoch' + str(epoch) + '.png',
                    central_crop=config['training'].get('test_plot_central_crop', 64),
                    ph_crop=config['training'].get('test_plot_ph_crop', 180),
                )

            # Rank 0 test plot / I/O can lag other ranks at end of epoch; align before next epoch.
            _ddp_barrier()

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
                print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] ========== Completed Epoch {epoch + 1}/{EPOCHS} ==========", flush=True)
                print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch + 1} | Train Loss: {metrics['training_loss'][-1]:.4f} | Val. Loss: {metrics['validation_loss'][-1]:.4f} | Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}", flush=True)
        trainer.save_model_and_states_checkpoint(epoch, metrics, optimizer, wandb_run_id, scheduler=scheduler)
    finally:
        cleanup_distributed()

    # ────────────────────────────────────────────────────────────────────────────────
    # Save final checkpoint only on main process
    # ────────────────────────────────────────────────────────────────────────────────
    if is_main_process:
        run_path = os.path.join(MODEL_SAVE_PATH, 'run' + str(config['trainer']['run_num']))
        with open(os.path.join(run_path, 'metrics.pickle'), 'wb') as file:
            pickle.dump(metrics, file)
        print('\nFinished Training!', flush=True)

    if is_main_process and config['wandb']['enabled']:
        wandb.finish()

    if is_main_process:
        mlflow_logger.finish()


if __name__ == "__main__":
    main()
