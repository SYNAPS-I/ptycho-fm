import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.distributed as dist
import pickle
import yaml

from dataloader import CombinedDataset
from model import PtychoViT
from training import Trainer

import wandb

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load configuration
config = load_config()

# Training parameters
NGPUS = config['training']['ngpus']
BATCH_SIZE = config['training']['batch_size']
LR = config['training']['learning_rate']
EPOCHS = config['training']['epochs']
MODEL_SAVE_PATH = config['paths']['model_save_path']

# Initialize DDP first (if using multiple GPUs)
if NGPUS > 1:
    dist.init_process_group()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = rank == 0
    # Set CUDA device for this process before any distributed operations
    torch.cuda.set_device(local_rank)
    # Keep this print for all ranks - useful for debugging DDP setup
    print(f"Rank: {rank}, Local Rank: {local_rank}")
    DEVICE = torch.device(f"cuda:{local_rank}")
else:
    rank = 0
    world_size = 1
    is_main_process = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create CombinedDataset with DDP support
# Support both 'datafiles' (list) and 'data_path' (directory)
if 'datafiles' in config['data'] and config['data']['datafiles'] is not None:
    data_source = config['data']['datafiles']
elif 'data_path' in config['data']:
    data_source = config['data']['data_path']
else:
    raise ValueError("Config must specify either 'datafiles' (list) or 'data_path' (directory)")

combined_dataset = CombinedDataset(
    file_paths=data_source,
    train_split=config['data']['train_split'],
    batch_size=BATCH_SIZE,
    rank=rank,
    world_size=world_size,
    shuffle=True,
    random_seed=config['data']['random_seed'],
    normalization_dict_path=config['data'].get('normalization_dict_path')
)

# Print configuration only on main process
if is_main_process:
    print("=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"Batch size: {BATCH_SIZE} | Learning rate: {LR}")
    print(f"Epochs: {EPOCHS} | GPUs: {NGPUS}")
    print(f"Loss function: {config['training']['loss_function']}")
    if isinstance(data_source, list):
        print(f"Data source: List of {len(data_source)} file(s)")
    else:
        print(f"Data source: {data_source}")
    print(f"Number of files: {len(combined_dataset.file_paths)}")
    print(f"Total batches/epoch (train): {combined_dataset.get_num_batches('train')}")
    print(f"Total batches/epoch (val): {combined_dataset.get_num_batches('val')}")
    print(f"Device: {DEVICE}")
    print(f"Model save path: {MODEL_SAVE_PATH}")
    print("=" * 50)

# Model setup with config
model = PtychoViT(config=config['model'])
if is_main_process:
    img_size = config['model']['encoder']['img_size']
    dummy_data = torch.randn((1, 1, img_size, img_size))
    dummy_probe = torch.randn((1, 1, 8, img_size, img_size, 2))
    summary(model, input_data={'x': dummy_data, 'probe': dummy_probe,
            'normalization': torch.randn((1, 1)), 'scale': torch.randn((1, 1))}, device='cpu')

# Move model to device and wrap with DDP
model = model.to(DEVICE)
if NGPUS > 1:
    model = DDP(model, device_ids=[local_rank])

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

# Initialize wandb only on main process
if is_main_process and config['wandb']['enabled']:
    wandb.login()
    run = wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        config={
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "dataset": config['wandb']['dataset_name'],
            "epochs": EPOCHS,
            "notes": config['wandb']['notes'],
            "model_config": config['model']
        }
    )

trainer = Trainer(
    model,
    config['trainer']['run_num'],
    DEVICE,
    MODEL_SAVE_PATH,
    is_main_process=is_main_process,
    use_ddp=(NGPUS > 1)
)

if is_main_process:
    print('\nStarting Training...\n')

for epoch in range(EPOCHS):
    # Verify split integrity if debug mode is enabled
    debug_mode = config['training'].get('debug', False)
    if debug_mode and is_main_process:
        combined_dataset.verify_split_integrity()
        print(f"[DEBUG] Epoch {epoch}: Train/val split integrity verified")

    # Training loop
    model.train()
    train_batches = combined_dataset.iterate_batches(split='train', epoch=epoch, debug=debug_mode)
    trainer.train(train_batches, criterion, optimizer, metrics)

    # Validation loop
    model.eval()
    val_batches = combined_dataset.iterate_batches(split='val', epoch=epoch, debug=debug_mode)
    plot = (epoch % config['training']['validation_plot_freq'] == 0)
    trainer.validate(val_batches, criterion, optimizer, metrics, plot=plot)

    if is_main_process:
        print('Epoch: %d | Train Loss: %.4f | Val. Loss: %.4f'
              %(epoch, metrics['training_loss'][-1], metrics['validation_loss'][-1]))

# Save final checkpoint only on main process
if is_main_process:
    trainer.save_model_and_states_checkpoint(epoch, metrics, optimizer, scheduler=None)
    with open(os.path.join(MODEL_SAVE_PATH, 'metrics.pickle'), 'wb') as file:
        pickle.dump(metrics, file)
    print('\nFinished Training!')

if is_main_process and config['wandb']['enabled']:
    wandb.finish()