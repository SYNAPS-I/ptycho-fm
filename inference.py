import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from data import PtychographyDataset
from model import PtychoViT, PtychoViT256
from model_cnn import PtychoCNN, PtychoCNN256

# NEED TO SET YOUR OWN CONFIG AND TEST DATA PATHS
config_path = '/scratch/aileenluo/ptycho-vit/models/run106/config.yaml'
#test_data_path = '/home/beams/AILEENLUO/ptycho_simulation_factory/outputs/horse256/horse256_dp.hdf5'
#test_data_path = '/home/beams/AILEENLUO/ptycho_simulation_factory/outputs/coins256/coins256_dp.hdf5'
#test_data_path = '/scratch/aileenluo/ptycho-vit/data/cameraman256_dp.hdf5'
test_data_path = '/scratch/aileenluo/ptycho-vit/data/brick256_dp.hdf5'

def load_config(path):
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config

config = load_config(config_path)

# Create test dataset
test_dataset = PtychographyDataset(
    test_data_path,
    config['data']['scale'],
    config['data'].get('normalization_dict_path'),
    apply_noise=True
)

BATCH_SIZE = 512
dataloader_kwargs = {
    'batch_size': BATCH_SIZE,
    'num_workers': config['data'].get('num_workers', 0),
    'pin_memory': config['data'].get('pin_memory', True),
}

# Add prefetch_factor and persistent_workers only if num_workers > 0
if dataloader_kwargs['num_workers'] > 0:
    dataloader_kwargs['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
    dataloader_kwargs['persistent_workers'] = config['data'].get('persistent_workers', False)

test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    **dataloader_kwargs
)

# Load model
print('Loading model')
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_path = os.path.join(config['paths']['model_save_path'], 'run' + str(config['trainer']['run_num']))
model.load_state_dict(torch.load(os.path.join(run_path, 'best_model.pth')))
print('Model loaded successfully')

# Make predictions and save directly to disk
def predict_and_save(model, dataloader, output_dir, prefix, img_size, device=DEVICE):
    """Run inference and save results directly to disk using memory-mapped arrays.

    This avoids loading the entire dataset into RAM.

    Args:
        model: The model to use for inference
        dataloader: DataLoader for the dataset
        output_dir: Directory to save results
        prefix: Prefix for output files (e.g., 'train' or 'val')
        device: Device to run inference on
    """
    # Get dataset size and image dimensions
    total_samples = len(dataloader.dataset)

    print(f'Creating memory-mapped arrays for {total_samples} samples...')

    # Create memory-mapped arrays that write directly to disk
    pred_diff = np.lib.format.open_memmap(
        os.path.join(output_dir, f'pred_diff_{prefix}.npy'),
        mode='w+', dtype=np.float32, shape=(total_samples, img_size, img_size)
    )
    pred_amp = np.lib.format.open_memmap(
        os.path.join(output_dir, f'pred_amp_{prefix}.npy'),
        mode='w+', dtype=np.float32, shape=(total_samples, img_size, img_size)
    )
    pred_ph = np.lib.format.open_memmap(
        os.path.join(output_dir, f'pred_ph_{prefix}.npy'),
        mode='w+', dtype=np.float32, shape=(total_samples, img_size, img_size)
    )
    gt_diff = np.lib.format.open_memmap(
        os.path.join(output_dir, f'gt_diff_{prefix}.npy'),
        mode='w+', dtype=np.float32, shape=(total_samples, img_size, img_size)
    )
    gt_amp = np.lib.format.open_memmap(
        os.path.join(output_dir, f'gt_amp_{prefix}.npy'),
        mode='w+', dtype=np.float32, shape=(total_samples, img_size, img_size)
    )
    gt_ph = np.lib.format.open_memmap(
        os.path.join(output_dir, f'gt_ph_{prefix}.npy'),
        mode='w+', dtype=np.float32, shape=(total_samples, img_size, img_size)
    )

    model.to(device)
    model.eval()

    sample_idx = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch

            batch_size = diff_amp.size(0)  # Actual batch size (may be smaller for last batch)

            input_diff = diff_amp.to(device)
            input_probe = torch.view_as_real(probe.clone().detach()).to(device)
            input_norm = norm.to(device)
            input_scale = scale.to(device)

            output_diff, output_amp, output_ph = model(input_diff, input_probe, input_norm, input_scale)

            # Write directly to memory-mapped arrays (which write to disk)
            pred_diff[sample_idx:sample_idx+batch_size] = output_diff.squeeze().detach().cpu().numpy()
            pred_amp[sample_idx:sample_idx+batch_size] = output_amp.squeeze().detach().cpu().numpy()
            pred_ph[sample_idx:sample_idx+batch_size] = output_ph.squeeze().detach().cpu().numpy()
            gt_diff[sample_idx:sample_idx+batch_size] = input_diff.squeeze().detach().cpu().numpy()
            gt_amp[sample_idx:sample_idx+batch_size] = amp_patch.squeeze().detach().numpy()
            gt_ph[sample_idx:sample_idx+batch_size] = ph_patch.squeeze().detach().numpy()

            sample_idx += batch_size

            if i % 50 == 0:
                print(f'Processed batch {i}/{len(dataloader)} ({sample_idx}/{total_samples} samples)')

    # Flush to ensure all data is written to disk
    print(f'Flushing results to disk...')
    pred_diff.flush()
    pred_amp.flush()
    pred_ph.flush()
    gt_diff.flush()
    gt_amp.flush()
    gt_ph.flush()

    print(f'Completed {prefix} inference: {sample_idx} samples saved')

# Save results
RESULTS_PATH = '/scratch/aileenluo/ptycho-vit/results'
results_run = os.path.join(RESULTS_PATH, 'run' + str(config['trainer']['run_num']))
if not os.path.isdir(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)
if not os.path.isdir(results_run):
    os.mkdir(results_run)

print('Running inference on test set...')
predict_and_save(model, test_loader, results_run, 'brick', img_size)