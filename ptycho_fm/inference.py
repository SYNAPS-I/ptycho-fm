import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ptycho_fm.data import PtychographyDataset
from ptycho_fm.model.model import PtychoFM
from ptycho_fm.utils.config import resolve_run_name, run_directory


def main() -> None:
    """Entry point for `ptycho-fm-infer` and `python -m ptycho_fm.inference`."""
    config_path = '/global/cfs/cdirs/m5073/pecomyint/ptycho-vit/scratch/models/run1/config.yaml'
    test_data_path = '/global/cfs/cdirs/m5073/synaps_data/simulated_data/n07581931_1001_dp.hdf5'

    def load_config(path):
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            return config

    config = load_config(config_path)
    run_name = resolve_run_name(config, generate=False)

    # Create test dataset
    test_dataset = PtychographyDataset(
        test_data_path,
        config['data']['scale'],
        '/home/beams/AILEENLUO/ptycho_simulation_factory/test2probes_norm.pkl',
        #config['data'].get('normalization_dict_path'),
        apply_noise=config['data'].get('test_apply_noise', False),
        cache_object=config['data'].get('cache_object', True),
        cache_memory_budget_mb=config['data'].get('cache_memory_budget_mb', 512),
        max_probe_modes=config['data'].get('max_probe_modes', 8),
        max_OPR_modes=config['data'].get('max_OPR_modes', 1),
    )

    BATCH_SIZE = 256
    dataloader_kwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': 0,  # Must be 0 for inference with memory-mapped arrays to avoid bus errors
        'pin_memory': False,  # Disable pin_memory to reduce memory pressure
    }

    # Note: num_workers is forced to 0 for inference because memory-mapped arrays
    # created in the main process cannot be safely accessed from DataLoader worker processes

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    # Load model
    print('Loading model')
    img_size = config['model']['encoder']['img_size']
    model = PtychoFM(config=config['model'])

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clear any cached memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f'GPU memory before loading model: {torch.cuda.memory_allocated()/1024**3:.2f} GiB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GiB reserved')

    run_path = str(run_directory(config))
    model.load_state_dict(torch.load(os.path.join(run_path, 'best_model.pth'), map_location=DEVICE, weights_only=True))
    print('Model loaded successfully')

    if torch.cuda.is_available():
        print(f'GPU memory after loading model: {torch.cuda.memory_allocated()/1024**3:.2f} GiB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GiB reserved')

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

                # Explicitly delete tensors and clear cache to prevent memory buildup
                del input_diff, input_probe, input_norm, input_scale
                del output_diff, output_amp, output_ph
                del diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale

                if device.type == 'cuda' and i % 10 == 0:
                    torch.cuda.empty_cache()

                sample_idx += batch_size

                if i % 10 == 0:
                    print(f'Processed batch {i}/{len(dataloader)} ({sample_idx}/{total_samples} samples)')
                    if device.type == 'cuda':
                        print(f'  GPU memory: {torch.cuda.memory_allocated(device)/1024**3:.2f} GiB allocated, {torch.cuda.memory_reserved(device)/1024**3:.2f} GiB reserved')

        # Flush to ensure all data is written to disk
        print('Flushing results to disk...')
        pred_diff.flush()
        pred_amp.flush()
        pred_ph.flush()
        gt_diff.flush()
        gt_amp.flush()
        gt_ph.flush()

        print(f'Completed {prefix} inference: {sample_idx} samples saved')

    # Save results
    RESULTS_PATH = '/scratch/aileenluo/ptycho-vit/results'
    results_run = os.path.join(RESULTS_PATH, 'run' + run_name)
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
    if not os.path.isdir(results_run):
        os.mkdir(results_run)

    print('Running inference on test set...')
    predict_and_save(model, test_loader, results_run, 'scan807', img_size)


if __name__ == "__main__":
    main()
