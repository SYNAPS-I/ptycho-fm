# ptycho-vit

Physics-informed Vision Transformer for Ptychography Reconstruction

## Overview

This project implements a physics-informed Vision Transformer (ViT) architecture for ptychographic image reconstruction. The model combines a standard ViT encoder with physics-based constraints in the decoder to reconstruct both amplitude and phase information from diffraction patterns.

## Architecture

The model consists of three main components:

1. **Vision Transformer Encoder** (`vit.py`): Processes diffraction patterns through patch embeddings and transformer blocks
2. **Dual CNN Decoders** (`decoders.py`): Separate decoders for amplitude (sigmoid activation) and phase (tanh activation) reconstruction
3. **Physics-Informed Forward Model** (`model.py`): Enforces ptychographic forward physics by computing predicted diffraction patterns from reconstructed object

### Key Features

- **Distributed training**: PyTorch DDP (DistributedDataParallel) support for multi-GPU training
- **Flexible configuration**: YAML-based config file for all hyperparameters and architecture settings
- **Multi-dataset support**: Load and train on multiple HDF5 files simultaneously with automatic discovery
- **Efficient data loading**: PyTorch DataLoader with DistributedSampler for DDP training
- **Configurable architecture**: All model layers and dimensions controllable via config
- **Multiple loss functions**: Support for SmoothL1, MSE, L1, and PoissonNLL losses
- **Physics-informed**: Enforces ptychographic forward model in training loop
- **Experiment tracking**: Integrated Weights & Biases logging

## Project Structure

```
ptycho-vit/
├── vit.py           # Vision Transformer encoder implementation
├── decoders.py      # CNN decoder for upsampling to full resolution
├── model.py         # PtychoViT model combining encoder, decoders, and physics
├── data.py          # PtychographyDataset and CombinedDataset for data loading
├── training.py      # Trainer class with train/validation loops
├── main.py          # Main training script with DDP initialization
├── config.yaml      # Configuration file for all training parameters
└── pyproject.toml   # Project dependencies
```

## Requirements

- Python >= 3.11
- PyTorch == 2.6.0
- NumPy >= 2.3.3
- h5py >= 3.15.0
- matplotlib >= 3.10.7
- torchinfo >= 1.8.0
- wandb >= 0.22.2

## Installation

```bash
pip install -e .
```

## Data Format

The dataset expects paired HDF5 files in Ptychodus format:
- `*_dp.hdf5`: Contains diffraction patterns under key 'dp'
- `*_para.hdf5`: Contains object (complex), probe positions, and probe information

**Data Structure:**
```
object_name/
├── object_name_dp.hdf5     # Diffraction patterns [N, H, W]
└── object_name_para.hdf5   # Object data:
                            #   - 'object': Complex object [1, H, W]
                            #   - 'probe': Probe function (optional)
```

## Configuration

All training parameters are configured in `config.yaml`:

### Key Configuration Sections

**Data Configuration:**
- `data_path`: Directory containing paired HDF5 files (auto-discovers all files)
- `datafiles`: Alternatively, specify explicit list of files
- `normalization_dict_path`: Path to pickle file with per-object normalization factors
- `train_split`: Train/validation split ratio (e.g., 0.9 for 90% train, 10% validation)
- `random_seed`: Seed for reproducible splits

**DataLoader Settings:**
- `num_workers`: Number of worker processes for data loading (0 = main process only)
  - Recommended: 2-4 for CPU data loading, 0 for fast storage or small datasets
- `prefetch_factor`: Batches to prefetch per worker (only if num_workers > 0)
  - Recommended: 2-10, increase if GPU is waiting for data
- `pin_memory`: Use pinned memory for faster CPU-to-GPU transfer
  - Recommended: true for GPU training
- `persistent_workers`: Keep workers alive between epochs (only if num_workers > 0)
  - Recommended: true to avoid worker startup overhead

**Training Configuration:**
- `batch_size`: Samples per batch per GPU
- `learning_rate`: Optimizer learning rate
- `epochs`: Number of training epochs
- `ngpus`: Number of GPUs for DDP
- `loss_function`: Choice of 'smooth_l1', 'mse', 'l1', or 'poisson_nll'
- `validation_plot_freq`: Plot validation results every N epochs
- `debug`: Enable split integrity verification

**Model Architecture:**
- Fully configurable encoder (ViT) dimensions
- Customizable decoder layer dimensions and activations
- All parameters in `config.yaml` under `model` section

## Usage

### Single-GPU Training

```bash
python main.py
```

### Multi-GPU Training with DDP

```bash
torchrun --nnodes 1 --nproc-per-node 2 main.py
```

This will launch distributed training across 2 GPUs on a single node.

### Model Configuration

Default configuration (ViT-Tiny):
- Image size: 512x512
- Patch size: 16x16
- Embedding dimension: 192
- Depth: 12 transformer blocks
- Attention heads: 3
- Decoder layers: [128, 64, 32, 16]

All parameters can be modified in `config.yaml`.

## Training Details

### Data Loading
- **PyTorch DataLoader**: Standard PyTorch data loading with configurable num_workers
- **DistributedSampler**: Automatic data partitioning across GPUs for DDP training
- **On-demand file access**: HDF5 files opened and closed per sample access for thread safety
- **Fixed splits**: Train/val split determined once and preserved across epochs
- **Normalization support**: Per-object normalization factors from pickle dictionary
- **Multi-file support**: CombinedDataset transparently handles multiple HDF5 file pairs

### Training Loop
- **Loss calculation**: Compares predicted diffraction amplitude to ground truth
- **Physics enforcement**: Forward model computes diffraction from reconstructed object
- **Optimizer**: Adam
- **Validation**: Periodic evaluation with optional visualization
- **Checkpointing**: Saves model, optimizer state, and metrics

### Distributed Training
- PyTorch DDP for efficient multi-GPU training
- Gradient synchronization across all GPUs
- **DistributedSampler**: Automatically partitions data across ranks and handles shuffling
- Only rank 0 process logs to WandB and saves checkpoints
- All ranks process data in parallel
- DataLoader automatically handles batch count equalization across ranks

## Model Architecture Details

### PtychoViT (model.py:7-81)

The main model integrates:
1. ViT encoder that processes diffraction patterns into latent features
2. Two parallel decoders for amplitude and phase reconstruction
3. Physics-based forward model applying Fourier transform and probe multiplication
4. Normalization and scaling to match experimental data range

### Physics Integration

The forward model enforces ptychographic physics:
```python
complex_object = amp * exp(i * phase)
Psi = FFT(complex_object * probe)
predicted_intensity = |Psi|^2
```

The model outputs predicted diffraction amplitude, reconstructed amplitude, and reconstructed phase.

## Dataset Classes

### PtychographyDataset (data.py)
- Handles single paired HDF5 file
- Opens and closes file on each __getitem__ call for thread safety
- Caches object and probe data for efficient access
- Extracts patches using Fourier shift for sub-pixel accuracy
- Applies Poisson noise to diffraction patterns

### CombinedDataset (data.py)
- PyTorch Dataset for multiple HDF5 file pairs
- Auto-discovers paired files from directory
- Manages multiple PtychographyDataset instances
- Provides unified indexing across all files
- Works seamlessly with PyTorch DataLoader and DistributedSampler

## Experiment Tracking

Weights & Biases integration tracks:
- Training and validation losses
- Amplitude and phase reconstruction losses
- Model architecture configuration
- All hyperparameters from config
- Periodic validation visualizations

Configure in `config.yaml` under `wandb` section.

## Development Status

See `TODO.md` for current development priorities and completed features.

## License

This project is under development at the Advanced Photon Source.

## Acknowledgments

This implementation adapts code from:
- Ming Du's [pty-chi](https://github.com/AdvancedPhotonSource/pty-chi) for patch extraction via Fourier shift
- Ming Du's [ptycho_simulation_factory](https://github.com/mdw771/ptycho_simulation_factory) for probe position generation
