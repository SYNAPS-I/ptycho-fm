# ptycho-vit

Physics-informed Vision Transformer and CNN for Ptychography Reconstruction

## Overview

This project implements physics-informed deep learning architectures for ptychographic image reconstruction. The framework supports two model types:
- **Vision Transformer (ViT)**: Transformer-based encoder with CNN decoders
- **Convolutional Neural Network (CNN)**: Symmetric encoder-decoder architecture

Both models combine learned representations with physics-based constraints to reconstruct amplitude and phase information from diffraction patterns.

## Architecture

### Model Types

Choose between two architectures via `model_type` in `config.yaml`:

#### Vision Transformer (model_type: 'vit')
1. **ViT Encoder** (`vit.py`): Processes diffraction patterns through patch embeddings and transformer blocks
2. **Dual CNN Decoders** (`decoders.py`): Separate decoders for amplitude (sigmoid) and phase (tanh) reconstruction
3. **Physics-Informed Forward Model** (`model.py`): Enforces ptychographic forward physics

#### CNN Architecture (model_type: 'cnn')
1. **CNN Encoder** (`model_cnn.py`): 5-stage convolutional encoder with progressive downsampling (512→16)
2. **Dual CNN Decoders** (`model_cnn.py`): Symmetric decoders for amplitude (sigmoid) and phase (tanh) with upsampling (16→512)
3. **Physics-Informed Forward Model** (`model_cnn.py`): Same ptychographic forward model as ViT

### Key Features

- **Multiple architectures**: Switch between ViT and CNN models via config
- **Distributed training**: PyTorch DDP (DistributedDataParallel) support for multi-GPU training
- **Flexible configuration**: YAML-based config file for all hyperparameters and architecture settings
- **Configurable dropout**: Independent dropout rates for encoder and decoders
- **Multi-dataset support**: Load and train on multiple HDF5 files simultaneously with automatic discovery
- **Efficient data loading**: PyTorch DataLoader with DistributedSampler for DDP training
- **Configurable architecture**: All model layers and dimensions controllable via config
- **Multiple loss functions**: Support for SmoothL1, MSE, L1, and PoissonNLL losses
- **Training modes**: Supervised (amp/phase loss) or unsupervised (diffraction loss)
- **Physics-informed**: Enforces ptychographic forward model in training loop
- **Experiment tracking**: Integrated Weights & Biases logging

## Project Structure

```
ptycho-vit/
├── vit.py                # Vision Transformer encoder implementation
├── decoders.py           # CNN decoder for upsampling to full resolution (ViT)
├── model.py              # PtychoViT model combining encoder, decoders, and physics
├── model_cnn.py          # PtychoCNN model with symmetric encoder-decoder architecture
├── data.py               # PtychographyDataset and CombinedDataset for data loading
├── training.py           # Trainer class with train/validation loops
├── main.py               # Main training script with DDP initialization
├── config.yaml           # Configuration file for all training parameters
├── tests/
│   ├── test_data.py              # Data loading tests
│   ├── test_model_selection.py  # Model selection and initialization tests
│   └── test_utils.py             # Utility function tests
└── pyproject.toml        # Project dependencies
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
- `mode`: 'supervised' (amp/phase loss) or 'unsupervised' (diffraction loss)
- `batch_size`: Samples per batch per GPU
- `learning_rate`: Optimizer learning rate
- `epochs`: Number of training epochs
- `ngpus`: Number of GPUs for DDP
- `loss_function`: Choice of 'smooth_l1', 'mse', 'l1', or 'poisson_nll'
- `validation_plot_freq`: Plot validation results every N epochs
- `resume_from_checkpoint`: Resume from checkpoint if available

**Model Architecture:**
- `model_type`: Choose 'vit' or 'cnn' architecture
- **ViT Configuration**: Fully configurable transformer encoder and CNN decoders
  - Embedding dimension, depth, attention heads, MLP ratio
  - Dropout for positional embeddings and attention
- **CNN Configuration**: Symmetric encoder-decoder with configurable channels
  - Base channels (e.g., 32 or 64) - determines model size
  - Latent dimension at bottleneck (e.g., 256 or 512)
  - Dropout for all convolutional layers (encoder and both decoders)
- All parameters in `config.yaml` under `model` section

## Model Selection

### Choosing Between ViT and CNN

Select model architecture in `config.yaml`:

```yaml
model:
  model_type: 'vit'  # or 'cnn'
```

### Model Comparison

| Model | Parameters | Architecture | Best For |
|-------|-----------|--------------|----------|
| **ViT** | 7.1M | Transformer encoder + CNN decoders | Global context, attention mechanisms |
| **CNN (small)** | 7.5M | base=32, latent=256 | Faster training, local features |
| **CNN (large)** | 29.8M | base=64, latent=512 | High capacity, detailed reconstruction |

**Adjust CNN model size** via `base_channels` and `latent_dim`:
- `base_channels`: 32 (small), 64 (large)
- `latent_dim`: 256 (small), 512 (large)

### Example Configurations

**Small CNN** (7.5M params):
```yaml
model:
  model_type: 'cnn'
  cnn:
    encoder:
      base_channels: 32
      latent_dim: 256
      dropout: 0.1
    amp_decoder:
      base_channels: 32
      latent_dim: 256
      dropout: 0.1
    ph_decoder:
      base_channels: 32
      latent_dim: 256
      dropout: 0.1
```

**ViT** (7.1M params):
```yaml
model:
  model_type: 'vit'
  encoder:
    embed_dim: 192
    depth: 12
    num_heads: 3
    dropout: 0.1
```

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

**Default ViT Configuration** (ViT-Tiny):
- Image size: 512x512
- Patch size: 16x16
- Embedding dimension: 192
- Depth: 12 transformer blocks
- Attention heads: 3
- Decoder layers: [128, 64, 32, 16]

**Default CNN Configuration** (Small):
- Image size: 512x512 (fixed)
- Base channels: 32
- Latent dimension: 256
- Architecture: 5-stage encoder-decoder
- Channel progression: 32 → 64 → 128 → 256 → 256 (latent) → 256 → 128 → 64 → 32

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

### PtychoViT (model.py)

The Vision Transformer-based model integrates:
1. ViT encoder that processes diffraction patterns into latent features via patch embeddings
2. Two parallel CNN decoders for amplitude and phase reconstruction
3. Physics-based forward model applying Fourier transform and probe multiplication
4. Normalization and scaling to match experimental data range

**Parameters:** ~7.1M

### PtychoCNN (model_cnn.py)

The CNN-based model integrates:
1. **Encoder**: 5-stage convolutional encoder with progressive downsampling
   - Architecture: 512×512 → 256×256 → 128×128 → 64×64 → 32×32 → 16×16
   - Each stage: 2 conv blocks + max pooling
   - Configurable dropout after each convolutional layer
2. **Dual Decoders**: Two symmetric decoders for amplitude and phase
   - Architecture: 16×16 → 32×32 → 64×64 → 128×128 → 256×256 → 512×512
   - Each stage: transpose conv + conv block
   - Independent dropout control for each decoder
3. Physics-based forward model (same as PtychoViT)

**Parameters:** ~7.5M (base=32, latent=256) or ~29.8M (base=64, latent=512)

### Physics Integration

Both models enforce ptychographic physics:
```python
complex_object = amp * exp(i * phase)
Psi = FFT(complex_object * probe)
predicted_intensity = |Psi|^2
```

Models output: predicted diffraction amplitude, reconstructed amplitude, and reconstructed phase.

## Testing

Run the test suite to verify model selection and functionality:

```bash
# Test model selection between ViT and CNN
python tests/test_model_selection.py

# Test data loading
python tests/test_data.py

# Test utility functions
python tests/test_utils.py
```

The `test_model_selection.py` verifies:
- Both ViT and CNN models initialize correctly
- Forward pass produces expected output shapes
- Dropout configurations work properly
- Parameter counts match expected values

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
- Model type (ViT or CNN) and architecture configuration
- All hyperparameters from config (including dropout rates)
- Learning rate (if using scheduler)
- Periodic validation visualizations
- Training/validation plots

Configure in `config.yaml` under `wandb` section:
```yaml
wandb:
  enabled: true
  entity: 'your-entity'
  project: 'PtychoViT'
  dataset_name: 'your-dataset'
  notes: 'Experiment description'
```

## License

This project is under development at the Advanced Photon Source.

## Acknowledgments

This implementation adapts code from:
- Ming Du's [pty-chi](https://github.com/AdvancedPhotonSource/pty-chi) for patch extraction via Fourier shift
- Ming Du's [ptycho_simulation_factory](https://github.com/mdw771/ptycho_simulation_factory) for probe position generation
