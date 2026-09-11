# ptycho-fm

Physics-informed Vision Transformer with CNN Decoders for Ptychography Reconstruction

## Overview

This project implements **PtychoFM**, a Vision Transformer encoder with CNN decoders for ptychographic image reconstruction. It supports custom and pretrained encoders and combines learned representations with physics-based constraints to reconstruct amplitude and phase information from diffraction patterns.

## Key Features

- **Encoder options**: Custom or pretrained Vision Transformer with CNN decoders
- **Physics-informed**: Enforces ptychographic forward model during training
- **Distributed training**: PyTorch DDP (DistributedDataParallel) for multi-GPU training
- **Flexible configuration**: YAML-based config with unified decoder specification
- **Multi-dataset support**: Automatic discovery and loading of HDF5 file pairs
- **Multiple loss functions**: SmoothL1, MSE, L1, and PoissonNLL
- **Training modes**: Supervised (amp/phase loss) or unsupervised (diffraction loss)
- **Experiment tracking**: Integrated Weights & Biases logging

## Project Structure

```
ptycho-fm/
├── ptycho_fm/                     # Installable package (import ptycho_fm)
│   ├── __init__.py
│   ├── train.py                   # Training entry point (ptycho-fm-train)
│   ├── inference.py               # Inference entry point (ptycho-fm-infer)
│   ├── training.py                # Trainer class with train/validation loops
│   ├── data.py                    # PtychographyDataset, CombinedDataset, RankShardedSubset
│   ├── data_simple_pack.py        # Packed-HDF5 dataset (PtychographyDatasetPacked)
│   ├── custom_loss.py             # Custom loss functions (WeightedLoss)
│   ├── prefetcher.py              # CUDAPrefetcher
│   ├── model/
│   │   ├── model.py               # PtychoFM (unified ViT-based model)
│   │   ├── vit.py                 # Custom ViT encoder
│   │   ├── vit_pretrained.py      # Pretrained ViT encoder
│   │   └── decoders.py            # Decoder256
│   └── utils/
│       ├── math.py                # Coordinate transforms and scan positions
│       └── utils.py               # compute_sha256, misc
├── tests/                         # Tests (not shipped in the wheel)
├── scripts/                       # Standalone CLIs / HPC helpers (not shipped in the wheel)
├── hpc_submission_scripts/        # Cluster submission wrappers
├── docker/                        # Container build
├── configs/                       # Per-cluster configs
├── config.yaml                    # Default training config
└── pyproject.toml                 # Package + dependencies
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

After install, the package is importable as `ptycho_fm`, and two console scripts are on PATH:

- `ptycho-fm-train --config "$PWD/config.yaml"` — training entry point (equivalent to `python -m ptycho_fm.train`)
- `ptycho-fm-infer` — inference entry point (equivalent to `python -m ptycho_fm.inference`)

The distribution is named `ptycho-fm`; the Python import name is `ptycho_fm`.
Existing code should update imports from `ptycho_vit` to `ptycho_fm` and use
the new command names above. Import the model as
`from ptycho_fm.model.model import PtychoFM` (formerly `PtychoViT`).
Developers can access `scripts/` and `visualization.ipynb` from their repository
checkout; these files are not included in the installed wheel.

The class rename preserves the architecture and parameter names. Existing
state-dict weights from the same architecture can be loaded for inference or
fine-tuning with the original model configuration. Training resumes from
`checkpoint_model.pth` plus `checkpoint.state`, which stores optimizer and
scheduler state, epoch, metrics, and the WandB run ID. Keep the same optimizer
parameter groups when resuming. Whole-model pickle files created with
`torch.save(model, ...)` depend on the old class/module path and require
conversion to a state dict in an environment containing the original class.

## Data Format

The dataset expects paired HDF5 files in Ptychodus format:
- `*_dp.hdf5`: Contains diffraction patterns under key `'dp'` with shape `[N, H, W]`
- `*_para.hdf5`: Contains reconstruction parameters

**Required keys in `*_para.hdf5`:**
- `'object'`: Complex object array with shape `[1, H, W]`
  - Attributes: `'pixel_height_m'` (pixel size in meters)
- `'probe'`: Probe function (complex array)
- `'probe_position_x_m'`: Probe x-positions in meters with shape `[N]`
- `'probe_position_y_m'`: Probe y-positions in meters with shape `[N]`

**Directory structure:**
```
data/
├── object1_dp.hdf5
├── object1_para.hdf5
├── object2_dp.hdf5
├── object2_para.hdf5
└── ...
```

## Configuration

All training parameters are configured in `config.yaml`. The configuration uses a unified decoder structure to reduce redundancy.

### Model Architecture

Training and inference use `PtychoFM`. Select its encoder via `encoder_type`:

```yaml
model:
  encoder_type: 'custom'  # Options: 'custom' (train from scratch), 'pretrained' (use pretrained encoder)
```

The model configuration has two main sections:
- `encoder`: Encoder-specific parameters (supports any image size)
- `decoder`: Decoder configuration (independent of encoder)

**Example - PtychoFM (256×256):**
```yaml
model:
  encoder_type: 'custom'
  encoder:
    img_size: 256
    patch_size: 16
    embed_dim: 512
    depth: 12
    num_heads: 8
    dropout: 0.1
    attn_dropout: 0.0
  decoder:
    base_channels: 64
    latent_dim: null      # null = auto-use encoder embed_dim
    num_stages: 4         # Number of upsampling stages (16×16 → 256×256)
    use_batchnorm: true
    dropout: 0.1
```

### Data Configuration

```yaml
data:
  # Option 1: List specific files
  datafiles:
    - '/path/to/object1_dp.hdf5'
    - '/path/to/object2_dp.hdf5'

  # Option 2: Auto-discover all paired files in directory
  # data_path: '/path/to/data/'

  scale: 10000.0
  normalization_dict_path: '/path/to/norm_factors.pkl'  # Optional
  train_split: 0.95
  random_seed: 8

  # DataLoader settings
  num_workers: 4
  prefetch_factor: 10
  pin_memory: true
  drop_last: false
  persistent_workers: false
```

### Training Configuration

```yaml
training:
  mode: 'unsupervised'  # 'supervised' or 'unsupervised'
  batch_size: 64
  learning_rate: 5.0e-4
  epochs: 201
  ngpus: 2
  loss_function: 'l1'  # Options: 'smooth_l1', 'mse', 'l1', 'poisson_nll'
  validation_plot_freq: 10
  resume_from_checkpoint: false
```

## Model Architecture Details

1. **ViT Encoder**: Processes images at the configured resolution (`img_size` and `patch_size`)
   - Example 256×256: 16×16 patches with patch_size=16 → 16×16 feature map
   - Example 512×512: 32×32 patches with patch_size=16 → 32×32 feature map
   - Supports both custom (train from scratch) and pretrained encoders
2. **CNN Decoders**: Two symmetric decoders (amplitude/phase) with configurable stages
   - Set `num_stages` to match the required upsampling from the encoder output
3. **Physics Forward Model**: Enforces ptychographic physics

### Physics Integration

PtychoFM enforces ptychographic physics:
```python
# Subtract probe intensity contribution
x = x - sqrt(probe_intensity)

# Encode to latent representation
latent = encoder(x)

# Decode to amplitude and phase
amp = amp_decoder(latent)      # sigmoid activation
phase = ph_decoder(latent)     # tanh activation, scaled to [-π, π]

# Apply physics forward model
complex_object = amp * exp(i * phase)
Psi = FFT(complex_object * probe)
predicted_diffraction = |Psi|²
```

## Usage

### Single-GPU Training

Run from the repository root to use its `config.yaml`:

```bash
ptycho-fm-train --config "$PWD/config.yaml"
# equivalent:
python -m ptycho_fm.train --config "$PWD/config.yaml"
```

### Multi-GPU Training with DDP

```bash
torchrun --nnodes 1 --nproc-per-node 2 -m ptycho_fm.train --config "$PWD/config.yaml"
```

### Inference

```bash
ptycho-fm-infer
# equivalent:
python -m ptycho_fm.inference
```

## Data Loading

### PtychographyDataset
- Loads probe positions from HDF5 file in meters
- Converts positions to pixels using `pixel_height_m` attribute
- Extracts patches using Fourier shift for sub-pixel accuracy
- Applies Poisson noise to diffraction patterns (optional)
- Efficient caching of object and probe data

### CombinedDataset
- Auto-discovers paired HDF5 files in directory
- Manages multiple `PtychographyDataset` instances
- Provides unified indexing across all files
- Works seamlessly with PyTorch `DataLoader` and `DistributedSampler`

### Normalization
- Per-object normalization factors loaded from pickle file
- Format: `{object_name: normalization_factor}`
- Falls back to default value if object not found

## Training Details

### Distributed Training
- PyTorch DDP for efficient multi-GPU training
- `DistributedSampler` automatically partitions data across GPUs
- Only rank 0 logs to WandB and saves checkpoints
- Gradient synchronization across all processes

### Checkpointing
Checkpoints include:
- Model state dict
- Optimizer state dict
- Epoch number
- Training/validation losses
- Model configuration

Resume training by setting `resume_from_checkpoint: true` in config.

## Experiment Tracking

Weights & Biases integration tracks:
- Training and validation losses
- Amplitude and phase reconstruction errors
- Model architecture and hyperparameters
- Periodic validation visualizations
- System metrics (GPU usage, memory)

Configure in `config.yaml`:
```yaml
wandb:
  enabled: true
  entity: 'your-entity'
  project: 'PtychoFM'
  dataset_name: 'dataset-description'
  notes: 'Experiment notes'
  resume_run_id: null  # Optional: resume existing run
```

## Testing

Run tests to verify functionality (after `pip install -e .`):

```bash
pytest tests/
```

## Development

### Code Organization
- **ptycho_fm/model/model.py**: ViT-based model (PtychoFM - unified, supports any image size)
- **ptycho_fm/model/decoders.py**: CNN decoders used by PtychoFM
- **ptychi.image_proc**: Fourier-shift patch extraction and placement from the ptychi dependency
- **ptycho_fm/utils/math.py**: Coordinate transforms and scan-position utilities
- **ptycho_fm/data.py**: Dataset classes for loading Ptychodus format files
- **ptycho_fm/training.py**: Training and validation logic

### Adding New Models
1. Define model class in `ptycho_fm/model/model.py`
2. Add config section in `config.yaml` under `model:`
3. Update model selection logic in `ptycho_fm/train.py`

## License

This project is under development at the Advanced Photon Source, Argonne National Laboratory.

## Acknowledgments

This implementation adapts code from:
- Ming Du's [pty-chi](https://github.com/AdvancedPhotonSource/pty-chi) for patch extraction via Fourier shift
- Ming Du's [ptycho_simulation_factory](https://github.com/mdw771/ptycho_simulation_factory) for probe position generation
