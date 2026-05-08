# ptycho-vit

Physics-informed Vision Transformer and CNN for Ptychography Reconstruction

## Overview

This project implements physics-informed deep learning architectures for ptychographic image reconstruction. The framework supports multiple model variants optimized for different image sizes and computational requirements:

- **PtychoViT**: Unified Vision Transformer encoder with CNN decoders (supports any image size via config)
- **PtychoCNN** (512×512): Symmetric CNN encoder-decoder architecture
- **PtychoCNN256** (256×256): CNN encoder-decoder with log-polar transform

All models combine learned representations with physics-based constraints to reconstruct amplitude and phase information from diffraction patterns.

## Key Features

- **Multiple architectures**: Four model variants optimized for different image sizes
- **Log-polar preprocessing**: Enhanced feature extraction for 256×256 models
- **Physics-informed**: Enforces ptychographic forward model during training
- **Distributed training**: PyTorch DDP (DistributedDataParallel) for multi-GPU training
- **Flexible configuration**: YAML-based config with unified decoder specification
- **Multi-dataset support**: Automatic discovery and loading of HDF5 file pairs
- **Multiple loss functions**: SmoothL1, MSE, L1, and PoissonNLL
- **Training modes**: Supervised (amp/phase loss) or unsupervised (diffraction loss)
- **Experiment tracking**: Integrated Weights & Biases logging

## Project Structure

```
ptycho-vit/
├── vit.py                # Vision Transformer encoder
├── model.py              # PtychoViT model (unified ViT-based model)
├── model_cnn.py          # PtychoCNN and PtychoCNN256 models, Encoder/Decoder classes
├── ptychi_utils.py       # Image processing utilities (Fourier shift, patch extraction)
├── data.py               # PtychographyDataset and CombinedDataset
├── training.py           # Trainer class with train/validation loops
├── main.py               # Main training script with DDP initialization
├── inference.py          # Inference script for model evaluation
├── config.yaml           # Configuration file for all parameters
├── custom_loss.py        # Custom loss functions
├── tests/
│   ├── test_data.py              # Data loading tests
│   ├── test_model_selection.py  # Model selection tests
│   └── test_utils.py             # Utility function tests
├── development_logs/     # Archive of deprecated code (gitignored)
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

Select model type via `model_type`:

```yaml
model:
  encoder_type: 'custom'  # Options: 'custom' (train from scratch), 'pretrained' (use pretrained encoder)
```

The model configuration has two main sections:
- `encoder`: Encoder-specific parameters (supports any image size)
- `decoder`: Decoder configuration (independent of encoder)

**Example - PtychoViT (256×256):**
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

## Model Comparison

| Model | Image Size | Parameters | Key Features |
|-------|-----------|------------|--------------|
| **PtychoViT** | Configurable | 7M - 50M+ | ViT encoder, flexible architecture, log-polar transform |
| **PtychoCNN** | 512×512 | 7.5M - 30M | CNN encoder-decoder, configurable size |
| **PtychoCNN256** | 256×256 | ~3M | CNN encoder-decoder, log-polar transform |

### Model Architecture Details

#### PtychoViT (Unified, Configurable)
1. **Log-polar Transform**: Preprocesses diffraction patterns to enhance rotational features
2. **ViT Encoder**: Processes images at any resolution (configurable via `img_size` and `patch_size`)
   - Example 256×256: 16×16 patches with patch_size=16 → 16×16 feature map
   - Example 512×512: 32×32 patches with patch_size=16 → 32×32 feature map
   - Supports both custom (train from scratch) and pretrained encoders
3. **CNN Decoders**: Two symmetric decoders (amplitude/phase) with configurable stages
   - Automatically adjusts upsampling based on encoder output size
4. **Physics Forward Model**: Enforces ptychographic physics

#### PtychoCNN (512×512)
1. **CNN Encoder**: 5-stage encoder with progressive downsampling
   - 512×512 → 256×256 → 128×128 → 64×64 → 32×32 → 16×16
2. **Dual Decoders**: Symmetric decoders for amplitude and phase
   - 5 upsampling stages: 16×16 → 512×512
3. **Configurable Size**: Adjust via `base_channels` and `latent_dim`

#### PtychoCNN256 (256×256)
1. **Log-polar Transform**: Preprocesses diffraction patterns
2. **CNN Encoder**: 5-stage encoder: 256×256 → 8×8
3. **Dual Decoders**: 5 upsampling stages: 8×8 → 256×256
4. **Compact Design**: Optimized for smaller images

### Physics Integration

All models enforce ptychographic physics:
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

```bash
python main.py
```

### Multi-GPU Training with DDP

```bash
torchrun --nnodes 1 --nproc-per-node 2 main.py
```

### Inference

```bash
python inference.py --model_path /path/to/checkpoint.pt --config config.yaml
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

### TiledPtychographyDataset

Sibling of `PtychographyDataset` that reads a per-run holoptycho container
from a Tiled server instead of paired HDF5 files. Same `__getitem__` contract
(7-tuple, identical shapes/dtypes), so it drops into the same training loop.

Only runs marked `fine_tunable: true` in their Tiled metadata are accepted —
that flag is set by holoptycho when `recon_mode` is `iterative` or `both`,
since fine-tuning needs the reconstructed `final/probe` and `final/object`
as supervised targets.

```python
from data_tiled import TiledPtychographyDataset

ds = TiledPtychographyDataset(
    tiled_uri="https://tiled.nsls2.bnl.gov/api/v1/metadata/hxn/processed/holoptycho/<run_uid>/",
)
```

#### Authentication

Auth precedence is **explicit `api_key=` arg > `TILED_API_KEY` env var > cached `tiled login` credentials**.

For training jobs (Slurm, CI, containers) the env-var form is the most
convenient — no code change between dev and prod, no secret in the script:

```bash
export TILED_API_KEY=<your-key>
python training.py  # TiledPtychographyDataset picks up the key automatically
```

For interactive use on a dev box, run `tiled login <server>` once and the
cached credentials get picked up; `api_key` can be omitted.

For one-off explicit overrides, pass `api_key=` directly:

```python
ds = TiledPtychographyDataset(tiled_uri="...", api_key="...")
```

#### Exporting a Tiled run to HDF5 (Polaris / air-gapped compute)

Polaris compute nodes can't reach the Tiled server. For training there, run
`scripts/export_tiled_to_hdf5.py` on a login node — which has internet — to
materialize the run as the paired `*_dp.hdf5` / `*_para.hdf5` files
`PtychographyDataset` already reads. The training job then sees a normal
local-file dataset; no Tiled access at runtime.

```bash
export TILED_API_KEY=<your-key>
python scripts/export_tiled_to_hdf5.py <run_uid> \
    --out-dir /lus/eagle/projects/<your-project>/holoptycho-runs
# Produces <scan_id>_dp.hdf5 + <scan_id>_para.hdf5 under --out-dir.
```

Point your training config's `data.data_path` at the output directory and
launch the job as usual. Run the export once per scan you want to fine-tune
on; the files are reusable across runs.

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
  project: 'PtychoViT'
  dataset_name: 'dataset-description'
  notes: 'Experiment notes'
  resume_run_id: null  # Optional: resume existing run
```

## Testing

Run tests to verify functionality:

```bash
# Test model selection and initialization
python tests/test_model_selection.py

# Test data loading
python tests/test_data.py

# Test utility functions
python tests/test_utils.py
```

## Development

### Code Organization
- **model.py**: ViT-based model (PtychoViT - unified, supports any image size)
- **model_cnn.py**: CNN-based models (PtychoCNN, PtychoCNN256) and shared Encoder/Decoder classes
- **ptychi_utils.py**: Image processing utilities adapted from pty-chi
- **data.py**: Dataset classes for loading Ptychodus format files
- **data_tiled.py**: `TiledPtychographyDataset` — reads holoptycho runs from a Tiled server
- **scripts/export_tiled_to_hdf5.py**: stage a Tiled run to `*_dp.hdf5` + `*_para.hdf5` for offline training (Polaris, etc.)
- **training.py**: Training and validation logic

### Adding New Models
1. Define model class in `model.py` or `model_cnn.py`
2. Add config section in `config.yaml` under `model:`
3. Update model selection logic in `main.py`

## License

This project is under development at the Advanced Photon Source, Argonne National Laboratory.

## Acknowledgments

This implementation adapts code from:
- Ming Du's [pty-chi](https://github.com/AdvancedPhotonSource/pty-chi) for patch extraction via Fourier shift
- Ming Du's [ptycho_simulation_factory](https://github.com/mdw771/ptycho_simulation_factory) for probe position generation
