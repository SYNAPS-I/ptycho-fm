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
- MLflow >= 3.0.0
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
scheduler state, epoch, metrics, and per-rank random state. Keep the same optimizer
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
  apply_noise: true        # Add synthetic noise to training and validation data
  test_apply_noise: false  # Add synthetic noise to test/inference data independently
  train_split: 0.95  # Fraction of complete objects assigned to training
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
  validation_plot_num_objects: 1  # Complete held-out objects to stitch; null logs all
  validation_plot_central_crop: 64
  validation_plot_object_crop: 180
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

## Iteration and isoFLOP training

See [iteration training and isoFLOP analysis](docs/iter_flops.md) for configuration inheritance, compute accounting, exact mid-epoch resume, cooldown, sweep/plot commands, and historical Figure 4 conventions. Epoch training remains the default.

## Experiment Tracking

Training supports either Weights & Biases or MLflow. Enable exactly one backend
and comment out the unused backend's complete configuration block. The shared
`tracking` section controls parameter, scalar metric, artifact, system metric,
and per-batch logging identically for both backends.

```yaml
trainer:
  # null creates YYYYMMDD-HHMMSS for a fresh run; set the saved name to resume
  run_name: null

tracking:
  dataset_name: 'dataset-description'
  notes: 'Experiment notes'
  log_parameters: true
  log_metrics: true
  log_artifacts: true
  log_system_metrics: true
  log_every_n_batches: 50  # Set to null to disable batch metrics

wandb:
  enabled: true
  entity: 'your-entity'
  project: 'ptycho-fm'

# mlflow:
#   enabled: true
#   tracking_uri: 'http://127.0.0.1:5000'  # Or set MLFLOW_TRACKING_URI
#   experiment_name: 'ptycho-fm'
#   tags: {}
#   azureml_compat: false  # Enable only for Azure ML registry endpoints
```

The resolved `trainer.run_name` is shared by the local `run<name>` directory,
MLflow, and W&B. On `resume_from_checkpoint: true`, set `run_name` to the
existing timestamp; the tracker run is looked up by that name, so no tracker ID
is required. Run names must be unique within the selected tracker experiment or
project.

For epoch-only tracking, set `tracking.log_every_n_batches: null` and leave
`training.log_every: 0` and `training.log_at_flops: []`. Training and validation
metrics are then emitted after each completed epoch. Set `training.log_every` to
a positive iteration interval, or populate `training.log_at_flops`, to enable
the iteration/FLOP validation and logging path instead. All scalar metric steps
refer to completed iterations.

Setting `tracking_uri` configures the MLflow client; it does not start a
tracking server. Start the repository's local server launcher with:

```bash
./scripts/start_mlflow_server.sh
```

The host and port defaults are editable near the top of the script. They can
also be overridden for one run, for example:

```bash
MLFLOW_SERVER_HOST=0.0.0.0 MLFLOW_SERVER_PORT=5001 ./scripts/start_mlflow_server.sh
```

`127.0.0.1` permits local clients only; `0.0.0.0` accepts connections on all
interfaces and should be protected by appropriate firewall and authentication
controls. Alternatively, point `tracking_uri` at an existing accessible
server. The Azure ML registry compatibility workaround is disabled by default
and should be enabled only for an Azure ML endpoint.

Both backends receive the same flattened run parameters, metric names, batch
cadence, global iteration step, configuration artifact, and validation/test
plots. Backend-native presentation still differs:

| Capability | W&B | MLflow |
| --- | --- | --- |
| Plots | Interactive media entries | Files under `val_plots/` and `test_plots/` |
| Configuration | Versioned W&B artifact | File under the `config/` artifact path |
| System metrics | W&B system monitor | MLflow system-metrics collector |
| Resume tracking run | Lookup by `trainer.run_name` | Lookup by `trainer.run_name` |
| Model registry | Not wired | Available in `MLflowLogger`, but not called by training |


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
