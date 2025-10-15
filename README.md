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

- Patch-based transformer architecture for processing 512x512 single-channel diffraction images
- Physics-informed loss incorporating forward diffraction simulation
- Support for multiple ViT sizes (tiny, small, base, large)
- Efficient HDF5 data loading with caching for large datasets

## Project Structure

```
ptycho-vit/
├── vit.py           # Vision Transformer encoder implementation
├── decoders.py      # CNN decoder for upsampling to full resolution
├── model.py         # PtychoViT model combining encoder, decoders, and physics
├── data.py          # Dataset loader for ptychography HDF5 files
├── training.py      # Training utilities and loops
├── main.py          # Main training script
└── pyproject.toml   # Project dependencies
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.6.0
- NumPy >= 2.3.3
- h5py (for data loading)
- torchinfo (for model summaries)
- wandb (for experiment tracking)

## Installation

```bash
pip install -e .
```

## Data Format

The dataset expects paired HDF5 files in Ptychodus format:
- `*_dp.hdf5`: Contains diffraction patterns
- `*_para.hdf5`: Contains probe positions, object amplitude/phase, and probe information

## Usage

### Training

Configure training parameters in `main.py` and run:

```bash
python main.py
```

Key hyperparameters:
- `BATCH_SIZE`: Batch size per GPU
- `LR`: Learning rate
- `EPOCHS`: Number of training epochs
- `NGPUS`: Number of GPUs for data parallel training

### Model Configuration

The current model uses a tiny ViT configuration:
- Image size: 512x512
- Patch size: 16x16
- Embedding dimension: 192
- Depth: 12 transformer blocks
- Attention heads: 3

## Training Details

- **Loss function**: SmoothL1Loss on predicted vs. actual diffraction amplitudes
- **Optimizer**: Adam
- **Multi-GPU**: Supports DataParallel training
- **Logging**: Weights & Biases integration for experiment tracking
- **Validation**: Periodic evaluation with visualization

## Model Architecture Details

### PtychoViT (model.py:6-49)

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

## TODOs

- [ ] Add configuration file (config.yaml) for hyperparameters
- [ ] Make model architecture layers configurable
- [ ] Implement learning rate scheduler
- [ ] Add support for additional data formats

## License

This project is under development at the Advanced Photon Source.

## Acknowledgments

This implementation adapts code from:
- Ming Du's [pty-chi](https://github.com/AdvancedPhotonSource/pty-chi) for patch extraction
- Ming Du's [ptycho_simulation_factory](https://github.com/mdw771/ptycho_simulation_factory) for probe position generation
