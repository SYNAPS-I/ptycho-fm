# PtychoViT — HXN Beamline Fine-tuning Guide

This guide walks HXN beamline users through fine-tuning PtychoViT on their own experimental data, from a clean login to a running training job.

---

## Prerequisites

- Access to an NSLS2 compute node with at least one GPU
- Your HXN scan data converted to the Ptychodus HDF5 format (`*_dp.hdf5` + `*_para.hdf5` pairs)
- The pre-trained model checkpoint (`best_model.pth`)

---

## 1. Clone the Repository

```bash
git clone https://github.com/SYNAPS-I/ptycho-vit.git
cd ptycho-vit
```

---

## 2. Install Dependencies with uv

[uv](https://docs.astral.sh/uv/) is available system-wide on NSLS2 nodes.

```bash
# Verify uv is available
uv --version

# Create a virtual environment and install all dependencies from the lockfile
uv sync

# Activate the environment
source .venv/bin/activate
```

> **Note:** `uv sync` reads `uv.lock` for exact reproducible versions. Do not use `pip install -r requirements.txt` — use `uv sync` instead.

> **Each new session:** re-run `source .venv/bin/activate` before using `python`. All `python` commands in this guide assume the environment is active.

To deactivate the environment later:

```bash
deactivate
```

---

## 3. Prepare Your Data

### 3.1 Convert HXN raw data to ptycho-vit format

`conv_working.py` converts your HXN reconstruction outputs into the `*_dp.hdf5` / `*_para.hdf5` pairs that ptycho-vit expects.

**Required inputs** (produced by `run-ptycho-backend`):

```
ptycho/                              # working directory set `mode` to this in the script
├── scan_404589.h5                   # raw HXN HDF5 for each scan
└── recon_result/
    └── S404589/
        └── t2/                      # reconstruction sign (t1, t2, …)
            └── recon_data/
                ├── 404589_t2.ptycho_hyan_*.txt   # reconstruction settings
                ├── recon_404589_t2_probe.npy
                └── recon_404589_t2_object.npy
```

**Configure and run:**

Open `conv_working.py` and set the variables at the bottom of the file:

```python
mode = "ptycho"            # the subdirectory name under your current working directory
single_scan_only = False   # True to process one scan; False to process all in parallel
```

`mode` is the name of the subdirectory that holds your `scan_*.h5` files and `recon_result/`. For example, if your data lives at:

```
/data/users/2026Q1/ZP_Comm_2026Q1/ptycho/scan_404589.h5
/data/users/2026Q1/ZP_Comm_2026Q1/ptycho/recon_result/...
```

then `mode = "ptycho"` and `/data/users/2026Q1/ZP_Comm_2026Q1/` is where you `cd` to.

For a single scan (useful for testing before batch processing):

```python
single_scan_only = True    # processes only the first scan found (sorted numerically)
```

To target a specific scan ID instead of the first one, also change this line:

```python
scan_id = tgt_scanids[0]   # replace with e.g. 404589 to pick a specific scan
```

Once all variables are set, `cd` to the parent of `ptycho/` and run the script:

```bash
cd /data/users/2026Q1/ZP_Comm_2026Q1/
python /path/to/conv_working.py
```

**Outputs** are written relative to where you `cd`'d. If you don't set a custom path:
- **Single scan** — saves to `converted_data/` → `/data/users/2026Q1/ZP_Comm_2026Q1/converted_data/`
- **Batch** — saves to `converted_data_{mode}_fix/` → `/data/users/2026Q1/ZP_Comm_2026Q1/converted_data_ptycho_fix/`

To use a custom path, edit the script directly:
- **Single scan** — edit the `process_data(...)` call and add `out_dir='/your/output/path/'`
- **Batch** — change `f"converted_data_{mode}_fix/"` in the `pool.apply_async(...)` args to your desired path

Each scan is saved in its own subfolder:

```
converted_data/
└── 404589/
    ├── 404589_dp.hdf5              # diffraction intensities, shape (N, 256, 256)
    ├── 404589_para.hdf5            # object, probe, positions
    └── 404589_probe_positions.png  # diagnostic plot
```

Move or symlink the output files into a single flat directory before pointing `config.yaml` at them.

> **Note:** If the raw diffraction patterns are not 256×256, the script automatically crops or pads them and re-runs `run-ptycho-backend` to refine the probe/object on the new grid. This requires `run-ptycho-backend` to be available on `$PATH`.

---

### 3.2 Expected file layout

Your data directory should contain paired HDF5 files:

```
/path/to/your/data/
├── 272460_dp.hdf5       # diffraction patterns — key: 'dp', shape (N, 256, 256), float32
├── 272460_para.hdf5     # reconstruction parameters (object, probe, positions)
├── 272461_dp.hdf5
├── 272461_para.hdf5
└── ...
```

The `*_para.hdf5` files must contain:

| Key | Shape | dtype | Notes |
|-----|-------|-------|-------|
| `object` | `(1, H, W)` | `complex64` | Attributes: `pixel_height_m`, `pixel_width_m`, `center_x_m`, `center_y_m` |
| `probe` | `(1, n_modes, H, W)` | `complex64` | e.g. `(1, 1, 256, 256)` for single-mode probe |
| `probe_position_indexes` | `(N,)` | `int64` | Index into the full position grid |
| `probe_position_x_m` | `(N,)` | `float64` | Probe x-positions in metres |
| `probe_position_y_m` | `(N,)` | `float64` | Probe y-positions in metres |

### 3.2 Create the normalization dictionary

The normalization dictionary maps each scan name to its maximum photon count. Run once per dataset:

```bash
python scripts/create_normalization.py /path/to/your/data/
```

By default the output is saved as `normalization.pkl` inside the data directory. To specify a different output path:

```bash
python scripts/create_normalization.py /path/to/your/data/ \
    --output /path/to/your/data/normalization.pkl
```

Point `normalization_dict_path` in `config.yaml` to this file.

---

## 4. Configure Fine-tuning

Edit `config.yaml`. The minimum set of fields to change for a new fine-tuning run:

```yaml
# ── Data ──────────────────────────────────────────────────────────────────────
data:
  data_path: '/path/to/your/training/data/'   # directory with *_dp.hdf5 files
  test_path:  '/path/to/your/test_dp.hdf5'    # single file used for test plots

  normalization_dict_path: '/path/to/your/data/normalization.pkl'
  test_normalization:      '/path/to/your/test/normalization.pkl'

  apply_noise: false            # HXN experimental data already contains noise
  transpose_object_patches: true  # required for HXN data axis convention

# ── Training ──────────────────────────────────────────────────────────────────
training:
  mode: 'unsupervised'          # recommended for experimental data
  batch_size: 64
  learning_rate: 1.0e-4
  epochs: 100

  finetune_from_model: '/path/to/best_model.pth'    # pre-trained checkpoint

  # Learning rate scheduler
  lr_scheduler:
    enabled: true
    scheduler_class: 'CosineAnnealingLR'
    kwargs:
      T_max: 100       # match total epochs
      eta_min: 1.0e-5  # floor LR

  # Weighted loss (only used when loss_function: 'weighted')
  weighted_loss:
    loss_type: 'mse'
    threshold: 0.0
    alpha: 0.0         # 0 = uniform weighting; 1 = inverse intensity weighting (1/intensity)

  val_plot_sample_idx: 0  # REQUIRED — global index in the validation set to plot each epoch;
                          # must be set to an integer, null will cause an error

# ── Output ────────────────────────────────────────────────────────────────────
paths:
  model_save_path: '/path/to/save/finetuning_results/'

# ── Run identifier ────────────────────────────────────────────────────────────
trainer:
  run_num: '001'    # change this for each experiment to avoid overwriting
```

> **Tip:** Keep `training.mode: 'unsupervised'` for real experimental data. This uses the diffraction loss (forward physics model) rather than requiring a reference reconstruction.

---

## 5. Verify the Setup

Before launching a long job, run the sanity-check script on a login or interactive node:

```bash
python scripts/test_finetuning_setup.py --config config.yaml
```

This checks:
1. Config paths resolve and files exist
2. HDF5 data loads correctly
3. Pre-trained weights load into the model
4. A forward pass completes without error
5. SSIM/PSNR metrics compute correctly
6. A real data inference sample runs end-to-end

Expected output ends with:
```
ALL TESTS PASSED!
You're ready to run fine-tuning:
  python main.py --config config.yaml
```

Add `--skip-real-inference` to skip the GPU-heavy forward pass on a CPU-only login node:

```bash
python scripts/test_finetuning_setup.py --config config.yaml --skip-real-inference
```

---

## 6. Run Fine-tuning

### 6a. Interactive session (single node)

Request an interactive GPU node, then launch training:

```bash
# Single GPU
python main.py --config config.yaml

# Multi-GPU on one node (e.g. 2 GPUs)
torchrun --nnodes 1 --nproc-per-node 2 main.py --config config.yaml
```

Set `training.ngpus` in `config.yaml` to match the number of GPUs you request.

---

## 7. Outputs

Checkpoints and plots are saved to the directory set in `paths.model_save_path`:

```
finetuning_results/
├── model_epoch_001.pth        # model weights only, every epoch
├── model_epoch_002.pth
├── ...
├── checkpoint_epoch_001.pth   # full training state (model + optimizer)
├── best_model.pth             # weights at the epoch with lowest validation loss
└── val_plot_epoch_*.png       # validation reconstruction plots
```

To resume an interrupted run, set `training.resume_from_checkpoint: true` in `config.yaml`.

---

## 8. Export Model to ONNX

After fine-tuning, export the trained checkpoint to ONNX for deployment (e.g. edge inference):

```bash
python scripts/convert_pt_to_onnx.py \
    --config /path/to/run/config.yaml \
    --checkpoint /path/to/run/best_model.pth \
    --output /path/to/run/model_b64.onnx \
    --batch-size 64 \
    --output-kind amp_phase
```

The `.onnx` file is saved to whatever path you pass to `--output`. Parent directories are created automatically if they don't exist. It is recommended to save it alongside the checkpoint for easy reference, e.g. `run042901/run042901_b64.onnx`.

The exported model takes diffraction patterns as input and returns the reconstruction directly — no probe or normalization tensors needed.

**`--batch-size`** bakes a **fixed** batch size into the ONNX graph. With `--batch-size 64` the model always expects input shape `[64, 1, 256, 256]` at inference time. Use `--dynamic-batch` instead if you want to support any batch size.

**`--output-kind`** controls what the model returns:

| Value | Output shape | Description |
|-------|-------------|-------------|
| `phase` (default) | `[B, 1, H, W]` | Phase only |
| `amplitude` | `[B, 1, H, W]` | Amplitude only |
| `amp_phase` | `[B, 2, H, W]` | Amplitude and phase stacked |

**Other useful options:**

```bash
# Flexible batch size at inference time — just add the flag, no value needed
--dynamic-batch

# Skip ONNX validation check after export
--no-verify

# Pass a run directory instead of a specific file;
# it will automatically find best_model.pth inside
--checkpoint /path/to/finetuning_results/
```

---

## 9. Register the ONNX Model to Azure ML

Once the `.onnx` file is exported, register it to the **Genesis MLflow** workspace on Azure ML Studio (Brookhaven National Laboratory):

1. Go to [Azure ML Studio](https://ml.azure.com) and open the **Genesis MLflow** workspace
2. In the left sidebar, go to **Assets → Models**
3. Click **+ Register → From local files**
4. Fill in the form:
   - **Name**: e.g. `ptycho_vit_amp_phase_b64`
   - **Model type**: `Custom`
   - **File**: upload your `.onnx` file
5. Click **Register**

The model will appear in the Model List and can be deployed or shared with other users in the workspace.

