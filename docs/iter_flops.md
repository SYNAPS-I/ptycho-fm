# Iteration and isoFLOP training

The features from `ss/iter-flops` use the current `ptycho_fm` package, model implementations, `Trainer`, and training entry point. Epoch training remains the default. The custom ViT and independent standard amplitude/phase decoders support analytical budgets; other model variants remain available with FLOP tracking disabled.

## Training and configuration

YAML supports a single `extends` path relative to the child file. Nested mappings merge recursively; lists replace their parents. CLI config paths remain relative to the caller's working directory. Data/output paths keep their existing working-directory semantics.

Start from `configs/isoflop/site.example.yaml`, extend a selected model YAML, and specify your actual data, normalization, output, tracking, and run identifiers. The 13 model configurations under `configs/isoflop/` inherit an explicit experiment base. Its placeholders are not production datasets.

```bash
python -m ptycho_fm.train --config my_site.yaml
bash hpc_submission_scripts/submit_isoflop.sh my_site.yaml
python -m scripts.sweep_flops --configs configs/isoflop/e512_d12.yaml \
  --budgets 1e18 3e18 --world-size 16 --batch-size 32
```

The sweep report assumes full batches; the trainer accounts for the batches actually processed. Specify the intended rank count explicitly; historical Figure 2 runs used different hardware from the isoFLOP sweep.

| Setting under `training` | Meaning |
| --- | --- |
| `log_every: 0` | Validation interval in processed batches; zero disables interval logging. |
| `log_at_flops: []` | Raw FLOP targets. When supplied, these replace `log_every` triggers. |
| `track_flops: false` | Track compute without requiring targets. |
| `stop_at_last_flop_target: false` | Explicitly stop after the final target. Enabled in the isoFLOP base. |
| `max_iters: null` | Optional total processed-batch limit, including pre-resume work. |
| `valid_batch_size` | Optional validation batch size; defaults to training batch size. |

Iteration mode also validates at epoch boundaries and at an explicit stop. Crossing several targets in one batch produces one validation snapshot listing all crossed targets. CSV, W&B and MLflow share the scalar metrics, successful update count, achieved compute, processed samples, batch/rank metadata and loss-averaging interval. CSV additionally records accounting provenance and the full crossed-target list. Training loss is averaged over rank batches in the interval; validation loss is a fresh snapshot. `samples_seen` includes distributed-sampler padding, and `tokens_seen` uses actual processed samples and configured patch geometry.

Use `lr_scheduler.scheduler_class: warmup-stable` with `kwargs.warmup_steps` for update-based warmup. Use ordinary Torch scheduler names for existing epoch/validation scheduling. A skipped optimizer update still consumes forward/backward compute, but does not advance an update-based LR schedule.

## Compute convention

`ptycho_fm.utils.flops.PtychoFMFlopsCalculator` returns **TFLOPs**, while configured targets are raw **FLOPs** (`1 TFLOP = 1e12 FLOPs`). Accounting version `ptycho_fm_v1` follows the current custom ViT and twin standard decoders:

- Two operations per multiply-add; estimated training cost is three forward passes.
- Attention scaling and residual additions are included. Per-element activation estimates: GELU 8, softmax 5, tanh 10, custom activation 12, sigmoid 4.
- Training BatchNorm includes statistics. Its per-channel overhead means the cost of a batch is not exactly batch size times a one-pattern estimate. Live training sums each rank's actual batch cost, including partial batches.
- Scope: encoder input through decoder output activations. Excludes input preprocessing, external output normalization arithmetic, physics propagation/loss, optimizer updates, validation, dropout RNG and memory traffic.
- Parameter counts include all three registered output-normalization parameters, including frozen ones.

Historical CSV compute/parameter columns remain recorded values. Missing legacy provenance is labeled `legacy_unspecified`; reconstructing an absent parameter count is explicitly labeled `current_model_reconstruction`.

`profile_fvcore()` is a lazy optional diagnostic using current modules in evaluation mode. It returns native fvcore totals (one operation per fused multiply-add), per-operator counts and unsupported operators, including fused attention when unavailable in fvcore. These partial diagnostic totals never determine budgets. Install `fvcore` separately if needed; analytical calculation does not import it or instantiate models.

## Resume and cooldown

Checkpoints contain format-version 2 training state, optimizer parameter names/shapes, scheduler state, rank-specific RNG and progress, interval accumulators, and model/data/batch/rank/scheduler compatibility metadata. Dataset identity hashes ordered file paths, sizes, nanosecond modification times and pattern offsets; it does not hash entire HDF5 payloads. Split seed and settings are checked too.

`checkpoint_model.pth` and `checkpoint.state` are the current resume pair. Iteration validation also writes matching `model_iters_N.pth` and `state_iters_N.pth`. A mid-epoch stop saves the consumed position rather than advancing the epoch. Set `resume_from_checkpoint: true` using the same run directory. Limits may be extended; model, dataset, batch/rank settings and scheduler must match.

Exact mid-epoch continuation requires dynamic sharding. If Poisson noise is enabled, the original run must use `data.deterministic_noise: true`; `noise_seed + global_sample_index` determines noise independently of workers/ranks. Static mid-epoch continuation is rejected. Empty or unequal rank loader lengths are rejected before entering training collectives.

Existing epoch checkpoints may load with compatible optimizer groups/shapes in epoch mode, but lack the new exact-continuation guarantees. Remote legacy iteration checkpoints require an **explicit conversion** of model keys, optimizer groups, progress/accounting and sufficient provenance; they are not silently treated as current state. `scripts/migrate_checkpoint.py` migrates model weights, not optimizer/RNG/compute state. Weights-only loading via `finetune_from_model` starts fresh training and is the supported fallback when legacy state cannot be established. No exact historical RNG or compute is invented.

For cooldown, select a new run identifier and a matching parent iteration state:

```yaml
training:
  resume_from_checkpoint: false
  stop_at_last_flop_target: false
  max_iters: null
  lr_scheduler:
    enabled: true
    scheduler_class: cooldown
    kwargs:
      branch_from: /path/to/parent/run1/state_iters_4000.pth
      cooldown_steps: 1000
```

The corresponding model file must exist alongside the state. Cooldown retains parent optimizer/progress/compute and starts a new square-root decay from the parent LR. It stops after 1000 additional successful updates in this example, unless an explicit iteration/compute limit stops it first. `epochs` must allow sufficient remaining work. Resuming this cooldown uses its own run directory and unchanged cooldown configuration.

## Collection, fits and Figure 4

```bash
uv sync --group analysis
python -m scripts.isoflop --configs my_site.yaml --targets 1e18 3e18 \
  --summary-csv output/isoflop_points.csv --output-dir output/figures
python -m notebooks.manuscript.plot_isoflop \
  --input tests/fixtures/isoflop_points.csv --output-dir output/historical
```

The collector prefers saved resolved run configs and finds the nearest valid log row within 5% compute tolerance (configurable). Latest duplicate iterations are authoritative. Training and validation metrics are handled independently. Canonical fits are log-log quadratics with model-size optima constrained to the observed range, paired training length interpolated in log space, and explicit sparse/non-convex outcomes. Collector `--fit-space log_params` or `--fit-space params` adds diagnostic plots; canonical optimum tables still use log-log fits.

`notebooks/manuscript/fig4.ipynb` shares these calculations while retaining manuscript token plots, SVGs, epoch markers and scaling fits. Run from the repository or notebook directory. Its default data/output directory is `workspace/paper`; set `PTYCHO_FM_PAPER_DIR` to use another directory. New run records must supply actual `tokens_seen`. Historical sweep records retain the documented 512 images/iteration and 256 patch tokens/image.

The Figure 2 historical compute tables within Figure 4 use **token-normalized current-model estimates**, not reconstructed iteration totals. A current one-pattern training reference is divided by 256 patch tokens, then multiplied by documented token totals. Hardware metadata does not change this approximation:

| History | Sharding | Ranks | Images/rank |
| --- | --- | --- | --- |
| 43M pretraining | dynamic | 1024 | likely 32 |
| 200M pretraining | static | 1024 | likely 32 |
| Both fine-tunes | dynamic | 24 | 128 |

Fine-tuning assumes incomplete final batches were retained. Sampler padding maps 44,827 unique patterns to 44,832 processed patterns/epoch for 101 epochs; those processed tokens feed the historical estimate. The pretraining batch size remains explicitly uncertain metadata. Existing historical isoFLOP point CSVs are unchanged. Tracked fixtures record their provenance and numerical fit baseline.

## Port audit and validation

| Incoming change | Treatment |
| --- | --- |
| `3509758` iteration stack, resume, utilities | Adapted into current train/Trainer and package utilities; no second training stack. |
| `8ed54b4`, `d214f57` sweep and iteration bounds | Explicit-config sweep CLI with world-size/batch/budget/bounds. |
| `67b16b5`, `8d965cd` configs/paths | 13 model YAMLs moved under `configs/isoflop`, shared experiment base and site overrides. |
| `90be4db` cooldown | Current checkpoints/models, fresh schedule, new run guard. |
| `331f238` commented QK normalization | Omitted: no runtime effect; current ViT retained byte-for-byte. |
| `7bd55fa` FLOP corrections | Re-derived from current ViT/decoder operations and approved convention. |
| `34c3fa3` target logging | Actual rank-batch compute, synchronized targets, explicit stop. |
| `ff8910e` deterministic noise | Adapted to current native and packed readers. |
| `a897761` collection/plots | Shared collector/fits, thin CLIs, historical fixture; generated PNG omitted. |
| Legacy scheduler/HPC paths | Current module entry point retained; explicit-config submission wrapper. |

Validation commands:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests -q
ruff check ptycho_fm scripts/isoflop.py scripts/sweep_flops.py \
  notebooks/manuscript/fig4.ipynb notebooks/manuscript/plot_isoflop.py
uv lock --check --offline
```

The entry-point tests use the real model, deterministic native HDF5 data and one/two CPU Gloo ranks; they compare uninterrupted and resumed weights, RNG, compute and CSVs, target crossing/stopping, partial final batches and cooldown. Other tests cover skipped updates, configuration, parameter counts, packed/native data, sharding, fine-tuning, checkpoint migration and historical fits. Figure 4 and the plot CLI are rendered against the historical fixture in temporary directories.

Validation results: 135 full-suite tests passed; the seven analysis tests also passed after the final degenerate-fit guard was added. Figure 4, collection from new run logs, all three diagnostic fit spaces, the plot/sweep CLIs, and optional fvcore profiling passed their smoke checks. The profiler reports its unsupported fused operators and preserves RNG state.

Validation limitations: GPU/NCCL, cluster submission and live W&B/MLflow services are not exercised. Full-repository Ruff has 45 existing findings in unrelated files, all present on the original branch (58 findings before this integration). Changed code passes Ruff. A separate pre-existing unsupervised plain-MSE dtype failure occurs when float64 normalization promotes predicted diffraction while the target is float32; the original branch reproduces it, and the default weighted-loss training passes. Model code and this unrelated loss behavior remain unchanged.
