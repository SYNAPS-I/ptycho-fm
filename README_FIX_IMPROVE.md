# Fixes and Improvements Log

This file documents non-trivial bugs that were fixed and improvements that were
made while adapting ptycho-vit for HXN experimental data (scan 405667).  It is
intended as a reference for anyone finetuning the model on a new beamline or
dataset.

---

## Critical: `amp_offset` / `amp_scale` must match the sample's amplitude range

### What goes wrong

The model's final amplitude output is:

```
amp = constrained_amp × amp_scale + amp_offset
```

where `constrained_amp ∈ (−1.716, +1.716)` (saturation limit of
`CustomActivation`, which is `1.7159 × tanh(2x/3)`).

The defaults — `amp_scale = 0.1`, `amp_offset = 0.975` — were chosen for
simulated APS data where samples have mild absorption and transmittance near 1.
They produce an output range of roughly **[0.80, 1.15]**.

HXN scan 405667 is a strongly absorbing sample.  The ptychographic
reconstruction stored in `para.hdf5` has:

| quantity | value |
|---|---|
| mean `|object|` | 0.46 |
| max `|object|` | 5.72 |
| model output range | **[0.80, 1.15]** |

The sample region sits entirely *below* the model's minimum output.  No amount
of gradient descent can fix this because the output is saturated at the
`CustomActivation` boundary — the model is physically incapable of producing
the correct amplitudes.  In practice the predicted amplitude map is a nearly
uniform blob around 0.80 and the Fourier-consistency loss is minimised by
using the phase degree of freedom to compensate, producing ring/probe-shaped
phase artefacts.

### How to fix

In the finetune config (`config.yaml`), add the `amp_offset` and `amp_scale`
keys under `model:` so they match the expected transmittance range of the
sample:

```yaml
model:
  amp_offset: 0.5      # centre of the expected amplitude range
  amp_scale:  0.6      # half-width; (amp_offset ± amp_scale × 1.716) covers [0, 1+]
  ph_scale:   3.14159  # leave as π unless sample phase exceeds ±π
```

For a sample with transmittance in `[0, 1]`, `amp_offset = 0.5` and
`amp_scale = 0.6` gives an output range of `[−0.03, 1.03]`, which is
appropriate.  Adjust empirically based on the sample and beamline.

### Why `log_scale_amp` in the checkpoint is not the problem here

`log_scale_amp` is an `nn.Parameter` with `requires_grad=False` and is
therefore saved in — and restored from — the checkpoint state dict.  However,
neither the APS training config nor the HXN finetune config sets `amp_scale`
explicitly, so both use the code default of `0.1`.  The APS checkpoint was
therefore trained with `log_scale_amp = log(0.1)`, and loading it produces the
same value that instantiation would have produced anyway.  The checkpoint load
is a no-op for this parameter in the current setup.

The practical risk is: if a *future* checkpoint is trained with a non-default
`amp_scale`, loading it will silently override whatever value you set in the
config.  To avoid surprises, always set `amp_scale` and `amp_offset` in the
config explicitly for every finetune run, and re-initialise the decoder weights
rather than carrying over scale parameters from an APS checkpoint.

`amp_offset` is **not** an `nn.Parameter` — it is a plain Python float read
from the config at model instantiation.  It is never persisted to or loaded
from a checkpoint.  This means it can be changed safely in the config without
worrying about the checkpoint.

---

## Fix #1: scan-position alignment (bounding-box centering)

Two upstream conventions exist for how probe positions are stored:

- **Simulated data** — positions are centered around `(0, 0)` in the object
  frame, so the dataset adds `object_shape / 2` to shift them to the
  top-left-origin pixel frame expected by `extract_patches_fourier_shift`.
- **HXN converted data** (`hxn_to_vit.py`) — positions are already in
  pixel units measured from the top-left corner of the object, so no offset
  should be applied.

The old code always added `object_shape / 2`, which doubled the offset for HXN
data and placed all probes outside the object boundaries (resulting in all-zero
GT patches during validation).

**Fix** (`data.py → _cache_positions`): the offset is now computed as
`obj_center − scan_bounding_box_center`.  For simulated data the scan is
centered in the object so this reproduces the old `+ object_shape/2` exactly.
For HXN data the scan already sits near the object center so the correction is
close to zero.

---

## Fix #2: `WeightedLoss` mask and weights must come from the measurement

The original weighted loss derived the pixel mask and inverse-intensity weights
from the *model output* (`input` tensor in `forward`).  This created a
positive-feedback loop: the model could reduce its loss by predicting ≈ 0
wherever it struggled, which masked those pixels and removed the gradient
signal entirely.  On HXN data with many true-zero detector pixels this was
catastrophic.

**Fix** (`custom_loss.py`): mask and weights are now computed from `target`
(the measured diffraction pattern).  The measurement is fixed, so the weighting
cannot be gamed by the model.

---

## Fix #3: SSIM variance numerical stability

Computing local variance as `E[X²] − E[X]²` in float32 can produce slightly
negative values when the local window is nearly constant.  This caused the SSIM
map to occasionally exceed its theoretical `[−1, 1]` range.

**Fix** (`training.py → compute_ssim`): variance terms are clamped to `≥ 0`,
and the final SSIM map is clamped to `[−1, 1]` as a safety guard.

---

## Fix #9: HXN object patch transposition

The ptychographic object stored by HXN reconstruction software has its row and
column axes swapped relative to the probe and diffraction-pattern frames.
Extracting a patch at position `(y, x)` without correction produces an object
patch in the transposed frame; multiplying it with the probe and FFT-ing gives
a forward-model residual roughly 40 % larger than after transposition.

**Fix**: `PtychographyDataset` accepts a `transpose_object_patches: true` flag
(set in the HXN finetune `config.yaml`).  After `extract_patches_fourier_shift`
the patch is transposed back into the probe/DP frame.  The `Trainer` carries the
same flag to un-transpose patches before accumulating them in
`generate_test_plot` and `generate_plot`, so the visualised GT and predicted
images are in the correct on-disk frame.

---

## Stitching improvements (`utils/ptychi_utils.py` → `stitch_patches`)

### Corner-placement bug

The previous stitching code placed patches at the bottom-right corner of the
canvas rather than at positions derived from the actual scan coordinates.
This was replaced by `stitch_patches`, which auto-calculates the canvas size
from the bounding box of the provided probe positions.

### Hann apodization window

Each patch is multiplied by a 2-D Hann (raised-cosine) window before being
scatter-added into the canvas.  In the overlap regions the Hann-weighted
average emphasises pixels where the model has full spatial context (centre of
patch) and de-emphasises edges.  Amplitude stitching uses a standard
weighted mean:

```
stitched[y, x] = Σ_i  w_i(y, x) · patch_i(y, x)  /  Σ_i  w_i(y, x)
```

### Circular mean for phase stitching

Naïve averaging of phase values fails near the ±π discontinuity (e.g., the
average of +3.1 and −3.1 is 0, not ±π).  Phase stitching now uses the
circular mean:

```
stitched_phase = atan2( Σ w·sin(φ),  Σ w·cos(φ) )
```

This is correct for any phase in `[−π, π]` regardless of wrap-around.

### `canvas_pad = 64` in `generate_test_plot`

The Fourier shift used to place each patch with sub-pixel accuracy introduces
wrap-around artefacts at the canvas edges.  Adding 64 pixels of padding on
each side of the canvas keeps those artefacts outside the region of interest.
The padding is trimmed before saving the figure.

### Dynamic colour limits

`generate_test_plot` now computes `vmin`/`vmax` independently for GT and
predicted rows using the same method (1st/99th percentile for amplitude,
mean ± 2 σ for phase), cropped by `ph_crop` pixels to exclude the noisy
canvas border.  This ensures the GT row is not washed out by predicted
outliers and vice versa.
