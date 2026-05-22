# All image processing utilities adapted from Ming Du's Pty-chi
# https://github.com/AdvancedPhotonSource/pty-chi
# Copyright © 2025 UChicago Argonne, LLC All right reserved

# Position creation for raster scans adapted from Ming Du's ptycho_simulation_factory
# https://github.com/mdw771/ptycho_simulation_factory

import numpy as np
import torch
from torch import Tensor
from typing import Optional, Tuple, Literal


def make_hann_window(h: int, w: int, device=None) -> Tensor:
    """Return a 2-D Hann (raised-cosine) apodization window of shape (h, w).

    Values range smoothly from 0 at every edge to 1 at the centre.  Multiplying
    each predicted patch by this window before accumulation removes hard-boundary
    artefacts without discarding any content.  When patches overlap, the weighted
    average naturally emphasises pixels where the model has full spatial context.
    """
    wy = torch.hann_window(h, periodic=False, device=device)  # (h,)
    wx = torch.hann_window(w, periodic=False, device=device)  # (w,)
    return wy.unsqueeze(1) * wx.unsqueeze(0)                  # (h, w)


def _place_patches_no_shift(
    image: Tensor,
    positions: Tensor,
    patches: Tensor,
) -> Tensor:
    """Integer-position scatter-add with no Fourier shift.

    Intended for weight/count accumulators where sub-pixel precision in the
    denominator is unnecessary.  Avoids an FFT pass compared to
    place_patches_fourier_shift, saving ~1/3 of stitching time.

    Parameters
    ----------
    image : Tensor
        (H, W) accumulator canvas.
    positions : Tensor
        (N, 2) centre positions (y, x) in pixels.
    patches : Tensor
        (N, H_p, W_p) or (H_p, W_p) values to scatter-add.

    Returns
    -------
    Tensor
        Updated (H, W) canvas.
    """
    if patches.ndim == 2:
        patches = patches.unsqueeze(0).expand(len(positions), -1, -1)

    h_p, w_p = patches.shape[-2:]
    sys = (positions[:, 0] - (h_p - 1.0) / 2.0).round().to(torch.int32)
    sxs = (positions[:, 1] - (w_p - 1.0) / 2.0).round().to(torch.int32)

    pl = [
        max(int(-sxs.min()), 0),
        max(int((sxs + w_p).max() - image.shape[1]), 0),
        max(int(-sys.min()), 0),
        max(int((sys + h_p).max() - image.shape[0]), 0),
    ]
    if any(p > 0 for p in pl):
        image = torch.nn.functional.pad(image, pl)
        sys = sys + pl[2]
        sxs = sxs + pl[0]

    image = batch_put(image, patches, sys, sxs, op="add")

    h, w = image.shape
    image = image[
        pl[2] : h - pl[3] if pl[3] > 0 else h,
        pl[0] : w - pl[1] if pl[1] > 0 else w,
    ]
    return image


# Cheng-Chu
def stitch_patches(patches, positions, patch_size=256, image_shape=None,
                    batch_size=1024, crop=0, pad=4, canvas_pad=0,
                    mode="amplitude", device=None, patch_weights=None):
    """
    Stitch patches using Fourier shift for sub-pixel accuracy.

    Overlapping patches are combined via a weighted average.  The weight at
    each pixel reflects where the beam was actually illuminating:

    * **probe weight** (preferred): pass the real-space probe intensity
      ``|probe|²`` as ``patch_weights``.  This is zero outside the beam,
      peaked where illumination is strongest, and adapts automatically to any
      probe geometry (zone-plate annulus, Gaussian, Airy disk, …).  Only
      pixels with actual beam exposure contribute to the reconstruction.

    * **Hann fallback**: when ``patch_weights`` is None a 2-D Hann window is
      used instead — a generic smooth taper that suppresses boundary artefacts
      but does not encode beam shape.

    For phase patches, a circular mean (via complex exponentials) is used in
    overlapping regions to handle ±π phase wrapping correctly:
        stitched_phase = atan2( Σ w·sin(φ),  Σ w·cos(φ) )

    Args:
        patches: Array of shape (N, H, W) containing predicted patches.
        positions: Array of shape (N, 2) containing (y, x) centre positions in pixels.
        patch_size: Original patch size (documentation only; actual size inferred
            from patches after any crop).
        image_shape: Optional (H, W) output canvas size. Auto-calculated if None.
        batch_size: Number of patches processed per FFT batch (default 1024).
        crop: Hard pixels to remove from each edge before stitching (default 0).
        pad: Fourier-shift padding to suppress wrap-around artefacts (default 4).
        canvas_pad: Extra border pixels added to the canvas on all sides (default 0).
        mode: "amplitude" (default) — weighted mean.
              "phase" — circular mean via atan2(Σ w·sin φ, Σ w·cos φ).
        device: Torch device for canvas accumulation. Defaults to CPU.
        patch_weights: Optional real-valued weight map applied to every patch
            before accumulation.  Shape: (H_p, W_p) for a single shared weight
            (same for all patches) or (N, H_p, W_p) for per-patch weights.
            If crop > 0 and patch_weights is (H_orig, W_orig), the crop is
            applied automatically.  Normalised internally so max == 1.
            When None, falls back to a 2-D Hann apodization window.

    Returns:
        stitched: Full stitched image (H_out, W_out) as a numpy array.
        weights: Weight denominator map (H_out, W_out) as a numpy array.
    """
    if device is None:
        device = torch.device("cpu")

    # Optional hard pre-crop (applied to patches AND to patch_weights if 2-D)
    if crop > 0:
        patches = patches[:, crop:-crop, crop:-crop]

    h_patch, w_patch = patches.shape[-2], patches.shape[-1]
    half_h = h_patch // 2
    half_w = w_patch // 2

    # Convert to tensors
    if not isinstance(patches, torch.Tensor):
        patches = torch.from_numpy(patches).float()
    if not isinstance(positions, torch.Tensor):
        positions = torch.from_numpy(positions).float()

    # Prepare weight map: probe intensity (preferred) or Hann fallback
    if patch_weights is not None:
        if not isinstance(patch_weights, torch.Tensor):
            patch_weights = torch.from_numpy(np.asarray(patch_weights)).float()
        # A (1, H, W) weight is logically a shared 2-D map — squeeze the leading dim
        if patch_weights.ndim == 3 and patch_weights.shape[0] == 1:
            patch_weights = patch_weights.squeeze(0)
        # Apply the same crop to the weight map when it is 2-D
        if patch_weights.ndim == 2 and crop > 0:
            patch_weights = patch_weights[crop:-crop, crop:-crop]
        elif patch_weights.ndim == 3 and crop > 0:
            patch_weights = patch_weights[:, crop:-crop, crop:-crop]
        # Normalise so maximum weight == 1 (shape is preserved)
        pw_max = patch_weights.max()
        if pw_max > 0:
            patch_weights = patch_weights / pw_max
        weight_map_2d = patch_weights.ndim == 2  # True → same weight for all patches
        weight_map_gpu = patch_weights.to(device) if weight_map_2d else patch_weights
        weight_map_cpu = patch_weights.cpu()
    else:
        weight_map_2d = True
        weight_map_gpu = make_hann_window(h_patch, w_patch, device=device)
        weight_map_cpu = weight_map_gpu.cpu()

    # Auto-calculate canvas from position extents
    if image_shape is None:
        min_y = positions[:, 0].min() - half_h
        max_y = positions[:, 0].max() + half_h
        min_x = positions[:, 1].min() - half_w
        max_x = positions[:, 1].max() + half_w

        canvas_h = int(np.ceil(max_y - min_y)) + 2 + 2 * canvas_pad
        canvas_w = int(np.ceil(max_x - min_x)) + 2 + 2 * canvas_pad
        image_shape = (canvas_h, canvas_w)

        offset_y = -min_y + 1 + canvas_pad
        offset_x = -min_x + 1 + canvas_pad
        positions = positions.clone()
        positions[:, 0] = positions[:, 0] + offset_y
        positions[:, 1] = positions[:, 1] + offset_x

    weight_source = "probe |ψ|²" if patch_weights is not None else "Hann window"
    n_patches = len(patches)
    print(f"Stitching ({mode}, weight={weight_source}): {n_patches} patches "
          f"({h_patch}×{w_patch} after crop={crop}) -> {image_shape[0]}×{image_shape[1]} canvas")
    print(f"  Position range: Y=[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}], "
          f"X=[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")

    # Value accumulators live on `device` (GPU when available, saves FFT time).
    # Weight accumulator stays on CPU: integer scatter-add needs no FFT.
    weight_acc = torch.zeros(image_shape)  # always CPU
    if mode == "phase":
        cos_acc = torch.zeros(image_shape, device=device)
        sin_acc = torch.zeros(image_shape, device=device)
    else:
        value_acc = torch.zeros(image_shape, device=device)

    for start in range(0, n_patches, batch_size):
        end = min(start + batch_size, n_patches)
        b = end - start
        batch_patches = patches[start:end].to(device)
        batch_positions = positions[start:end].to(device)

        # Build weight for this batch
        if weight_map_2d:
            w_exp     = weight_map_gpu.unsqueeze(0).expand(b, -1, -1)   # GPU
            w_exp_cpu = weight_map_cpu.unsqueeze(0).expand(b, -1, -1)   # CPU
        else:
            w_exp     = weight_map_gpu[start:end].to(device)
            w_exp_cpu = weight_map_cpu[start:end]

        if mode == "phase":
            # Circular mean: accumulate weight-scaled cos and sin separately
            cos_acc = place_patches_fourier_shift(
                cos_acc, batch_positions, torch.cos(batch_patches) * w_exp,
                op="add", adjoint_mode=False, pad=pad,
            )
            sin_acc = place_patches_fourier_shift(
                sin_acc, batch_positions, torch.sin(batch_patches) * w_exp,
                op="add", adjoint_mode=False, pad=pad,
            )
        else:
            value_acc = place_patches_fourier_shift(
                value_acc, batch_positions, batch_patches * w_exp,
                op="add", adjoint_mode=False, pad=pad,
            )

        # Weight accumulator: integer placement (no FFT) — accurate enough for denominator
        weight_acc = _place_patches_no_shift(weight_acc, batch_positions.cpu(), w_exp_cpu)

        if (b == batch_size) and (end < n_patches):
            print(f"  Placed {end}/{n_patches} patches...")

    if mode == "phase":
        # atan2 is the correct circular mean — scale-invariant, no divide needed
        stitched = torch.atan2(sin_acc, cos_acc)
    else:
        stitched = value_acc / torch.clamp(weight_acc.to(device), min=1e-8)

    stitched_np = stitched.cpu().numpy()
    weight_np = weight_acc.cpu().numpy()

    print(f"  Stitched shape: {stitched_np.shape},  max weight: {weight_np.max():.2f}")

    return stitched_np, weight_np


def stitch_patches_multi(
    channels: list,
    positions,
    patch_size: int = 256,
    image_shape=None,
    batch_size: int = 1024,
    crop: int = 0,
    pad: int = 4,
    canvas_pad: int = 0,
    device=None,
    patch_weights=None,
) -> list:
    """
    Stitch multiple patch maps in a single loop, sharing canvas setup,
    weight-map preparation, and per-batch weight accumulation.

    Equivalent to calling :func:`stitch_patches` once per channel, but
    significantly faster because:

    * Canvas extents and position offsets are computed **once** for all channels.
    * The probe-weight denominator is accumulated **once** per batch instead of
      once per channel per batch.
    * When ``device`` is a CUDA device all Fourier-shift FFTs run on GPU.

    Parameters
    ----------
    channels : list of dict
        Each dict must contain:

        ``patches`` : (N, H, W) ndarray or Tensor
            Patch values to stitch.
        ``mode`` : ``"amplitude"`` | ``"phase"``
            How overlapping patches are merged.  Defaults to ``"amplitude"``.
        ``label`` : str, optional
            Short name used in progress output.

    positions : (N, 2) array-like
        Centre (y, x) positions in pixels.
    patch_size, image_shape, batch_size, crop, pad, canvas_pad, device, patch_weights :
        Same semantics as :func:`stitch_patches`.

    Returns
    -------
    list of tuple (stitched_np, weights_np)
        One pair per input channel, in the same order.
    """
    if not channels:
        return []

    if device is None:
        device = torch.device("cpu")

    # ── Pre-crop all channels ─────────────────────────────────────────────────
    patches_all = []
    for ch in channels:
        p = ch["patches"]
        if not isinstance(p, torch.Tensor):
            p = torch.from_numpy(np.asarray(p)).float()
        if crop > 0:
            p = p[:, crop:-crop, crop:-crop]
        patches_all.append(p)

    h_patch, w_patch = patches_all[0].shape[-2], patches_all[0].shape[-1]
    half_h = h_patch // 2
    half_w = w_patch // 2
    n_patches = len(patches_all[0])

    if not isinstance(positions, torch.Tensor):
        positions = torch.from_numpy(np.asarray(positions)).float()

    # ── Shared weight map ─────────────────────────────────────────────────────
    if patch_weights is not None:
        if not isinstance(patch_weights, torch.Tensor):
            patch_weights = torch.from_numpy(np.asarray(patch_weights)).float()
        if patch_weights.ndim == 3 and patch_weights.shape[0] == 1:
            patch_weights = patch_weights.squeeze(0)
        if patch_weights.ndim == 2 and crop > 0:
            patch_weights = patch_weights[crop:-crop, crop:-crop]
        elif patch_weights.ndim == 3 and crop > 0:
            patch_weights = patch_weights[:, crop:-crop, crop:-crop]
        pw_max = patch_weights.max()
        if pw_max > 0:
            patch_weights = patch_weights / pw_max
        weight_map_2d = patch_weights.ndim == 2
        weight_map_gpu = patch_weights.to(device) if weight_map_2d else patch_weights
        weight_map_cpu = patch_weights.cpu()
        weight_source = "probe |ψ|²"
    else:
        weight_map_2d = True
        weight_map_gpu = make_hann_window(h_patch, w_patch, device=device)
        weight_map_cpu = weight_map_gpu.cpu()
        weight_source = "Hann window"

    # ── Shared canvas setup ───────────────────────────────────────────────────
    if image_shape is None:
        min_y = positions[:, 0].min() - half_h
        max_y = positions[:, 0].max() + half_h
        min_x = positions[:, 1].min() - half_w
        max_x = positions[:, 1].max() + half_w
        canvas_h = int(np.ceil(max_y - min_y)) + 2 + 2 * canvas_pad
        canvas_w = int(np.ceil(max_x - min_x)) + 2 + 2 * canvas_pad
        image_shape = (canvas_h, canvas_w)
        offset_y = -min_y + 1 + canvas_pad
        offset_x = -min_x + 1 + canvas_pad
        positions = positions.clone()
        positions[:, 0] += offset_y
        positions[:, 1] += offset_x

    labels = [ch.get("label", ch.get("mode", "?")) for ch in channels]
    print(
        f"Stitching multi-channel ({', '.join(labels)}, weight={weight_source}): "
        f"{n_patches} patches ({h_patch}\u00d7{w_patch} after crop={crop}) "
        f"-> {image_shape[0]}\u00d7{image_shape[1]} canvas"
    )
    print(
        f"  Position range: "
        f"Y=[{float(positions[:, 0].min()):.1f}, {float(positions[:, 0].max()):.1f}], "
        f"X=[{float(positions[:, 1].min()):.1f}, {float(positions[:, 1].max()):.1f}]"
    )

    # ── Per-channel accumulators; weight_acc is shared across all channels ────
    weight_acc = torch.zeros(image_shape)  # always CPU
    accs: list = []
    for ch in channels:
        mode = ch.get("mode", "amplitude")
        if mode == "phase":
            accs.append({
                "mode": "phase",
                "cos":  torch.zeros(image_shape, device=device),
                "sin":  torch.zeros(image_shape, device=device),
            })
        else:
            accs.append({
                "mode": "amplitude",
                "val":  torch.zeros(image_shape, device=device),
            })

    # ── Main loop ─────────────────────────────────────────────────────────────
    for start in range(0, n_patches, batch_size):
        end = min(start + batch_size, n_patches)
        b   = end - start
        batch_positions = positions[start:end].to(device)

        if weight_map_2d:
            w_exp_gpu = weight_map_gpu.unsqueeze(0).expand(b, -1, -1)
            w_exp_cpu = weight_map_cpu.unsqueeze(0).expand(b, -1, -1)
        else:
            w_exp_gpu = weight_map_gpu[start:end].to(device)
            w_exp_cpu = weight_map_cpu[start:end]

        # Per-channel Fourier-shift placement — all run on `device`
        for patches_ch, acc in zip(patches_all, accs):
            batch_patches = patches_ch[start:end].to(device)
            if acc["mode"] == "phase":
                acc["cos"] = place_patches_fourier_shift(
                    acc["cos"], batch_positions,
                    torch.cos(batch_patches) * w_exp_gpu,
                    op="add", adjoint_mode=False, pad=pad,
                )
                acc["sin"] = place_patches_fourier_shift(
                    acc["sin"], batch_positions,
                    torch.sin(batch_patches) * w_exp_gpu,
                    op="add", adjoint_mode=False, pad=pad,
                )
            else:
                acc["val"] = place_patches_fourier_shift(
                    acc["val"], batch_positions,
                    batch_patches * w_exp_gpu,
                    op="add", adjoint_mode=False, pad=pad,
                )

        # Weight denominator — accumulated once, shared by all channels
        weight_acc = _place_patches_no_shift(weight_acc, batch_positions.cpu(), w_exp_cpu)

        if (b == batch_size) and (end < n_patches):
            print(f"  Placed {end}/{n_patches} patches...")

    # ── Finalise each channel ─────────────────────────────────────────────────
    weight_np = weight_acc.numpy()
    results = []
    for acc in accs:
        if acc["mode"] == "phase":
            stitched = torch.atan2(acc["sin"], acc["cos"])
        else:
            stitched = acc["val"] / torch.clamp(weight_acc.to(device), min=1e-8)
        results.append((stitched.cpu().numpy(), weight_np))

    print(f"  Stitched shape: {results[0][0].shape},  max weight: {weight_np.max():.2f}")
    return results

def batch_slice(image: Tensor, sy: Tensor, sx: Tensor, patch_size: Tuple[int, int]) -> Tensor:
    """
    Slice patches from an image at given window positions. The patch size is determined
    from the starting and ending coordinates in each direction, and is assumed to be
    the same for all patches.
    
    Parameters
    ----------
    image : Tensor
        A (H, W) tensor of the image.
    sy : Tensor
        A (N,) tensor of integers giving the starting y-coordinates of the patches.
    sx : Tensor
        A (N,) tensor of integers giving the starting x-coordinates of the patches.
    patch_size : tuple of int
        A tuple giving the patch shape in pixels.

    Returns
    -------
    Tensor
        A tensor of shape (N, h, w) containing the extracted patches.
    """
    h, w = image.shape[-2:]
    if (
        sy.min() < 0 
        or sy.max() + patch_size[0] > image.shape[-2] 
        or sx.min() < 0 
        or sx.max() + patch_size[1] > image.shape[-1]
    ):
        raise ValueError(
            f"Patch indices are out of bounds.\n"
            f"Image shape: {image.shape}\n"
            f"Patch size: {patch_size}\n"
            f"sy range: [{sy.min().item()}, {sy.max().item()}]\n"
            f"sx range: [{sx.min().item()}, {sx.max().item()}]\n"
            f"sy max + patch height: {sy.max().item() + patch_size[0]} (limit: {image.shape[-2]})\n"
            f"sx max + patch width: {sx.max().item() + patch_size[1]} (limit: {image.shape[-1]})"
        )
    
    x = torch.arange(patch_size[1], device=sx.device)[None, :]
    y = torch.arange(patch_size[0], device=sy.device)[None, :]
    x = x.expand(len(sx), x.shape[1])
    y = y.expand(len(sy), y.shape[1])
    x = x + sx[:, None]
    y = y + sy[:, None]
    inds = (y * w).unsqueeze(-1) + x.unsqueeze(1)
    patches = image.view(-1)[inds.view(-1)]
    patches = patches.reshape(len(sy), patch_size[0], patch_size[1])
    return patches

def batch_put(
    image: Tensor, 
    patches: Tensor, 
    sy: Tensor, 
    sx: Tensor, 
    op: Literal["add", "set"] = "add"
) -> Tensor:
    """
    Slice patches from an image at given window positions. The patch size is assumed 
    to be the same for all patches.
    
    Parameters
    ----------
    image : Tensor
        A (H, W) tensor of the buffer to place the patches into.
    patches : Tensor
        A (N, h, w) tensor of the patches.
    sy : Tensor
        A (N,) tensor of integers giving the starting y-coordinates of the patches.
    sx : Tensor
        A (N,) tensor of integers giving the starting x-coordinates of the patches.
    op : Literal["add", "set"]
        The operation to perform. "add" adds the patches to the image, 
        "set" sets the patches to the image replacing the existing values.
    
    Returns
    -------
    Tensor
        A tensor of shape (H, W) containing the image with patches added or set.
    """
    h, w = image.shape[-2:]
    if (
        sy.min() < 0 
        or sy.max() + patches.shape[-2] > image.shape[-2] 
        or sx.min() < 0 
        or sx.max() + patches.shape[-1] > image.shape[-1]
    ):
        raise ValueError("Patch indices are out of bounds.")
    
    patch_size = patches.shape[-2:]
    x = torch.arange(patch_size[1], device=sx.device)[None, :]
    y = torch.arange(patch_size[0], device=sy.device)[None, :]
    x = x.expand(len(sx), x.shape[1])
    y = y.expand(len(sy), y.shape[1])
    x = x + sx[:, None]
    y = y + sy[:, None]
    inds = (y * w).unsqueeze(-1) + x.unsqueeze(1)
    image = image.reshape(-1)
    
    try:
        patches_flattened = patches.view(-1)
    except RuntimeError:
        patches_flattened = patches.reshape(-1)
        
    # scatter_add_ and scatter_ are non-deterministic but faster. 
    # This part is modified from pty-chi to only allow the non-deterministic option
    if op == "add":
        image.scatter_add_(0, inds.view(-1), patches_flattened)
    else:
        image.scatter_(0, inds.view(-1), patches_flattened)
    
    return image.reshape(h, w)

def fourier_shift(images: Tensor, shifts: Tensor, strictly_preserve_zeros: bool = False) -> Tensor:
    """
    Apply Fourier shift to a batch of images. From Ming Du's pty-chi:
    https://github.com/AdvancedPhotonSource/pty-chi 

    Parameters
    ----------
    images : Tensor
        A [N, H, W] tensor of images.
    shifts : Tensor
        A [N, 2] tensor of shifts in pixels.
    strictly_preserve_zeros : bool
        If True, mask of strictly zero pixels will be generated and shifted
        by the same amount. Pixels that have a non-zero value in the shifted
        mask will be set to zero in the shifted image. This preserves the zero
        pixels in the original image, preventing FFT from introducing small
        non-zero values due to machine precision.

    Returns
    -------
    Tensor
        Shifted images.
    """
    if strictly_preserve_zeros:
        zero_mask = images == 0
        zero_mask = zero_mask.float()
        zero_mask_shifted = fourier_shift(zero_mask, shifts, strictly_preserve_zeros=False)
    # Stay in complex64 throughout — complex128 intermediates offered no practical benefit
    # for sub-pixel shifts (<1 px) and doubled memory use + halved FFT throughput.
    ft_images = torch.fft.fft2(images, norm=None)
    freq_y, freq_x = torch.meshgrid(
        torch.fft.fftfreq(images.shape[-2], device=images.device),
        torch.fft.fftfreq(images.shape[-1], device=images.device),
        indexing="ij",
    )
    freq_x = freq_x.repeat(images.shape[0], 1, 1)
    freq_y = freq_y.repeat(images.shape[0], 1, 1)
    mult = torch.exp(
        1j
        * -2
        * torch.pi
        * (freq_x * shifts[:, 1].view(-1, 1, 1) + freq_y * shifts[:, 0].view(-1, 1, 1))
    )
    ft_images = ft_images * mult.to(ft_images.dtype)
    shifted_images = torch.fft.ifft2(ft_images, norm=None)
    if not images.dtype.is_complex:
        shifted_images = shifted_images.real
    if strictly_preserve_zeros:
        shifted_images[zero_mask_shifted > 0] = 0
    return shifted_images

def extract_patches_fourier_shift(
    image: Tensor, positions: Tensor, shape: Tuple[int, int], pad: Optional[int] = 1
) -> Tensor:
    """
    Extract patches from 2D object. If a patch's footprint goes outside the image,
    the image is padded with zeros to account for the missing pixels. From Ming Du's pty-chi:
    https://github.com/AdvancedPhotonSource/pty-chi 

    Parameters
    ----------
    image : Tensor
        The whole image.
    positions : Tensor
        A tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    shape : tuple of int
        A tuple giving the patch shape in pixels.
    pad : Optional[int]
        If given, patches with larger size than the intended size by this amount are cropped
        out from the patches before shifting.

    Returns
    -------
    Tensor
        A tensor of shape (N, H, W) containing the extracted patches.
    """
    # Floating point ranges over which interpolations should be done
    sys_float = positions[:, 0] - (shape[0] - 1.0) / 2.0
    sxs_float = positions[:, 1] - (shape[1] - 1.0) / 2.0

    # Crop one more pixel each side for Fourier shift
    sys = sys_float.floor().int() - pad
    eys = sys + shape[0] + 2 * pad
    sxs = sxs_float.floor().int() - pad
    exs = sxs + shape[1] + 2 * pad

    fractional_shifts = torch.stack([sys_float - sys - pad, sxs_float - sxs - pad], -1)

    pad_lengths = [
        max(-sxs.min(), 0),
        max(exs.max() - image.shape[1], 0),
        max(-sys.min(), 0),
        max(eys.max() - image.shape[0], 0),
    ]
    image = torch.nn.functional.pad(image, pad_lengths)
    sys = sys + pad_lengths[2]
    eys = eys + pad_lengths[2]
    sxs = sxs + pad_lengths[0]
    exs = exs + pad_lengths[0]

    patches = batch_slice(image, sys, sxs, patch_size=[shape[i] + 2 * pad for i in range(2)])

    # Apply Fourier shift to account for fractional shifts
    if not torch.allclose(fractional_shifts, torch.zeros_like(fractional_shifts), atol=1e-7):
        patches = fourier_shift(patches, -fractional_shifts)
    patches = patches[:, pad : patches.shape[-2] - pad, pad : patches.shape[-1] - pad]
    return patches

def place_patches_fourier_shift( 
    image: Tensor, 
    positions: Tensor, 
    patches: Tensor, 
    op: Literal["add", "set"] = "add", 
    adjoint_mode: bool = False,
    pad: Optional[int] = 1 
) -> Tensor:
    """
    Place patches into a 2D object. If a patch's footprint goes outside the image,
    the image is padded with zeros to account for the missing pixels. This version removes timer and plotting

    Parameters
    ----------
    image : Tensor
        The whole image.
    positions : Tensor
        A tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    patches : Tensor
        A (N, H, W) or (H, W) tensor of image patches.
    op : Literal["add", "set"]
        The operation to perform. "add" adds the patches to the image, 
        "set" sets the patches to the image replacing the existing values.
    adjoint_mode : bool
        If True, this function performs the exact adjoint operation of `extract_patches_fourier_shift`.
        This means it will run the adjoint operation of every step of the extraction 
        function in reverse order: it first zero-pads the patches, shifts them back, 
        and puts them back into the image. Turn it on if this function is used in
        backpropagating the gradient. Note that due to the zero-padding, ripple
        artifacts may appear around the borders of each patch, so it is not suitable
        for placing patches that are not gradients. In that case, set this to False,
        and it will skip the zero-padding and crop the patches before placing them
        to remove Fourier shift wrap-arounds.
    pad : Optional[int]
        If given, patches are padded (or cropped) by this amount before shifting. 
        The actual operations depend on `adjoint_mode`: when `adjoint_mode` is True, 
        the patches are padded with zeros; otherwise, pathces are cropped by this amount
        after shifting to remove the wrap-around portions.

    Returns
    -------
    Tensor
        A tensor with the same shape as the object with patches added onto it.
    """
    # If the input is a single patch, add the third dimension
    # and expand it to the correct number of patches
    if len(patches.shape) == 2:
        patches = patches[None].expand(len(positions), -1, -1)

    shape = patches.shape[-2:]
    
    if adjoint_mode:
        patch_padding = pad
        patches = torch.nn.functional.pad(patches, [patch_padding] * 4)
    else:
        patch_padding = -pad

    sys_float = positions[:, 0] - (shape[0] - 1.0) / 2.0
    sxs_float = positions[:, 1] - (shape[1] - 1.0) / 2.0

    # Crop one more pixel each side for Fourier shift
    sys = sys_float.floor().int() - patch_padding
    eys = sys + shape[0] + 2 * patch_padding
    sxs = sxs_float.floor().int() - patch_padding
    exs = sxs + shape[1] + 2 * patch_padding

    fractional_shifts = torch.stack([sys_float - sys - patch_padding, sxs_float - sxs - patch_padding], -1)

    pad_lengths = [
        max(-sxs.min(), 0),
        max(exs.max() - image.shape[1], 0),
        max(-sys.min(), 0),
        max(eys.max() - image.shape[0], 0),
    ]
    image = torch.nn.functional.pad(image, pad_lengths)
    sys = sys + pad_lengths[2]
    eys = eys + pad_lengths[2]
    sxs = sxs + pad_lengths[0]
    exs = exs + pad_lengths[0]

    if not torch.allclose(fractional_shifts, torch.zeros_like(fractional_shifts), atol=1e-7):
        patches = fourier_shift(patches, fractional_shifts)
        
    if not adjoint_mode:
        patches = patches[
            :, 
            abs(patch_padding) : patches.shape[-2] - abs(patch_padding), 
            abs(patch_padding) : patches.shape[-1] - abs(patch_padding)
        ]
    
    image = batch_put(image, patches, sys, sxs, op=op)

    # Undo padding
    image = image[
        pad_lengths[2] : image.shape[0] - pad_lengths[3],
        pad_lengths[0] : image.shape[1] - pad_lengths[1],
    ]
    return image

def bilinear_shift(images: Tensor, shifts: Tensor) -> Tensor:
    """
    Apply bilinear shift to a batch of images.

    Parameters
    ----------
    images : Tensor
        A [N, H, W] tensor of images.
    shifts : Tensor
        A [N, 2] tensor of shifts in pixels.

    Returns
    -------
    Tensor
        Shifted images.
    """
    if torch.allclose(shifts, torch.zeros_like(shifts)):
        return images
    y, x = torch.meshgrid(torch.arange(images.shape[-2]), torch.arange(images.shape[-1]), indexing="ij")
    y = y.to(images.device)
    x = x.to(images.device)
    y = y.repeat(images.shape[0], 1, 1)
    x = x.repeat(images.shape[0], 1, 1)

    # We want to sample the image at these positions.
    y = y - shifts[:, 0].view(-1, 1, 1)
    x = x - shifts[:, 1].view(-1, 1, 1)

    y_f = y.floor().int()
    y_c = (y + 1).floor().int()
    x_f = x.floor().int()
    x_c = (x + 1).floor().int()

    w_00 = (y_c - y) * (x_c - x)
    w_01 = (y_c - y) * (x - x_f)
    w_10 = (y - y_f) * (x_c - x)
    w_11 = (y - y_f) * (x - x_f)

    y_f = y_f.clip(0, images.shape[-2] - 1).view(-1)
    y_c = y_c.clip(0, images.shape[-2] - 1).view(-1)
    x_f = x_f.clip(0, images.shape[-1] - 1).view(-1)
    x_c = x_c.clip(0, images.shape[-1] - 1).view(-1)

    orig_shape = images.shape
    batch_inds = torch.arange(images.shape[0]).int().repeat_interleave(images.shape[-1] * images.shape[-2])
    images = \
        w_00.view(-1) * images[batch_inds, y_f, x_f] + \
        w_01.view(-1) * images[batch_inds, y_f, x_c] + \
        w_10.view(-1) * images[batch_inds, y_c, x_f] + \
        w_11.view(-1) * images[batch_inds, y_c, x_c]
    images = images.reshape(orig_shape)
    return images

def place_patches_bilinear_shift( 
    image: Tensor, 
    positions: Tensor, 
    patches: Tensor, 
    op: Literal["add", "set"] = "add", 
    adjoint_mode: bool = False,
    pad: Optional[int] = 1 
) -> Tensor:
    """
    Place patches into a 2D object. If a patch's footprint goes outside the image,
    the image is padded with zeros to account for the missing pixels. This version removes timer and plotting

    Parameters
    ----------
    image : Tensor
        The whole image.
    positions : Tensor
        A tensor of shape (N, 2) giving the center positions of the patches in pixels.
        The origin of the given positions are assumed to be the TOP LEFT corner of the image.
    patches : Tensor
        A (N, H, W) or (H, W) tensor of image patches.
    op : Literal["add", "set"]
        The operation to perform. "add" adds the patches to the image, 
        "set" sets the patches to the image replacing the existing values.
    adjoint_mode : bool
        Legacy from place_patches_fourier_shift. Just always set it to False here
    pad : Optional[int]
        If given, patches are padded (or cropped) by this amount before shifting. 
        The actual operations depend on `adjoint_mode`: when `adjoint_mode` is True, 
        the patches are padded with zeros; otherwise, pathces are cropped by this amount
        after shifting to remove the wrap-around portions.

    Returns
    -------
    Tensor
        A tensor with the same shape as the object with patches added onto it.
    """
    # If the input is a single patch, add the third dimension
    # and expand it to the correct number of patches
    if len(patches.shape) == 2:
        patches = patches[None].expand(len(positions), -1, -1)

    shape = patches.shape[-2:]
    
    if adjoint_mode:
        patch_padding = pad
        patches = torch.nn.functional.pad(patches, [patch_padding] * 4)
    else:
        patch_padding = -pad

    sys_float = positions[:, 0] - (shape[0] - 1.0) / 2.0
    sxs_float = positions[:, 1] - (shape[1] - 1.0) / 2.0

    # Crop one more pixel each side for Fourier shift
    sys = sys_float.floor().int() - patch_padding
    eys = sys + shape[0] + 2 * patch_padding
    sxs = sxs_float.floor().int() - patch_padding
    exs = sxs + shape[1] + 2 * patch_padding

    fractional_shifts = torch.stack([sys_float - sys - patch_padding, sxs_float - sxs - patch_padding], -1)

    pad_lengths = [
        max(-sxs.min(), 0),
        max(exs.max() - image.shape[1], 0),
        max(-sys.min(), 0),
        max(eys.max() - image.shape[0], 0),
    ]
    image = torch.nn.functional.pad(image, pad_lengths)
    sys = sys + pad_lengths[2]
    eys = eys + pad_lengths[2]
    sxs = sxs + pad_lengths[0]
    exs = exs + pad_lengths[0]

    if not torch.allclose(fractional_shifts, torch.zeros_like(fractional_shifts), atol=1e-7):
        patches = bilinear_shift(patches, fractional_shifts)
        
    if not adjoint_mode:
        patches = patches[
            :, 
            abs(patch_padding) : patches.shape[-2] - abs(patch_padding), 
            abs(patch_padding) : patches.shape[-1] - abs(patch_padding)
        ]
    
    image = batch_put(image, patches, sys, sxs, op=op)

    # Undo padding
    image = image[
        pad_lengths[2] : image.shape[0] - pad_lengths[3],
        pad_lengths[0] : image.shape[1] - pad_lengths[1],
    ]
    return image

def create_positions(object_shape, probe_lateral_shape, target_overlap=0.8, fwhm=98):
    """Create probe positions in pixels. Adapted from Ming Du's ptycho_simulation_factory:
    https://github.com/mdw771/ptycho_simulation_factory

    This version removes the choice to pre-define number of positions or spacing. 
    Instead, the spacing is calculated by overlap and probe size.
    
    Parameters
    ----------
    object_shape : tuple of int
        Lateral shape of the object.
    probe_lateral_shape : tuple of int
        Lateral shape of the probe. This is used to determine the safety margin,
        so that the probe does not reach outside the object.
    target_overlap : float
        Overlap ratio
    fwhm : int
        Full width at half maximum of the probe in pixels (approximate is fine here)
    """
    spacing = (1 - target_overlap) * fwhm # Spacing is now enforced to be the same in y and x

    margin = [probe_lateral_shape[i] // 2 for i in range(len(probe_lateral_shape))]
    y = np.arange(margin[0], object_shape[0] - margin[0] - 1, spacing)
    x = np.arange(margin[1], object_shape[1] - margin[1] - 1, spacing)

    y, x = np.meshgrid(y, x)
    positions = np.stack([y.reshape(-1), x.reshape(-1)], axis=1)
    positions = positions - positions.mean(0) # Center is (0, 0)
    return torch.from_numpy(positions)