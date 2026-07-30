# All image processing utilities adapted from Ming Du's Pty-chi
# https://github.com/AdvancedPhotonSource/pty-chi
# Copyright © 2025 UChicago Argonne, LLC All right reserved

# Position creation for raster scans adapted from Ming Du's ptycho_simulation_factory
# https://github.com/mdw771/ptycho_simulation_factory

from typing import Literal

import numpy as np
import torch
from torch import Tensor


def batch_slice(image: Tensor, sy: Tensor, sx: Tensor, patch_size: tuple[int, int]) -> Tensor:
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
    w = image.shape[-1]
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
    # This version intended for torch.complex64 images only, though it inherits type from image
    ft_images = torch.fft.fft2(images.type(torch.complex128), norm=None).type(torch.complex64)
    freq_y, freq_x = torch.meshgrid(
        torch.fft.fftfreq(images.shape[-2]), torch.fft.fftfreq(images.shape[-1]), indexing="ij"
    )
    freq_x = freq_x.to(ft_images.device)
    freq_y = freq_y.to(ft_images.device)
    freq_x = freq_x.repeat(images.shape[0], 1, 1)
    freq_y = freq_y.repeat(images.shape[0], 1, 1)
    mult = torch.exp(
        1j
        * -2
        * torch.pi
        * (freq_x * shifts[:, 1].view(-1, 1, 1) + freq_y * shifts[:, 0].view(-1, 1, 1))
    )
    ft_images = ft_images * mult
    # Complex images (pty-chi original version supports real datatypes with higher precision)
    shifted_images = torch.fft.ifft2(ft_images.type(torch.complex128), norm=None).type(torch.complex64) 
    if not images.dtype.is_complex:
        shifted_images = shifted_images.real
    if strictly_preserve_zeros:
        shifted_images[zero_mask_shifted > 0] = 0
    return shifted_images

def extract_patches_fourier_shift(
    image: Tensor, positions: Tensor, shape: tuple[int, int], pad: int | None = 1
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
    pad: int | None = 1 
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
    pad: int | None = 1 
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