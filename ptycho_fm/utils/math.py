"""
Mathematical utility functions.

This module contains coordinate transforms and mathematical operations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


def create_logpolar_grid(height, width, device='cpu'):
    """
    Create a log-polar sampling grid for torch.nn.functional.grid_sample.
    Follows numpy 'ij' indexing convention to match scipy behavior.

    Args:
        height, width: Output dimensions
        device: torch device

    Returns:
        grid: Tensor of shape (1, height, width, 2) with normalized coordinates

    Example:
        Create a grid on the same device as the images and apply the transform:

        >>> images = torch.rand(2, 1, 256, 256)
        >>> height, width = images.shape[-2:]
        >>> grid = create_logpolar_grid(height, width, device=images.device)
        >>> transformed = apply_logpolar_transform(images, grid, mode='bicubic')
        >>> transformed.shape
        torch.Size([2, 1, 256, 256])

        Reuse the grid for subsequent batches with the same spatial dimensions
        and device.
    """
    # Match scipy's np.mgrid convention with 'ij' indexing
    # np.mgrid[-h//2:h//2, -w//2:w//2] creates grids where:
    # - first dimension (y) varies along rows (axis 0)
    # - second dimension (x) varies along columns (axis 1)
    half_h, half_w = height // 2, width // 2
    y, x = torch.meshgrid(
        torch.linspace(-half_h, half_h, height, device=device),
        torch.linspace(-half_w, half_w, width, device=device),
        indexing='ij'
    )

    # Compute log-polar coordinates
    rho = torch.log(torch.sqrt(x**2 + y**2) + 1e-10)
    _theta = torch.atan2(y, x)

    # Define output log-polar grid (evenly sampled in log-polar space)
    valid_rho = rho[torch.isfinite(rho)]
    rho_out = torch.linspace(valid_rho.min(), rho.max(), height, device=device)
    theta_out = torch.linspace(-torch.pi, torch.pi, width, device=device)

    # np.meshgrid default is 'xy', but we need to match scipy exactly
    # In the scipy code: rho_grid, theta_grid = np.meshgrid(rho_out, theta_out)
    # This uses 'xy' indexing, so theta varies along axis 0, rho along axis 1
    theta_grid, rho_grid = torch.meshgrid(theta_out, rho_out, indexing='ij')

    # Convert log-polar grid back to Cartesian coordinates
    x_sample = torch.exp(rho_grid) * torch.cos(theta_grid)
    y_sample = torch.exp(rho_grid) * torch.sin(theta_grid)

    # Normalize to [-1, 1] for grid_sample
    # grid_sample expects (x, y) in the last dimension
    x_norm = x_sample / half_w
    y_norm = y_sample / half_h

    # Stack to (1, H, W, 2) - last dim is (x, y)
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

    return grid


def apply_logpolar_transform(images, grid, mode='bilinear'):
    """
    Apply log-polar transform to batched images.

    Args:
        images: Tensor of shape (B, C, H, W)
        grid: Pre-computed grid from create_logpolar_grid
        mode: Interpolation mode

    Returns:
        Transformed images of shape (B, C, H, W)

    Example:
        Transform a batch using a single grid, which is expanded across the batch:

        >>> images = torch.rand(2, 1, 256, 256)
        >>> height, width = images.shape[-2:]
        >>> grid = create_logpolar_grid(height, width, device=images.device)
        >>> transformed = apply_logpolar_transform(images, grid, mode='bicubic')
        >>> transformed.shape
        torch.Size([2, 1, 256, 256])
    """
    batch_size = images.shape[0]

    # Expand grid to batch size
    grid_expanded = grid.expand(batch_size, -1, -1, -1)

    # Apply transformation
    transformed = F.grid_sample(
        images,
        grid_expanded,
        mode=mode,
        padding_mode='zeros',
        align_corners=False
    )

    return transformed


# Copyright © 2025 UChicago Argonne, LLC All right reserved
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
    spacing = (1 - target_overlap) * fwhm  # Spacing is now enforced to be the same in y and x

    margin = [probe_lateral_shape[i] // 2 for i in range(len(probe_lateral_shape))]
    y = np.arange(margin[0], object_shape[0] - margin[0] - 1, spacing)
    x = np.arange(margin[1], object_shape[1] - margin[1] - 1, spacing)

    y, x = np.meshgrid(y, x)
    positions = np.stack([y.reshape(-1), x.reshape(-1)], axis=1)
    positions = positions - positions.mean(0)  # Center is (0, 0)
    return torch.from_numpy(positions)


def estimate_scan_overlap(positions, probe_fwhm, k=1):
    """Estimate probe overlap from arbitrary two-dimensional scan positions.

    For every scan position, the local step size is the mean distance to its
    ``k`` nearest spatial neighbors. The corresponding linear overlap is
    ``1 - local_spacing / probe_fwhm``. Positions and ``probe_fwhm`` must use
    the same physical units.

    Negative overlap values are retained because they identify positions whose
    local spacing is larger than the probe FWHM. Exact duplicate positions are
    treated as having 100 percent overlap.

    Args:
        positions: Array-like object with shape ``(N, 2)`` containing scan
            coordinates.
        probe_fwhm: Positive intensity FWHM of the probe.
        k: Positive number of nearest neighbors averaged for each position.
            ``k=1`` gives the conventional nearest-neighbor overlap estimate.

    Returns:
        Dictionary containing mean and median spacing, mean and median overlap,
        the 10th and 90th overlap percentiles, the fraction of positions with
        non-positive overlap, and the per-position spacing and overlap arrays.

    Raises:
        ValueError: If the inputs have invalid shapes or values, or if there
            are not enough scan positions for the requested ``k``.
    """
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if positions.shape[0] == 0:
        raise ValueError("positions must contain at least one scan position")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite values")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if positions.shape[0] <= k:
        raise ValueError("the number of scan positions must be greater than k")
    if not np.isscalar(probe_fwhm):
        raise ValueError("probe_fwhm must be a positive finite scalar")

    probe_fwhm = float(probe_fwhm)
    if not np.isfinite(probe_fwhm) or probe_fwhm <= 0:
        raise ValueError("probe_fwhm must be a positive finite scalar")

    tree = cKDTree(positions)
    distances, _ = tree.query(positions, k=k + 1)
    local_spacing = distances[:, 1:].mean(axis=1)
    local_overlap = 1.0 - local_spacing / probe_fwhm

    return {
        "mean_spacing": float(np.mean(local_spacing)),
        "median_spacing": float(np.median(local_spacing)),
        "mean_overlap": float(np.mean(local_overlap)),
        "median_overlap": float(np.median(local_overlap)),
        "overlap_p10": float(np.percentile(local_overlap, 10)),
        "overlap_p90": float(np.percentile(local_overlap, 90)),
        "fraction_with_no_fwhm_overlap": float(np.mean(local_overlap <= 0)),
        "local_spacing": local_spacing,
        "local_overlap": local_overlap,
    }
