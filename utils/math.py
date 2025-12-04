"""
Mathematical utility functions for ptychography reconstruction.

This module contains coordinate transforms and mathematical operations.
"""

import torch
import torch.nn.functional as F


def create_logpolar_grid(height, width, device='cpu'):
    """
    Create a log-polar sampling grid for torch.nn.functional.grid_sample.
    Follows numpy 'ij' indexing convention to match scipy behavior.

    Args:
        height, width: Output dimensions
        device: torch device

    Returns:
        grid: Tensor of shape (1, height, width, 2) with normalized coordinates
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
