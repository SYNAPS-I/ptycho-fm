"""
Safe inference script for experimental data.
This script ONLY reads the model checkpoint - it will NOT modify or overwrite anything.
Results are saved to a separate 'results/' directory.
"""
import os
import time
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle

from data import PtychographyDataset
from model.model import PtychoViT
from torch.utils.data import DataLoader
# [MODIFIED] import stitch_patches from shared utils instead of defining it locally
from utils.ptychi_utils import place_patches_fourier_shift, stitch_patches

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR DATA
# =============================================================================
# Path to your experimental data (HDF5 file with diffraction patterns)
# EXPERIMENTAL_DATA_PATH = r'/nsls2/users/cchung/SYNAPS-I/ptycho-vit/test_data/slice0_dp.hdf5'  # e.g., 'C:/path/to/your_data_dp.hdf5'
EXPERIMENTAL_DATA_PATH = r'/nsls2/data/hxn/legacy/home/home/SYNAPS/hgoel1/converted_data_old/363910/363910_dp.hdf5'  # Real experimental data when using Orion cluster
EXPERIMENTAL_DATA_PATH = r'/nsls2/data/hxn/legacy/home/home/SYNAPS/chengchu/scan807_test_data/scan807_dp.hdf5'  # Test from APS
# EXPERIMENTAL_DATA_PATH = r'/nsls2/users/cchung/SYNAPS-I/ptycho-vit/test_data/363910/363910_dp.hdf5'   # My folder
# EXPERIMENTAL_DATA_PATH = r'/nsls2/data/hxn/legacy/home/home/SYNAPS/chengchu/converted_dp_para/387134/387134_dp.hdf5'   # My folder
EXPERIMENTAL_DATA_PATH = '/nsls2/data/hxn/legacy/home/home/SYNAPS/chengchu/converted_dp_para/n15091304_742/n15091304_742_dp.hdf5'
EXPERIMENTAL_DATA_PATH = '/nsls2/data/hxn/legacy/home/home/SYNAPS/chengchu/converted_dp_para/272452/272452_dp.hdf5'
EXPERIMENTAL_DATA_PATH = '/nsls2/data/hxn/legacy/home/home/SYNAPS/chengchu/converted_dp_para/272458/272458_dp.hdf5'


print('='*30)
print(f"Predict scan {Path(EXPERIMENTAL_DATA_PATH).stem.removesuffix('_dp')}...")  # → 381477
print('-'*30)

SCAN_ID = '272458'

# Output directory for results (will be created if it doesn't exist)
RESULTS_DIR = '/nsls2/users/cchung/SYNAPS-I/ptycho-vit/results'
RESULTS_DIR = f'/nsls2/data/hxn/legacy/home/home/SYNAPS/chengchu/predicted_results/scan_{SCAN_ID}'

# Batch size (reduce if you run out of GPU memory)
BATCH_SIZE = 64

# Visualization sample index (which patch to visualize)
SLICE_INDEX = 10

# =============================================================================
# VISUALIZATION OPTIONS
# =============================================================================

CH, CW = 2500, 2500   # Cropped heigh and width

# Amplitude contrast mode: 'percentile', 'minmax', or 'manual'
AMP_CONTRAST = 'percentile'
# Phase contrast mode: 'full', 'percentile', 'minmax', or 'manual'
PH_CONTRAST = 'percentile'

# For 'percentile' mode: (low_percentile, high_percentile)
# Lower values = more contrast. Try (1, 99), (2, 98), (5, 95), or (0.5, 99.5)
AMP_PERCENTILE = (1, 99)
PH_PERCENTILE = (1, 99)

# For 'manual' mode: set explicit vmin and vmax values
AMP_VMIN = 0.93  # e.g., 0.8
AMP_VMAX = 0.96  # e.g., 1.2

PH_VMIN = -np.pi*0.93  # e.g., -np.pi
PH_VMAX = np.pi*0.96  # e.g., np.pi

# =============================================================================
# FIXED PATHS - DO NOT MODIFY (these are your trained model files)
# =============================================================================
CHECKPOINT_DIR = Path('/nsls2/users/cchung/SYNAPS-I/ptycho-vit/Model Checkpoints/run145')
CHECKPOINT_DIR = Path('/nsls2/data/hxn/legacy/home/home/SYNAPS/chengchu/20260323-ptycho-vit/Finetuning_results')  # My folder
# CHECKPOINT_DIR = Path('/nsls2/data/hxn/legacy/home/home/SYNAPS/chengchu/20260323-ptycho-vit/Himanshu_model_results')  # My folder
CHECKPOINT_DIR = Path('/nsls2/data/hxn/legacy/home/home/SYNAPS/chengchu/20260323-ptycho-vit/Finetuning_results/run041701_hg_alpha1')
CONFIG_PATH = CHECKPOINT_DIR / 'config.yaml'
# CONFIG_PATH = r"/nsls2/data/hxn/legacy/home/home/SYNAPS/chengchu/20260323-ptycho-vit/Finetuning_results/config.yaml"
print(f"Config path: {CONFIG_PATH}")
MODEL_PATH = CHECKPOINT_DIR / 'best_model.pth'


# [MODIFIED] stitch_patches moved to utils/ptychi_utils.py and imported above


def visualize_stitching(positions, counts, patch_size=1024, save_path=None):
    """
    Visualize stitching geometry and overlap heatmap.

    Left panel: Scan positions as dots with sample patch rectangles (corners + center)
    Right panel: Overlap heatmap from the counts array with colorbar

    Args:
        positions: Array of shape (N, 2) containing (y, x) center positions in pixels
        counts: 2D array from stitch_patches showing overlap count per pixel
        patch_size: Size of each patch
        save_path: Path to save figure (optional)
    """
    half_size = patch_size // 2

    # Calculate canvas bounds (same logic as stitch_patches)
    min_y = positions[:, 0].min() - half_size
    min_x = positions[:, 1].min() - half_size
    offset_y = -min_y + 1
    offset_x = -min_x + 1

    # Transform positions to canvas coordinates
    canvas_positions = positions.copy()
    canvas_positions[:, 0] = positions[:, 0] + offset_y
    canvas_positions[:, 1] = positions[:, 1] + offset_x

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Scan {SCAN_ID}', fontsize=16, y=1)

    # =========================================================================
    # LEFT: Geometry diagram — scan positions + sample patch rectangles
    # =========================================================================
    ax1 = axes[0]
    ax1.set_title('Stitch Geometry', fontsize=12, fontweight='bold')

    # Plot all positions as small dots
    ax1.scatter(canvas_positions[:, 1], canvas_positions[:, 0],
                s=1, c='blue', alpha=0.3, label='Probe centers')

    # Draw sample patch rectangles: pick by actual spatial position
    n = len(positions)
    cy_all = canvas_positions[:, 0]
    cx_all = canvas_positions[:, 1]
    sample_indices = [
        np.argmin(cy_all + cx_all),           # Top-left (min y+x)
        np.argmax(cy_all + cx_all),           # Bottom-right (max y+x)
        np.argmin(np.abs(cy_all - cy_all.mean()) + np.abs(cx_all - cx_all.mean())),  # Center
        np.argmin(cy_all - cx_all),           # Top-right (min y-x)
        np.argmax(cy_all - cx_all),           # Bottom-left (max y-x)
    ]
    sample_labels = ['Top-left', 'Bottom-right', 'Center', 'Top-right', 'Bottom-left']
    patch_colors = ['red', 'green', 'orange', 'purple', 'brown']

    for idx, color, label in zip(sample_indices, patch_colors, sample_labels):
        cy = canvas_positions[idx, 0]
        cx = canvas_positions[idx, 1]
        rect = Rectangle((cx - half_size, cy - half_size), patch_size, patch_size,
                          linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.15,
                          label=f'{label} patch')
        ax1.add_patch(rect)
        ax1.scatter(cx, cy, s=50, c=color, marker='+', linewidths=2, label=f'{label} center')

    ax1.set_xlabel('X (canvas pixels)')
    ax1.set_ylabel('Y (canvas pixels)')
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # RIGHT: Overlap heatmap from counts array
    # =========================================================================
    ax2 = axes[1]
    ax2.set_title('Overlap Heatmap (patches per pixel)', fontsize=12, fontweight='bold')

    im = ax2.imshow(counts, cmap='hot', interpolation='nearest')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='Patch count')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Stitching visualization saved to: {save_path}")

    plt.show()

def load_model(config_path, model_path, device):
    """Load the pre-trained model (read-only, no modifications)."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = PtychoViT(config=config['model'])

    # Load weights (weights_only=True for safety)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()  # Set to evaluation mode

    print(f"Model loaded from: {model_path}")
    print(f"  - Image size: {config['model']['encoder']['img_size']}x{config['model']['encoder']['img_size']}")
    print(f"  - Encoder: {config['model']['encoder_type']} ViT")
    print(f"  - Embed dim: {config['model']['encoder']['embed_dim']}")

    return model, config


def run_inference(model, data_path, config, device, results_dir, batch_size=64):
    """Run inference on experimental data and save results."""

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # [MODIFIED] prefer test_normalization over normalization_dict_path for inference —
    # normalization_dict_path is keyed to the training scan and won't contain the test
    # scan name, causing a fallback to default_normalization (100000.0).
    norm_path = config['data'].get('test_normalization') or config['data'].get('normalization_dict_path')
    print(f"Normalization dict path: {norm_path}")

    # Create dataset
    # NOTE: apply_noise=False for experimental data (noise already present)
    dataset = PtychographyDataset(
        file_path=data_path,
        scale=config['data']['scale'],
        normalization_dict_path=norm_path,
        apply_noise=False,  # IMPORTANT: False for experimental data
        cache_object=True,
        max_probe_modes=config['data'].get('max_probe_modes', 8),
        # [MODIFIED] Fix #9: forward the transpose flag from config so patches are
        # extracted in the correct frame for HXN data.
        transpose_object_patches=config['data'].get('transpose_object_patches', False),
    )

    print(f"\nDataset loaded: {len(dataset)} diffraction patterns")

    # Create dataloader (num_workers=0 for inference stability)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Storage for results
    all_pred_amp = []
    all_pred_ph = []
    all_pred_diff = []
    all_gt_diff = []
    all_positions = []

    img_size = config['model']['encoder']['img_size']

    print('\n'+'='*30)
    print(f"Running inference for scan {SCAN_ID}...")
    print('-'*30)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale = batch

            # Move to device and get target size from config
            target_size = config['model']['encoder']['img_size']
            input_diff = diff_amp.to(device)

            # Apply fftshift (move zero-frequency to center) then center crop
            h, w = input_diff.shape[-2], input_diff.shape[-1]
            if h > target_size or w > target_size:
                input_diff = torch.fft.fftshift(input_diff, dim=(-2, -1))
                start_h = (h - target_size) // 2
                start_w = (w - target_size) // 2
                input_diff = input_diff[..., start_h:start_h+target_size, start_w:start_w+target_size]
                if batch_idx == 0:
                    print(f'The diffraction pattern was fftshifted and cropped from ({h}, {w}) to ({target_size}, {target_size}) to match the model input size.')

            if batch_idx == 0:
                print(f'diff shape after crop: {input_diff.shape}')
                preview = input_diff[0, 0].cpu().numpy()
                plt.imshow(preview, cmap='gray')
                plt.title('Cropped Input Diffraction Pattern (log scale)')
                plt.colorbar()
                plt.show(block=False)
                plt.pause(3)   # replaces time.sleep
                plt.close()
                print('Continuing inference...')

            # Convert complex probe to real format: (B, 1, modes, H, W) -> (B, 1, modes, H, W, 2)
            # Note: OPR dimension (the 1) already exists from dataset
            if probe.is_complex():
                probe = torch.stack([probe.real, probe.imag], dim=-1)
            # Pad probe to match model's expected size (256x256)
            # probe shape: (B, 1, modes, H, W, 2)
            h, w = probe.shape[-3], probe.shape[-2]
            if h < target_size or w < target_size:
                pad_h = target_size - h
                pad_w = target_size - w
                # Center-pad: distribute padding equally on all sides
                pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                # Pad format: (left, right, top, bottom, ...) working backwards from last dim
                # For shape (..., H, W, 2): skip dim -1, then W (left/right), then H (top/bottom)
                probe = torch.nn.functional.pad(probe, (0, 0, pad_left, pad_right, pad_top, pad_bottom))
                if batch_idx == 0:
                    print(f'Probe was padded from ({h}, {w}) to ({target_size}, {target_size}) to match the model input size.')
            input_probe = probe.to(device)
            input_norm = norm.to(device)
            input_scale = scale.to(device)

            # Forward pass
            output_diff, output_amp, output_ph = model(
                input_diff, input_probe, input_norm, input_scale
            )

            # Store results (move to CPU)
            # Use squeeze(1) to only remove channel dim, keep batch dim -> (B, H, W)
            pred_amp_batch = output_amp.squeeze(1).cpu()
            pred_ph_batch  = output_ph.squeeze(1).cpu()
            # [MODIFIED] Fix #9: patches were transposed at extraction time to match
            # the model's frame (HXN convention). Un-transpose before saving and
            # stitching so the output arrays are in the on-disk object frame.
            if dataset.transpose_object_patches:
                pred_amp_batch = pred_amp_batch.transpose(-2, -1).contiguous()
                pred_ph_batch  = pred_ph_batch.transpose(-2, -1).contiguous()
            all_pred_diff.append(output_diff.squeeze(1).cpu().numpy())
            all_pred_amp.append(pred_amp_batch.numpy())
            all_pred_ph.append(pred_ph_batch.numpy())
            all_gt_diff.append(input_diff.squeeze(1).cpu().numpy())
            all_positions.append(probe_pos.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} patterns")

    # Concatenate batches into (N, H, W) arrays
    pred_diff = np.concatenate(all_pred_diff, axis=0)
    pred_amp = np.concatenate(all_pred_amp, axis=0)
    pred_ph = np.concatenate(all_pred_ph, axis=0)
    gt_diff = np.concatenate(all_gt_diff, axis=0)
    positions = np.concatenate(all_positions, axis=0)

    # Stitch patches into full object
    print(f"\nStitching {len(positions)} patches...")
    # [MODIFIED] align stitching with training.py generate_test_plot:
    #   - patch_size derived from actual array shape (not config model img_size)
    #   - crop=central_crop from config (trims Fourier boundary edges per patch)
    #   - canvas_pad=64 prevents Fourier wrap-around stripes at canvas edges
    #   - image_shape removed; canvas auto-sizes from position bounding box
    patch_size   = pred_amp.shape[-1]
    central_crop = config['training'].get('test_plot_central_crop', 0)
    canvas_pad   = 64
    if dataset._cached_probe_positions is None:
        dataset._cache_positions()
    print(f'Run amplitude stitching...')
    stitched_amp, counts_amp = stitch_patches(pred_amp, positions, patch_size, crop=central_crop, canvas_pad=canvas_pad)
    print(f'Run phase stitching...')
    stitched_ph, counts_ph = stitch_patches(pred_ph, positions, patch_size, crop=central_crop, canvas_pad=canvas_pad)
    print(f"  Stitched object size: {stitched_amp.shape}")

    # Save results
    data_name = Path(data_path).stem.replace('_dp', '')

    # Save individual patches
    np.save(os.path.join(results_dir, f'{data_name}_pred_amplitude.npy'), pred_amp)
    np.save(os.path.join(results_dir, f'{data_name}_pred_phase.npy'), pred_ph)
    np.save(os.path.join(results_dir, f'{data_name}_pred_diffraction.npy'), pred_diff)
    np.save(os.path.join(results_dir, f'{data_name}_input_diffraction.npy'), gt_diff)
    np.save(os.path.join(results_dir, f'{data_name}_positions.npy'), positions)

    # Save stitched results
    np.save(os.path.join(results_dir, f'{data_name}_stitched_amplitude.npy'), stitched_amp)
    np.save(os.path.join(results_dir, f'{data_name}_stitched_phase.npy'), stitched_ph)
    np.save(os.path.join(results_dir, f'{data_name}_stitch_counts.npy'), counts_amp)

    print(f"\nResults saved to: {results_dir}")
    print(f"  Patches:")
    print(f"    - {data_name}_pred_amplitude.npy: shape {pred_amp.shape}")
    print(f"    - {data_name}_pred_phase.npy: shape {pred_ph.shape}")
    print(f"    - {data_name}_pred_diffraction.npy: shape {pred_diff.shape}")
    print(f"  Stitched:")
    print(f"    - {data_name}_stitched_amplitude.npy: shape {stitched_amp.shape}")
    print(f"    - {data_name}_stitched_phase.npy: shape {stitched_ph.shape}")

    return pred_amp, pred_ph, pred_diff, gt_diff, stitched_amp, stitched_ph, counts_amp, positions


def visualize_sample(pred_amp, pred_ph, pred_diff, gt_diff, sample_idx=0, save_path=None,
                     amp_contrast='percentile', amp_percentile=(1, 99), amp_vmin=None, amp_vmax=None):
    """
    Visualize a single prediction.

    Args:
        pred_amp, pred_ph, pred_diff, gt_diff: Prediction arrays
        sample_idx: Index of sample to visualize
        save_path: Path to save figure (optional)
        amp_contrast: Contrast mode for amplitude ('percentile', 'minmax', or 'manual')
        amp_percentile: Percentile range for 'percentile' mode, e.g. (1, 99) or (5, 95)
        amp_vmin, amp_vmax: Manual vmin/vmax for 'manual' mode
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'Scan {SCAN_ID}', fontsize=16, y=1)

    # Input diffraction INTENSITY (amplitude^2 with log scale for better visualization)
    diff_amp = gt_diff[sample_idx]
    diff_intensity = diff_amp ** 2  # Convert amplitude to intensity
    axes[0, 0].imshow(np.log1p(diff_intensity), cmap='gray')
    axes[0, 0].set_title(f'Input Diffraction Intensity (log scale) {sample_idx}')
    axes[0, 0].axis('off')

    # Reconstructed diffraction INTENSITY (amplitude^2 with log scale)
    pred_diff_amp = pred_diff[sample_idx]
    pred_diff_intensity = pred_diff_amp ** 2  # Convert amplitude to intensity
    axes[0, 1].imshow(np.log1p(pred_diff_intensity), cmap='gray')
    axes[0, 1].set_title(f'Reconstructed Diffraction Intensity (log scale) {sample_idx}')
    axes[0, 1].axis('off')

    # Predicted amplitude with contrast from central 50% of the patch
    amp_img = pred_amp[sample_idx]
    h, w = amp_img.shape
    ch, cw = h // 4, w // 4
    amp_center = amp_img[ch:-ch, cw:-cw]
    if amp_contrast == 'percentile':
        vmin = np.percentile(amp_center, amp_percentile[0])
        vmax = np.percentile(amp_center, amp_percentile[1])
    elif amp_contrast == 'manual' and amp_vmin is not None and amp_vmax is not None:
        vmin, vmax = amp_vmin, amp_vmax
    else:  # 'minmax'
        vmin, vmax = amp_center.min(), amp_center.max()

    im_amp = axes[1, 0].imshow(amp_img, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f'Predicted Amplitude (contrast: {amp_contrast}) {sample_idx}')
    axes[1, 0].axis('off')
    plt.colorbar(im_amp, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Predicted phase with contrast from central 50% of the patch (mean ± 2σ)
    ph_img = pred_ph[sample_idx]
    ph_center = ph_img[ch:-ch, cw:-cw]
    ph_vmin = ph_center.mean() - 2 * ph_center.std()
    ph_vmax = ph_center.mean() + 2 * ph_center.std()
    im_ph = axes[1, 1].imshow(ph_img, cmap='magma', vmin=ph_vmin, vmax=ph_vmax)
    axes[1, 1].set_title(f'Predicted Phase {sample_idx}')
    axes[1, 1].axis('off')
    plt.colorbar(im_ph, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def visualize_stitched(stitched_amp, stitched_ph, save_path=None,
                       amp_contrast='percentile', amp_percentile=(1, 99),
                       ph_contrast='full', ph_percentile=(1, 99), counts=None,
                       cropped_height=CH, cropped_width=CW, invert_y=False, scan_id=SCAN_ID,
                       central_contrast=False):
    """
    Visualize stitched reconstruction.

    Args:
        stitched_amp: Stitched amplitude image
        stitched_ph: Stitched phase image
        save_path: Path to save figure (optional)
        amp_contrast: Contrast mode for amplitude ('percentile' or 'minmax')
        amp_percentile: Percentile range for contrast
        ph_contrast: Contrast mode for phase ('percentile', 'full', or 'std')
        ph_percentile: Percentile range when ph_contrast='percentile'
        central_contrast: If True, compute contrast limits from the central 50% of
                          the image only (matches training.py generate_test_plot behaviour).
                          amplitude uses percentile, phase uses mean±2σ.
                          If False (default), contrast is computed from the full image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Scan {scan_id}', fontsize=16, y=1)

    # Crop to valid (non-padded) region using counts
    if counts is not None:
        valid = counts > 0
        rows = np.any(valid, axis=1)
        cols = np.any(valid, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        stitched_amp = stitched_amp[rmin:rmax+1, cmin:cmax+1]
        stitched_ph  = stitched_ph[rmin:rmax+1, cmin:cmax+1]

    ch, cw = cropped_height, cropped_width
    h, w = stitched_amp.shape
    if h >= ch and w >= cw:
        r0, c0 = (h - ch) // 2, (w - cw) // 2
        stitched_amp = stitched_amp[r0:r0+ch, c0:c0+cw]
        stitched_ph  = stitched_ph[r0:r0+ch, c0:c0+cw]

    # Region used for contrast calculation
    if central_contrast:
        ch4, cw4 = stitched_amp.shape[0] // 4, stitched_amp.shape[1] // 4
        amp_ref = stitched_amp[ch4:-ch4, cw4:-cw4]
        ph_ref  = stitched_ph [ch4:-ch4, cw4:-cw4]
    else:
        amp_ref = stitched_amp
        ph_ref  = stitched_ph

    # Amplitude contrast
    if amp_contrast == 'percentile':
        vmin = np.percentile(amp_ref, amp_percentile[0])
        vmax = np.percentile(amp_ref, amp_percentile[1])
    else:
        vmin, vmax = amp_ref.min(), amp_ref.max()

    # Phase contrast
    if central_contrast:
        ph_vmin = ph_ref.mean() - 2 * ph_ref.std()
        ph_vmax = ph_ref.mean() + 2 * ph_ref.std()
    elif ph_contrast == 'percentile':
        ph_vmin = np.percentile(ph_ref, ph_percentile[0])
        ph_vmax = np.percentile(ph_ref, ph_percentile[1])
    else:  # 'full' — keep the physical -pi to pi range
        ph_vmin, ph_vmax = -np.pi, np.pi

    

    im_amp = axes[0].imshow(stitched_amp, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Stitched Amplitude ({stitched_amp.shape[0]}x{stitched_amp.shape[1]})')
    # axes[0].axis('off')
    plt.colorbar(im_amp, ax=axes[0], fraction=0.046, pad=0.04)

    # Phase
    im_ph = axes[1].imshow(stitched_ph, cmap='magma', vmin=ph_vmin, vmax=ph_vmax)
    axes[1].set_title(f'Stitched Phase ({stitched_ph.shape[0]}x{stitched_ph.shape[1]})')
    # axes[1].axis('off')
    plt.colorbar(im_ph, ax=axes[1], fraction=0.046, pad=0.04)

    if invert_y:
        axes[0].invert_yaxis()
        axes[1].invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Stitched figure saved to: {save_path}")

    plt.show()


def main():
    # Check if data path is set
    if EXPERIMENTAL_DATA_PATH is None:
        print("=" * 60)
        print("SETUP REQUIRED")
        print("=" * 60)
        print("\nEdit this script and set EXPERIMENTAL_DATA_PATH to your data file.")
        print("\nExample:")
        print("  EXPERIMENTAL_DATA_PATH = 'C:/path/to/your_data_dp.hdf5'")
        print("=" * 60)
        return

    # Check files exist
    if not os.path.exists(EXPERIMENTAL_DATA_PATH):
        print(f"ERROR: Data file not found: {EXPERIMENTAL_DATA_PATH}")
        return

    if not MODEL_PATH.exists():
        print(f"ERROR: Model checkpoint not found: {MODEL_PATH}")
        return

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    model, config = load_model(CONFIG_PATH, MODEL_PATH, device)

    # Run inference
    pred_amp, pred_ph, pred_diff, gt_diff, stitched_amp, stitched_ph, counts, positions = run_inference(
        model, EXPERIMENTAL_DATA_PATH, config, device, RESULTS_DIR, BATCH_SIZE
    )

    # Visualize first sample (individual patch)
    visualize_sample(
        pred_amp, pred_ph, pred_diff, gt_diff,
        sample_idx=SLICE_INDEX,
        save_path=os.path.join(RESULTS_DIR, f'sample_visualization_{SCAN_ID}.png'),
        amp_contrast=AMP_CONTRAST,
        amp_percentile=AMP_PERCENTILE,
        amp_vmin=AMP_VMIN,
        amp_vmax=AMP_VMAX
    )

    # Visualize stitched result
    visualize_stitched(
        stitched_amp, stitched_ph,
        save_path=os.path.join(RESULTS_DIR, f'stitched_visualization_{SCAN_ID}.png'),
        amp_contrast=AMP_CONTRAST,
        amp_percentile=AMP_PERCENTILE,
        ph_contrast=PH_CONTRAST,
        ph_percentile=PH_PERCENTILE,
        counts=counts
    )

    # Visualize stitching geometry + overlap heatmap
    patch_size = config['model']['encoder']['img_size']
    visualize_stitching(
        positions, counts, patch_size=patch_size,
        save_path=os.path.join(RESULTS_DIR, f'stitching_visualization_{SCAN_ID}.png')
    )


if __name__ == '__main__':
    # Record running time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal inference time: {end_time - start_time:.2f} seconds")
