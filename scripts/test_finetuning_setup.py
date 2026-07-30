#!/usr/bin/env python3
"""
Quick test to verify fine-tuning setup before running full training.
Tests: data loading, model loading, forward pass, and SSIM/PSNR metrics.
"""

from __future__ import annotations

import argparse

import torch
import yaml
from pathlib import Path


def test_config(config_path: Path):
    """Test that config loads and paths exist."""
    print("=" * 60)
    print("TEST 1: Configuration")
    print("=" * 60)

    print(f"Config path: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check data path
    data_path = Path(config['data']['data_path'])
    print(f"Data path: {data_path}")
    print(f"  Exists: {data_path.exists()}")

    if data_path.exists():
        dp_files = list(data_path.glob("*_dp.hdf5"))
        print(f"  Found {len(dp_files)} *_dp.hdf5 files")

    # Check normalization
    norm_path = config['data'].get('normalization_dict_path')
    if norm_path:
        norm_path = Path(norm_path)
        print(f"Normalization path: {norm_path}")
        print(f"  Exists: {norm_path.exists()}")

    # Check finetune model
    finetune_path = config['training'].get('finetune_from_model')
    if finetune_path:
        finetune_path = Path(finetune_path)
        print(f"Finetune model path: {finetune_path}")
        print(f"  Exists: {finetune_path.exists()}")

    print(f"Train split: {config['data']['train_split']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Epochs: {config['training']['epochs']}")

    print("\n[PASS] Config loaded successfully\n")
    return config


def test_data_loading(config, num_samples=3):
    """Test that data loads correctly."""
    print("=" * 60)
    print("TEST 2: Data Loading")
    print("=" * 60)

    from ptycho_vit.data import PtychographyDataset

    # Get first file from data path
    data_path = Path(config['data']['data_path'])
    dp_files = sorted(data_path.glob("*_dp.hdf5"))[:num_samples]

    for dp_file in dp_files:
        print(f"\nLoading: {dp_file.name}")
        dataset = PtychographyDataset(
            file_path=str(dp_file),
            scale=config['data']['scale'],
            normalization_dict_path=config['data'].get('normalization_dict_path'),
            apply_noise=False,
            max_probe_modes=config['data'].get('max_probe_modes', 8),
            target_size=config['data'].get('target_size', 256)
        )

        print(f"  Num patterns: {len(dataset)}")
        print(f"  Pattern shape: {dataset.pattern_shape}")
        print(f"  Object shape: {dataset.object_shape}")
        print(f"  Normalization: {dataset.normalization}")

        # Load one sample
        sample = dataset[0]
        diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale = sample

        print("  Sample shapes:")
        print(f"    diff_amp: {diff_amp.shape}, dtype: {diff_amp.dtype}")
        print(f"    amp_patch: {amp_patch.shape}, range: [{amp_patch.min():.3f}, {amp_patch.max():.3f}]")
        print(f"    ph_patch: {ph_patch.shape}, range: [{ph_patch.min():.3f}, {ph_patch.max():.3f}]")
        print(f"    probe: {probe.shape}")

        dataset.close()

    print("\n[PASS] Data loading works\n")
    return dataset.pattern_shape


def test_model_loading(config):
    """Test that model loads with pretrained weights."""
    print("=" * 60)
    print("TEST 3: Model Loading")
    print("=" * 60)

    from ptycho_vit.model.model import PtychoViT

    # Create model
    model = PtychoViT(config=config['model'])
    print("Model created: PtychoViT")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Check log_scale values before loading
    print(f"  log_scale_amp (before load): {model.log_scale_amp.item():.4f} (exp={torch.exp(model.log_scale_amp).item():.4f})")
    print(f"  log_scale_ph (before load): {model.log_scale_ph.item():.4f} (exp={torch.exp(model.log_scale_ph).item():.4f})")

    # Load pretrained weights
    finetune_path = config['training'].get('finetune_from_model')
    if finetune_path and Path(finetune_path).exists():
        state = torch.load(finetune_path, map_location='cpu')
        model.load_state_dict(state)
        print(f"\n  Loaded weights from: {finetune_path}")
        print(f"  log_scale_amp (after load): {model.log_scale_amp.item():.4f} (exp={torch.exp(model.log_scale_amp).item():.4f})")
        print(f"  log_scale_ph (after load): {model.log_scale_ph.item():.4f} (exp={torch.exp(model.log_scale_ph).item():.4f})")
    else:
        print(f"\n  [WARNING] Finetune path not found: {finetune_path}")

    print("\n[PASS] Model loading works\n")
    return model


def test_forward_pass(model, config):
    """Test forward pass with dummy data."""
    print("=" * 60)
    print("TEST 4: Forward Pass")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = model.to(device)
    model.eval()

    # Create dummy inputs
    batch_size = 2
    img_size = 256
    max_probe_modes = config['data'].get('max_probe_modes', 8)

    dummy_diff = torch.randn(batch_size, 1, img_size, img_size, device=device)
    dummy_probe = torch.randn(batch_size, 1, max_probe_modes, img_size, img_size, 2, device=device)
    dummy_norm = torch.tensor([[100000.0]] * batch_size, device=device)
    dummy_scale = torch.tensor([[10000.0]] * batch_size, device=device)

    print("Input shapes:")
    print(f"  diff: {dummy_diff.shape}")
    print(f"  probe: {dummy_probe.shape}")

    with torch.no_grad():
        output_diff, output_amp, output_ph = model(dummy_diff, dummy_probe, dummy_norm, dummy_scale)

    print("Output shapes:")
    print(f"  output_diff: {output_diff.shape}")
    print(f"  output_amp: {output_amp.shape}, range: [{output_amp.min():.3f}, {output_amp.max():.3f}]")
    print(f"  output_ph: {output_ph.shape}, range: [{output_ph.min():.3f}, {output_ph.max():.3f}]")

    print("\n[PASS] Forward pass works\n")
    return output_amp, output_ph


def test_metrics():
    """Test SSIM and PSNR computation."""
    print("=" * 60)
    print("TEST 5: SSIM/PSNR Metrics")
    print("=" * 60)

    from ptycho_vit.training import compute_ssim, compute_psnr

    # Create test tensors
    img_size = 256
    gt = torch.rand(1, 1, img_size, img_size)

    # Test with identical images (should give perfect scores)
    pred_identical = gt.clone()
    ssim_identical = compute_ssim(pred_identical.squeeze(), gt.squeeze())
    psnr_identical = compute_psnr(pred_identical.squeeze(), gt.squeeze())
    print("Identical images:")
    print(f"  SSIM: {ssim_identical:.4f} (expected: ~1.0)")
    print(f"  PSNR: {psnr_identical:.2f} dB (expected: very high/inf)")

    # Test with noisy image
    noise = torch.randn_like(gt) * 0.1
    pred_noisy = gt + noise
    ssim_noisy = compute_ssim(pred_noisy.squeeze(), gt.squeeze())
    psnr_noisy = compute_psnr(pred_noisy.squeeze(), gt.squeeze())
    print("Noisy image (10% noise):")
    print(f"  SSIM: {ssim_noisy:.4f} (expected: < 1.0)")
    print(f"  PSNR: {psnr_noisy:.2f} dB (expected: ~20-30 dB)")

    # Test with very different image
    pred_random = torch.rand_like(gt)
    ssim_random = compute_ssim(pred_random.squeeze(), gt.squeeze())
    psnr_random = compute_psnr(pred_random.squeeze(), gt.squeeze())
    print("Random image:")
    print(f"  SSIM: {ssim_random:.4f} (expected: low)")
    print(f"  PSNR: {psnr_random:.2f} dB (expected: low)")

    # Sanity checks
    assert ssim_identical > 0.99, f"SSIM for identical images should be ~1.0, got {ssim_identical}"
    assert ssim_noisy < ssim_identical, "SSIM for noisy should be less than identical"
    assert psnr_noisy < psnr_identical, "PSNR for noisy should be less than identical"

    print("\n[PASS] SSIM/PSNR metrics work correctly\n")


def test_real_inference(model, config):
    """Test inference on real data sample."""
    print("=" * 60)
    print("TEST 6: Real Data Inference")
    print("=" * 60)

    from ptycho_vit.data import PtychographyDataset
    from ptycho_vit.training import compute_ssim, compute_psnr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Load real data
    data_path = Path(config['data']['data_path'])
    dp_file = sorted(data_path.glob("*_dp.hdf5"))[0]

    dataset = PtychographyDataset(
        file_path=str(dp_file),
        scale=config['data']['scale'],
        normalization_dict_path=config['data'].get('normalization_dict_path'),
        apply_noise=False,
        max_probe_modes=config['data'].get('max_probe_modes', 8),
        target_size=config['data'].get('target_size', 256)
    )

    print(f"  Raw pattern shape: {dataset._raw_pattern_shape}")
    print(f"  Target pattern shape: {dataset.pattern_shape}")

    # Get a sample
    diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale = dataset[0]

    # Add batch dimension and move to device
    diff_amp = diff_amp.unsqueeze(0).to(device)
    amp_patch = amp_patch.unsqueeze(0).to(device)
    ph_patch = ph_patch.unsqueeze(0).to(device)
    probe = torch.view_as_real(probe.unsqueeze(0)).to(device)
    norm = torch.tensor([[norm]], device=device)
    scale = torch.tensor([[scale]], device=device)

    print(f"Test file: {dp_file.name}")
    print(f"Input diff_amp range: [{diff_amp.min():.3f}, {diff_amp.max():.3f}]")
    print(f"GT amp_patch range: [{amp_patch.min():.3f}, {amp_patch.max():.3f}]")
    print(f"GT ph_patch range: [{ph_patch.min():.3f}, {ph_patch.max():.3f}]")

    with torch.no_grad():
        output_diff, output_amp, output_ph = model(diff_amp, probe, norm, scale)

    print(f"\nPredicted output_amp range: [{output_amp.min():.3f}, {output_amp.max():.3f}]")
    print(f"Predicted output_ph range: [{output_ph.min():.3f}, {output_ph.max():.3f}]")

    # Compute metrics
    ssim_amp = compute_ssim(output_amp.squeeze(), amp_patch.squeeze())
    psnr_amp = compute_psnr(output_amp.squeeze(), amp_patch.squeeze())
    ssim_ph = compute_ssim(output_ph.squeeze(), ph_patch.squeeze())
    psnr_ph = compute_psnr(output_ph.squeeze(), ph_patch.squeeze())

    print("\nMetrics vs ground truth reconstruction:")
    print(f"  Amplitude - SSIM: {ssim_amp:.4f}, PSNR: {psnr_amp:.2f} dB")
    print(f"  Phase     - SSIM: {ssim_ph:.4f}, PSNR: {psnr_ph:.2f} dB")

    dataset.close()

    print("\n[PASS] Real data inference works\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sanity-check fine-tuning setup before running training.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config.yaml"),
        help="Path to config.yaml (default: <repo>/config.yaml)",
    )
    parser.add_argument("--num-samples", type=int, default=2, help="Number of files to probe in data-loading test")
    parser.add_argument(
        "--skip-real-inference",
        action="store_true",
        help="Skip running a forward pass on real data (avoids GPU / heavy I/O).",
    )
    args = parser.parse_args(argv)

    print("\n" + "=" * 60)
    print("FINE-TUNING SETUP TEST")
    print("=" * 60 + "\n")

    try:
        # Run all tests
        config = test_config(Path(args.config))
        test_data_loading(config, num_samples=args.num_samples)
        model = test_model_loading(config)
        test_forward_pass(model, config)
        test_metrics()
        if not args.skip_real_inference:
            test_real_inference(model, config)

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou're ready to run fine-tuning:")
        print("  python main.py")
        print()

    except Exception as e:
        print(f"\n[FAILED] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
