"""
Test script to verify model selection between ViT and CNN works correctly.
"""
import sys
import os
# Add parent directory to path to import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from model import PtychoViT
from model_cnn import PtychoCNN, PtychoCNN256


def load_config(config_path='../config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_model_initialization(model_type):
    """Test model initialization for given model type."""
    print(f"\n{'='*60}")
    print(f"Testing {model_type.upper()} model initialization")
    print(f"{'='*60}")

    # Load config
    config = load_config()

    # Override model type
    config['model']['model_type'] = model_type

    # Initialize model based on type
    if model_type == 'vit':
        model = PtychoViT(config=config['model'])
        img_size = config['model']['encoder']['img_size']
    elif model_type == 'cnn':
        model = PtychoCNN(config=config['model']['cnn'])
        img_size = 512
    elif model_type == 'cnn256':
        model = PtychoCNN256(config=config['model']['cnn256'])
        img_size = 256
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"✓ Model created successfully: {model.__class__.__name__}")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, img_size, img_size)
    probe = torch.randn(batch_size, 1, 8, img_size, img_size, 2)
    normalization = torch.randn(batch_size, 1)
    scale = torch.randn(batch_size, 1)

    pred_diff_amp, amp, ph = model(x, probe, normalization, scale)

    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Predicted diffraction amplitude: {pred_diff_amp.shape}")
    print(f"  Amplitude: {amp.shape}")
    print(f"  Phase: {ph.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Model Selection Test Suite")
    print("="*60)

    success = True

    # Test ViT model
    try:
        test_model_initialization('vit')
    except Exception as e:
        print(f"✗ ViT model test failed: {e}")
        success = False

    # Test CNN model
    try:
        test_model_initialization('cnn')
    except Exception as e:
        print(f"✗ CNN model test failed: {e}")
        success = False

    # Test CNN256 model
    try:
        test_model_initialization('cnn256')
    except Exception as e:
        print(f"✗ CNN256 model test failed: {e}")
        success = False

    print("\n" + "="*60)
    if success:
        print("✓ All tests passed!")
        print("\nYou can now select between models by setting 'model_type'")
        print("in config.yaml to 'vit', 'cnn' (512x512), or 'cnn256' (256x256)")
    else:
        print("✗ Some tests failed")
    print("="*60 + "\n")
