"""
Test script to verify rank-based sharding in CombinedDataset.

Tests:
1. No overlap between ranks
2. Total samples = sum of all ranks
3. Train/val split is correct

Usage:
    python tests/test_sharding.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from data import CombinedDataset
import yaml


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / config_path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_sharding():
    """Test rank-based sharding with 2 ranks."""
    print("=" * 70)
    print("Testing Rank-Based Sharding in CombinedDataset")
    print("=" * 70)
    
    # Load config
    config = load_config()
    
    # Get data source
    if 'datafiles' in config['data'] and config['data']['datafiles'] is not None:
        data_source = config['data']['datafiles']
    elif 'data_path' in config['data']:
        data_source = config['data']['data_path']
    else:
        raise ValueError("Config must specify either 'datafiles' (list) or 'data_path' (directory)")
    
    print(f"\nData source: {data_source}")
    
    # Test parameters
    world_size = 2
    train_split = config['data']['train_split']
    random_seed = config['data']['random_seed']
    
    print(f"World size: {world_size}")
    print(f"Train split: {train_split}")
    print(f"Random seed: {random_seed}")
    
    # Create datasets for rank 0 and rank 1 (train mode)
    print("\n" + "-" * 70)
    print("Creating Train Datasets")
    print("-" * 70)
    
    train_rank0 = CombinedDataset(
        file_paths=data_source,
        rank=0,
        world_size=world_size,
        train_split=train_split,
        shuffle=True,
        random_seed=random_seed,
        mode='train',
        scale=config['data']['scale'],
        normalization_dict_path=config['data'].get('normalization_dict_path'),
        apply_noise=False  # Disable noise for testing
    )
    
    train_rank1 = CombinedDataset(
        file_paths=data_source,
        rank=1,
        world_size=world_size,
        train_split=train_split,
        shuffle=True,
        random_seed=random_seed,
        mode='train',
        scale=config['data']['scale'],
        normalization_dict_path=config['data'].get('normalization_dict_path'),
        apply_noise=False
    )
    
    # Create datasets for rank 0 and rank 1 (val mode)
    print("\n" + "-" * 70)
    print("Creating Validation Datasets")
    print("-" * 70)
    
    val_rank0 = CombinedDataset(
        file_paths=data_source,
        rank=0,
        world_size=world_size,
        train_split=train_split,
        shuffle=False,
        random_seed=random_seed,
        mode='val',
        scale=config['data']['scale'],
        normalization_dict_path=config['data'].get('normalization_dict_path'),
        apply_noise=False
    )
    
    val_rank1 = CombinedDataset(
        file_paths=data_source,
        rank=1,
        world_size=world_size,
        train_split=train_split,
        shuffle=False,
        random_seed=random_seed,
        mode='val',
        scale=config['data']['scale'],
        normalization_dict_path=config['data'].get('normalization_dict_path'),
        apply_noise=False
    )
    
    # Get total patterns from first dataset
    total_patterns = train_rank0.total_patterns
    expected_train = int(total_patterns * train_split)
    expected_val = total_patterns - expected_train
    
    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    
    # Test 1: No overlap between ranks (train)
    print("\nTest 1: No overlap between ranks (train)")
    train_indices_rank0 = set(train_rank0.current_indices)
    train_indices_rank1 = set(train_rank1.current_indices)
    train_overlap = train_indices_rank0.intersection(train_indices_rank1)
    
    if len(train_overlap) == 0:
        print("✓ PASS: No overlap between train ranks")
    else:
        print(f"✗ FAIL: Found {len(train_overlap)} overlapping indices in train")
        print(f"  Overlapping indices: {list(train_overlap)[:10]}...")
    
    # Test 2: Total samples = sum of all ranks (train)
    print("\nTest 2: Total samples = sum of all ranks (train)")
    train_total = len(train_rank0) + len(train_rank1)
    print(f"  Expected train samples: {expected_train}")
    print(f"  Actual train samples: {train_total}")
    print(f"  Rank 0: {len(train_rank0)} samples")
    print(f"  Rank 1: {len(train_rank1)} samples")
    
    if train_total == expected_train:
        print("✓ PASS: Train sample count matches expected")
    else:
        print(f"✗ FAIL: Train sample count mismatch (expected {expected_train}, got {train_total})")
    
    # Test 3: No overlap between ranks (val)
    print("\nTest 3: No overlap between ranks (val)")
    val_indices_rank0 = set(val_rank0.current_indices)
    val_indices_rank1 = set(val_rank1.current_indices)
    val_overlap = val_indices_rank0.intersection(val_indices_rank1)
    
    if len(val_overlap) == 0:
        print("✓ PASS: No overlap between val ranks")
    else:
        print(f"✗ FAIL: Found {len(val_overlap)} overlapping indices in val")
        print(f"  Overlapping indices: {list(val_overlap)[:10]}...")
    
    # Test 4: Total samples = sum of all ranks (val)
    print("\nTest 4: Total samples = sum of all ranks (val)")
    val_total = len(val_rank0) + len(val_rank1)
    print(f"  Expected val samples: {expected_val}")
    print(f"  Actual val samples: {val_total}")
    print(f"  Rank 0: {len(val_rank0)} samples")
    print(f"  Rank 1: {len(val_rank1)} samples")
    
    if val_total == expected_val:
        print("✓ PASS: Val sample count matches expected")
    else:
        print(f"✗ FAIL: Val sample count mismatch (expected {expected_val}, got {val_total})")
    
    # Test 5: Train and val don't overlap
    print("\nTest 5: Train and val don't overlap")
    all_train = train_indices_rank0.union(train_indices_rank1)
    all_val = val_indices_rank0.union(val_indices_rank1)
    train_val_overlap = all_train.intersection(all_val)
    
    if len(train_val_overlap) == 0:
        print("✓ PASS: No overlap between train and val")
    else:
        print(f"✗ FAIL: Found {len(train_val_overlap)} overlapping indices between train and val")
        print(f"  Overlapping indices: {list(train_val_overlap)[:10]}...")
    
    # Test 6: All indices are accounted for
    print("\nTest 6: All indices are accounted for")
    all_indices = all_train.union(all_val)
    print(f"  Total patterns: {total_patterns}")
    print(f"  Accounted for: {len(all_indices)}")
    
    if len(all_indices) == total_patterns:
        print("✓ PASS: All indices accounted for")
    else:
        print(f"✗ FAIL: Not all indices accounted for (expected {total_patterns}, got {len(all_indices)})")
    
    # Test 7: Verify deterministic shuffling (same seed = same indices)
    print("\nTest 7: Verify deterministic shuffling")
    train_rank0_v2 = CombinedDataset(
        file_paths=data_source,
        rank=0,
        world_size=world_size,
        train_split=train_split,
        shuffle=True,
        random_seed=random_seed,
        mode='train',
        scale=config['data']['scale'],
        normalization_dict_path=config['data'].get('normalization_dict_path'),
        apply_noise=False
    )
    
    if np.array_equal(train_rank0.current_indices, train_rank0_v2.current_indices):
        print("✓ PASS: Deterministic shuffling works (same seed produces same indices)")
    else:
        print("✗ FAIL: Shuffling is not deterministic")
    
    print("\n" + "=" * 70)
    print("Testing Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_sharding()

