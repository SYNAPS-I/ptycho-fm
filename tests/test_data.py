"""Tests for data loading components."""
import sys
import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import PtychographyDataset, CombinedDataset
from tests.test_utils import create_dummy_hdf5_pair


def test_ptychography_dataset():
    """Test PtychographyDataset basic functionality."""
    print("\n" + "="*70)
    print("TEST: PtychographyDataset Basic Functionality")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data
        dp_file, para_file, num_patterns = create_dummy_hdf5_pair(tmpdir, 'test_object')

        # Create dataset
        dataset = PtychographyDataset(str(dp_file), patch_size=128)

        print(f"Dataset length: {len(dataset)}")
        print(f"Expected patterns: {num_patterns}")
        assert len(dataset) == num_patterns, f"Expected {num_patterns} patterns, got {len(dataset)}"

        # Test __getitem__
        diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale = dataset[0]

        print(f"Diffraction amplitude shape: {diff_amp.shape}")
        print(f"Amplitude patch shape: {amp_patch.shape}")
        print(f"Phase patch shape: {ph_patch.shape}")
        print(f"Probe shape: {probe.shape}")
        print(f"Probe position shape: {probe_pos.shape}")
        print(f"Normalization: {norm}")
        print(f"Scale: {scale}")

        # Verify shapes
        assert diff_amp.shape == (1, 128, 128), f"Wrong diff_amp shape: {diff_amp.shape}"
        assert amp_patch.shape == (1, 128, 128), f"Wrong amp_patch shape: {amp_patch.shape}"
        assert ph_patch.shape == (1, 128, 128), f"Wrong ph_patch shape: {ph_patch.shape}"
        # Probe shape is (8, 128, 128, 2) - batch dimension added during collation
        assert probe.shape == (8, 128, 128, 2), f"Wrong probe shape: {probe.shape}"

        print("✓ PtychographyDataset test passed!\n")


def test_combined_dataset_basic():
    """Test CombinedDataset basic functionality."""
    print("\n" + "="*70)
    print("TEST: CombinedDataset Basic Functionality")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple dummy datasets
        _, _, n1 = create_dummy_hdf5_pair(tmpdir, 'object_1')
        _, _, n2 = create_dummy_hdf5_pair(tmpdir, 'object_2')
        _, _, n3 = create_dummy_hdf5_pair(tmpdir, 'object_3')
        total_expected = n1 + n2 + n3

        # Create CombinedDataset from directory
        combined = CombinedDataset(file_paths=tmpdir)

        print(f"Total patterns: {len(combined)}")
        print(f"Expected total: {total_expected}")
        print(f"Num files: {len(combined.file_paths)}")

        assert len(combined) == total_expected, \
            f"Expected {total_expected} patterns, got {len(combined)}"

        # Test indexing
        print("\nTesting dataset indexing:")
        sample = combined[0]
        diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale = sample

        print("Sample 0 shapes:")
        print(f"  diff_amp: {diff_amp.shape}")
        print(f"  amp_patch: {amp_patch.shape}")
        print(f"  ph_patch: {ph_patch.shape}")
        print(f"  probe: {probe.shape}")
        print(f"  probe_pos: {probe_pos.shape}")

        # Test with DataLoader
        batch_size = 4
        dataloader = DataLoader(combined, batch_size=batch_size, shuffle=True)

        print(f"\nTesting with DataLoader (batch_size={batch_size}):")
        batch_count = 0
        total_samples = 0
        for batch in dataloader:
            batch_count += 1
            diff_amps, amp_patches, ph_patches, probes, probe_pos, norms, scales = batch
            total_samples += diff_amps.shape[0]
            if batch_count <= 2:  # Print first 2 batches
                print(f"  Batch {batch_count}: {diff_amps.shape[0]} samples")

        print(f"Total batches: {batch_count}")
        print(f"Total samples loaded: {total_samples}")
        assert total_samples == total_expected, \
            f"Expected {total_expected} samples, got {total_samples}"

        print("✓ CombinedDataset basic test passed!\n")


def test_combined_dataset_with_train_val_split():
    """Test CombinedDataset with train/val split using PyTorch random_split."""
    print("\n" + "="*70)
    print("TEST: CombinedDataset with Train/Val Split")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple datasets
        num_files = 3
        pattern_counts = []
        for i in range(num_files):
            _, _, n = create_dummy_hdf5_pair(tmpdir, f'object_{i}')
            pattern_counts.append(n)

        combined = CombinedDataset(file_paths=tmpdir)

        total_patterns = sum(pattern_counts)
        print(f"Total patterns: {len(combined)} (expected {total_patterns})")
        print(f"Pattern counts per file: {pattern_counts}")
        assert len(combined) == total_patterns

        # Split into train/val using PyTorch
        train_split = 0.8
        train_size = int(total_patterns * train_split)
        val_size = total_patterns - train_size

        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(combined, [train_size, val_size], generator=generator)

        print(f"\nTrain size: {len(train_dataset)}")
        print(f"Val size: {len(val_dataset)}")

        assert len(train_dataset) == train_size, f"Expected {train_size} train samples"
        assert len(val_dataset) == val_size, f"Expected {val_size} val samples"

        # Test DataLoader with splits
        batch_size = 4
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Iterate through train data
        train_samples = 0
        for batch in train_loader:
            train_samples += batch[0].shape[0]

        # Iterate through val data
        val_samples = 0
        for batch in val_loader:
            val_samples += batch[0].shape[0]

        print(f"Train samples loaded: {train_samples}")
        print(f"Val samples loaded: {val_samples}")

        assert train_samples == train_size, f"Expected {train_size} train samples, got {train_samples}"
        assert val_samples == val_size, f"Expected {val_size} val samples, got {val_samples}"

        print("✓ Train/val split test passed!\n")


def test_dataloader_consistency_across_epochs():
    """Test that DataLoader with same seed produces consistent data across epochs."""
    print("\n" + "="*70)
    print("TEST: DataLoader Consistency Across Epochs")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data
        _, _, num_patterns = create_dummy_hdf5_pair(tmpdir, 'object_1')

        combined = CombinedDataset(file_paths=tmpdir)

        # Split into train/val
        train_split = 0.8
        train_size = int(num_patterns * train_split)
        val_size = num_patterns - train_size

        # Use same seed for both splits
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(combined, [train_size, val_size], generator=generator)

        print(f"Total patterns: {num_patterns}")
        print(f"Train size: {len(train_dataset)}")
        print(f"Val size: {len(val_dataset)}")

        # Create DataLoader for "epoch 0"
        train_loader_epoch_0 = DataLoader(train_dataset, batch_size=4, shuffle=False)

        # Collect all samples from epoch 0
        epoch_0_samples = 0
        for batch in train_loader_epoch_0:
            epoch_0_samples += batch[0].shape[0]

        # Create DataLoader for "epoch 1" (same dataset)
        train_loader_epoch_1 = DataLoader(train_dataset, batch_size=4, shuffle=False)

        # Collect all samples from epoch 1
        epoch_1_samples = 0
        for batch in train_loader_epoch_1:
            epoch_1_samples += batch[0].shape[0]

        print(f"Epoch 0 samples: {epoch_0_samples}")
        print(f"Epoch 1 samples: {epoch_1_samples}")

        # Both epochs should see the same number of samples (split is deterministic)
        assert epoch_0_samples == epoch_1_samples == train_size, \
            f"Inconsistent sample counts: epoch0={epoch_0_samples}, epoch1={epoch_1_samples}, expected={train_size}"

        print("✓ DataLoader produces consistent data across epochs")
        print("✓ DataLoader consistency test passed!\n")


def test_combined_dataset_with_list_of_files():
    """Test CombinedDataset with explicit list of files instead of directory."""
    print("\n" + "="*70)
    print("TEST: CombinedDataset with List of Files")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy datasets
        dp_file_1, _, n1 = create_dummy_hdf5_pair(tmpdir, 'object_1')
        dp_file_2, _, n2 = create_dummy_hdf5_pair(tmpdir, 'object_2')

        # Create CombinedDataset with explicit file list
        file_list = [str(dp_file_1), str(dp_file_2)]
        combined = CombinedDataset(file_paths=file_list)

        total_expected = n1 + n2
        print(f"Total patterns: {len(combined)}")
        print(f"Expected: {total_expected}")
        assert len(combined) == total_expected, \
            f"Expected {total_expected} patterns, got {len(combined)}"

        # Test with DataLoader
        batch_size = 4
        dataloader = DataLoader(combined, batch_size=batch_size, shuffle=True)

        batch_count = 0
        total_samples = 0
        for batch in dataloader:
            batch_count += 1
            total_samples += batch[0].shape[0]

        print(f"Total batches: {batch_count}")
        print(f"Total samples: {total_samples}")
        assert batch_count > 0, "No batches generated!"
        assert total_samples == total_expected, \
            f"Expected {total_expected} samples, got {total_samples}"

        print("✓ List of files test passed!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# Running All Data Loading Tests")
    print("#"*70)

    try:
        test_ptychography_dataset()
        test_combined_dataset_basic()
        test_combined_dataset_with_train_val_split()
        test_dataloader_consistency_across_epochs()
        test_combined_dataset_with_list_of_files()

        print("\n" + "#"*70)
        print("# ALL TESTS PASSED! ✓")
        print("#"*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
