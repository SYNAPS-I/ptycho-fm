"""Tests for data loading components."""
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import PtychographyDataset
from dataloader import CombinedDataset
from tests.test_utils import create_dummy_hdf5_pair, cleanup_test_files


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

        # Test file handle cleanup
        dataset.close()

        print("✓ PtychographyDataset test passed!\n")


def test_combined_dataset_single_rank():
    """Test CombinedDataset with single rank (no DDP)."""
    print("\n" + "="*70)
    print("TEST: CombinedDataset Single Rank")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple dummy datasets
        _, _, n1 = create_dummy_hdf5_pair(tmpdir, 'object_1')
        _, _, n2 = create_dummy_hdf5_pair(tmpdir, 'object_2')
        _, _, n3 = create_dummy_hdf5_pair(tmpdir, 'object_3')
        total_expected = n1 + n2 + n3

        # Create CombinedDataset from directory
        combined = CombinedDataset(
            file_paths=tmpdir,
            train_split=0.8,
            batch_size=4,
            rank=0,
            world_size=1,
            shuffle=True,
            random_seed=42
        )

        print(f"Total patterns: {combined.total_patterns}")
        print(f"Expected total: {total_expected}")
        print(f"Train size: {combined.train_size}")
        print(f"Val size: {combined.val_size}")
        print(f"Num files: {len(combined.file_paths)}")

        assert combined.total_patterns == total_expected, \
            f"Expected {total_expected} patterns, got {combined.total_patterns}"
        expected_train = int(total_expected * 0.8)
        expected_val = total_expected - expected_train
        assert combined.train_size == expected_train, \
            f"Expected {expected_train} train patterns, got {combined.train_size}"
        assert combined.val_size == expected_val, \
            f"Expected {expected_val} val patterns, got {combined.val_size}"

        # Verify split integrity
        combined.verify_split_integrity()
        print("✓ Split integrity verified")

        # Test train batches
        print("\nIterating through train batches:")
        train_batch_count = 0
        for batch, metadata in combined.iterate_batches(split='train', epoch=0):
            train_batch_count += 1
            diff_amps, amp_patches, ph_patches, probes, probe_pos, norms, scales = batch
            print(f"  Train batch {train_batch_count}: {diff_amps.shape[0]} samples "
                  f"from file {metadata['file_num']}/{metadata['total_files']} "
                  f"({metadata['file_name']})")

        print(f"Total train batches: {train_batch_count}")
        expected_train_batches = combined.train_size // combined.batch_size
        # Allow for ±1 batch difference due to incomplete batches being dropped
        assert abs(train_batch_count - expected_train_batches) <= 1, \
            f"Expected ~{expected_train_batches} train batches, got {train_batch_count}"

        # Test val batches
        print("\nIterating through validation batches:")
        val_batch_count = 0
        for batch, metadata in combined.iterate_batches(split='val', epoch=0):
            val_batch_count += 1
            diff_amps, amp_patches, ph_patches, probes, probe_pos, norms, scales = batch
            print(f"  Val batch {val_batch_count}: {diff_amps.shape[0]} samples "
                  f"from file {metadata['file_num']}/{metadata['total_files']} "
                  f"({metadata['file_name']})")

        print(f"Total val batches: {val_batch_count}")
        expected_val_batches = combined.val_size // combined.batch_size
        # Allow for ±1 batch difference due to incomplete batches being dropped
        assert abs(val_batch_count - expected_val_batches) <= 1, \
            f"Expected ~{expected_val_batches} val batches, got {val_batch_count}"

        print("✓ CombinedDataset single rank test passed!\n")


def test_combined_dataset_multi_file_iteration():
    """Test that CombinedDataset properly iterates through multiple files."""
    print("\n" + "="*70)
    print("TEST: CombinedDataset Multi-File Iteration")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 5 small datasets
        num_files = 5
        pattern_counts = []
        for i in range(num_files):
            _, _, n = create_dummy_hdf5_pair(tmpdir, f'object_{i}')
            pattern_counts.append(n)

        combined = CombinedDataset(
            file_paths=tmpdir,
            train_split=0.9,
            batch_size=4,
            rank=0,
            world_size=1,
            shuffle=False,  # Disable shuffle for predictable testing
            random_seed=42
        )

        total_patterns = sum(pattern_counts)
        print(f"Total patterns: {combined.total_patterns} (expected {total_patterns})")
        print(f"Pattern counts per file: {pattern_counts}")
        assert combined.total_patterns == total_patterns

        # Track which files were processed
        files_processed = set()
        total_samples_seen = 0

        print("\nIterating through train batches:")
        for batch, metadata in combined.iterate_batches(split='train', epoch=0):
            files_processed.add(metadata['file_name'])
            batch_size = batch[0].shape[0]
            total_samples_seen += batch_size
            print(f"  File {metadata['file_num']}/{metadata['total_files']}: "
                  f"{metadata['file_name']}, "
                  f"batch {metadata['batch_in_file']}/{metadata['total_batches_in_file']}, "
                  f"batch_size={batch_size}")

        print(f"\nFiles processed: {len(files_processed)}")
        print(f"Total samples seen: {total_samples_seen}")
        print(f"Expected samples: {combined.train_size}")

        assert total_samples_seen == combined.train_size, \
            f"Mismatch: saw {total_samples_seen} samples, expected {combined.train_size}"

        print("✓ Multi-file iteration test passed!\n")


def test_split_consistency_across_epochs():
    """Test that train/val split remains consistent across epochs."""
    print("\n" + "="*70)
    print("TEST: Split Consistency Across Epochs")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data
        _, _, num_patterns = create_dummy_hdf5_pair(tmpdir, 'object_1')

        combined = CombinedDataset(
            file_paths=tmpdir,
            train_split=0.8,
            batch_size=4,
            rank=0,
            world_size=1,
            shuffle=True,
            random_seed=42
        )

        # Collect indices seen in epoch 0
        epoch_0_train_samples = []
        for batch, _ in combined.iterate_batches(split='train', epoch=0):
            epoch_0_train_samples.append(batch[0])
        epoch_0_train_samples = torch.cat(epoch_0_train_samples, dim=0)

        # Collect indices seen in epoch 1
        epoch_1_train_samples = []
        for batch, _ in combined.iterate_batches(split='train', epoch=1):
            epoch_1_train_samples.append(batch[0])
        epoch_1_train_samples = torch.cat(epoch_1_train_samples, dim=0)

        print(f"Epoch 0 train samples: {epoch_0_train_samples.shape[0]}")
        print(f"Epoch 1 train samples: {epoch_1_train_samples.shape[0]}")

        # The number of samples should be the same
        assert epoch_0_train_samples.shape[0] == epoch_1_train_samples.shape[0], \
            "Different number of samples across epochs!"

        # Verify split integrity
        combined.verify_split_integrity()
        print("✓ Split remains consistent across epochs")

        print("✓ Split consistency test passed!\n")


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
        combined = CombinedDataset(
            file_paths=file_list,
            train_split=0.8,
            batch_size=4,
            rank=0,
            world_size=1,
            shuffle=True,
            random_seed=42
        )

        total_expected = n1 + n2
        print(f"Total patterns: {combined.total_patterns}")
        assert combined.total_patterns == total_expected, \
            f"Expected {total_expected} patterns, got {combined.total_patterns}"

        batch_count = 0
        for batch, metadata in combined.iterate_batches(split='train', epoch=0):
            batch_count += 1

        print(f"Train batches: {batch_count}")
        assert batch_count > 0, "No batches generated!"

        print("✓ List of files test passed!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# Running All Data Loading Tests")
    print("#"*70)

    try:
        test_ptychography_dataset()
        test_combined_dataset_single_rank()
        test_combined_dataset_multi_file_iteration()
        test_split_consistency_across_epochs()
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
