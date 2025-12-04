"""
Multi-file data loading with DDP support for ptychography datasets.

This module provides CombinedDataset, which orchestrates loading data from
multiple HDF5 files, handles train/val splitting, and distributes data across
multiple GPUs for Distributed Data Parallel (DDP) training.
"""
import numpy as np
import torch
from pathlib import Path

from data import PtychographyDataset


class CombinedDataset:
    """
    Handles multiple ptychography datasets with efficient file management.

    Key features:
    - Opens only one file at a time per process (on-demand mode)
    - Optional pre-loading: Load entire dataset to CPU RAM for faster training
    - Supports DDP by partitioning patterns across ranks
    - **Automatic DDP deadlock prevention**: Equalizes batch counts across ranks
    - Iterates file-by-file (on-demand) or from memory (pre-loaded)
    - Closes files immediately after processing (on-demand mode)
    - Can auto-discover paired files from a directory

    Args:
        file_paths: Either:
                   - List of paths to data files (*_dp.hdf5 or *_para.hdf5), OR
                   - Single directory path to scan for paired files
        train_split: Fraction of data for training (e.g., 0.9)
        batch_size: Number of samples per batch
        rank: Current process rank for DDP (0 if not using DDP)
        world_size: Total number of processes for DDP (1 if not using DDP)
        shuffle: Whether to shuffle file order and patterns within files
        random_seed: Random seed for reproducibility
        preload_to_memory: If True, load all data to CPU RAM at initialization.
                          Significantly faster but requires dataset to fit in memory.
                          Default: False (load on-demand from disk)
        **dataset_kwargs: Additional arguments passed to PtychographyDataset
    """

    @staticmethod
    def find_paired_files(directory):
        """
        Scan directory and find all objects that have paired *_dp.hdf5 and *_para.hdf5 files.

        IMPORTANT: Returns only ONE file per object (the _dp.hdf5 file).
        PtychographyDataset will automatically find and open the paired _para.hdf5 file.
        This avoids creating duplicate dataset instances for the same object.

        Args:
            directory: Path to directory to scan

        Returns:
            list: List of paths to _dp.hdf5 files ONLY (one per object, not both files)
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Find all _dp.hdf5 files
        dp_files = list(directory.glob('*_dp.hdf5'))

        # Verify each has a matching _para.hdf5 file, but only return the _dp file
        # PtychographyDataset will find the _para file automatically
        paired_files = []
        for dp_file in sorted(dp_files):
            # Extract object name
            object_name = dp_file.stem[:-3]  # Remove '_dp' suffix
            para_file = directory / f"{object_name}_para.hdf5"

            if para_file.exists():
                # Only append the _dp file, not both
                paired_files.append(dp_file)
            else:
                print(f"Warning: Skipping {dp_file.name} - no matching {para_file.name}", flush=True)

        if len(paired_files) == 0:
            raise ValueError(f"No paired HDF5 files found in {directory}")

        print(f"Found {len(paired_files)} paired dataset(s) in {directory}", flush=True)
        for f in paired_files:
            object_name = f.stem[:-3]
            print(f"  - {object_name}", flush=True)

        return paired_files

    def __init__(
        self,
        file_paths,
        train_split: float = 0.9,
        batch_size: int = 32,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        random_seed: int = 42,
        preload_to_memory: bool = False,
        **dataset_kwargs
    ):
        self.train_split = train_split
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.preload_to_memory = preload_to_memory
        self.dataset_kwargs = dataset_kwargs
        self.preloaded_data = None  # Will store pre-loaded data if enabled

        # Batch count limits (set by _equalize_batch_counts for DDP)
        self.train_batches_per_rank = None
        self.val_batches_per_rank = None

        # Only rank 0 processes file paths and creates metadata
        if rank == 0:
            # Auto-detect if file_paths is a directory or a list
            if isinstance(file_paths, (str, Path)):
                file_path = Path(file_paths)
                if file_path.is_dir():
                    # Scan directory for paired files
                    self.file_paths = self.find_paired_files(file_path)
                else:
                    # Single file provided as string
                    self.file_paths = [file_path]
            else:
                # List of files provided
                self.file_paths = [Path(f) for f in file_paths]

            # Load metadata from each file
            self.file_info = []
            total_patterns = 0

            for file_path in self.file_paths:
                # Temporarily create dataset to get metadata
                temp_dataset = PtychographyDataset(str(file_path), **dataset_kwargs)
                num_patterns = len(temp_dataset)
                temp_dataset.close()

                self.file_info.append({
                    'path': file_path,
                    'num_patterns': num_patterns,
                    'start_idx': total_patterns,
                    'end_idx': total_patterns + num_patterns
                })
                total_patterns += num_patterns

            self.total_patterns = total_patterns

            # Create global train/val split - THIS IS FIXED FOR THE ENTIRE TRAINING PROCESS
            # The split is created ONCE here and never changes across epochs
            self.train_size = int(total_patterns * train_split)
            self.val_size = total_patterns - self.train_size

            # Create global indices and optionally shuffle them ONCE
            self.rng = np.random.default_rng(random_seed)
            self.global_indices = np.arange(total_patterns)
            if shuffle:
                self.rng.shuffle(self.global_indices)

            # Split into train/val ONCE - these arrays are never modified
            # Only file order and patterns within files are shuffled per epoch
            self.train_indices = self.global_indices[:self.train_size].copy()  # Make immutable copy
            self.val_indices = self.global_indices[self.train_size:].copy()    # Make immutable copy

            print(f"CombinedDataset: {len(self.file_paths)} files, "
                  f"{total_patterns} total patterns, "
                  f"{self.train_size} train, {self.val_size} val", flush=True)

        # Broadcast metadata and indices from rank 0 to all other ranks
        if world_size > 1:
            import torch.distributed as dist

            # Broadcast file_paths as a list of strings
            if rank == 0:
                file_paths_str = [str(p) for p in self.file_paths]
                data_to_broadcast = [file_paths_str, self.file_info, self.total_patterns,
                                    self.train_size, self.val_size,
                                    self.train_indices, self.val_indices]
            else:
                data_to_broadcast = [None, None, None, None, None, None, None]

            # Broadcast using object list
            dist.broadcast_object_list(data_to_broadcast, src=0)

            # Unpack on non-rank-0 processes
            if rank != 0:
                file_paths_str, self.file_info, self.total_patterns, \
                self.train_size, self.val_size, \
                self.train_indices, self.val_indices = data_to_broadcast

                # Convert back to Path objects
                self.file_paths = [Path(p) for p in file_paths_str]

                print(f"Rank {rank}: Received CombinedDataset metadata: {len(self.file_paths)} files, "
                      f"{self.total_patterns} total patterns, "
                      f"{self.train_size} train, {self.val_size} val", flush=True)

        # Store original copies for verification (on all ranks)
        self._original_train_indices = self.train_indices.copy()
        self._original_val_indices = self.val_indices.copy()

        # Equalize batch counts across ranks for DDP safety
        if self.world_size > 1:
            self._equalize_batch_counts()

        # Pre-load all data into memory if requested
        if self.preload_to_memory:
            self._preload_all_data()

    def _equalize_batch_counts(self):
        """
        Equalize batch counts across all DDP ranks by dropping excess samples.

        This prevents DDP deadlocks caused by different ranks processing different
        numbers of batches. Truncates each rank's data to match the minimum batch
        count across all ranks.

        Called during initialization when world_size > 1.
        """
        import torch.distributed as dist

        if self.is_main_process:
            print("\n" + "=" * 50, flush=True)
            print("EQUALIZING BATCH COUNTS ACROSS RANKS", flush=True)
            print("=" * 50, flush=True)

        # Process both train and val splits
        for split_name in ['train', 'val']:
            if split_name == 'train':
                split_indices = self.train_indices
            else:
                split_indices = self.val_indices

            # Partition for this rank
            rank_indices = self._partition_for_rank(split_indices)

            # Calculate number of batches this rank would have
            num_batches_this_rank = len(rank_indices) // self.batch_size

            # Gather batch counts from all ranks
            batch_count_tensor = torch.tensor(num_batches_this_rank, dtype=torch.long)
            gathered_counts = [torch.zeros_like(batch_count_tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered_counts, batch_count_tensor)

            # Find minimum batch count across all ranks
            all_counts = [int(t.item()) for t in gathered_counts]
            min_batches = min(all_counts)
            max_batches = max(all_counts)

            if self.is_main_process:
                print(f"\n{split_name.upper()} split:", flush=True)
                print(f"  Batch counts per rank: {all_counts}", flush=True)
                print(f"  Min batches: {min_batches}, Max batches: {max_batches}", flush=True)

            # Truncate indices to match minimum batch count
            samples_to_keep = min_batches * self.batch_size
            original_count = len(rank_indices)

            # Truncate the rank's portion of the split indices
            # We need to truncate the indices AFTER partitioning
            _rank_indices_truncated = rank_indices[:samples_to_keep]

            # Calculate what indices to actually use from the global split
            # We need to update the original split indices to reflect the truncation
            # But we can't directly modify the split indices because they're shared
            # Instead, we'll store the truncated count and handle it in iteration

            if split_name == 'train':
                self.train_batches_per_rank = min_batches
                _samples_dropped = original_count - samples_to_keep
            else:
                self.val_batches_per_rank = min_batches
                _samples_dropped = original_count - samples_to_keep

            if self.is_main_process:
                print(f"  Equalizing to {min_batches} batches per rank", flush=True)
                if max_batches > min_batches:
                    total_dropped = sum([max(0, all_counts[i] - min_batches) * self.batch_size
                                        for i in range(self.world_size)])
                    print(f"  Total samples dropped across all ranks: {total_dropped}", flush=True)

        if self.is_main_process:
            print("=" * 50 + "\n", flush=True)

    def _preload_all_data(self):
        """
        Pre-load all data from all files into CPU memory.

        Creates a list of all samples indexed by global index.
        This significantly speeds up training when dataset fits in RAM.
        """
        if self.is_main_process:
            print("\n" + "=" * 50, flush=True)
            print("PRE-LOADING DATA TO MEMORY", flush=True)
            print("=" * 50, flush=True)

        # Initialize storage for all patterns
        self.preloaded_data = [None] * self.total_patterns

        # Load data from each file
        for file_idx, file_info in enumerate(self.file_info):
            if self.is_main_process:
                print(f"\nLoading file {file_idx + 1}/{len(self.file_info)}: {file_info['path'].name}", flush=True)

            # Create temporary dataset for this file
            dataset = PtychographyDataset(
                str(file_info['path']),
                **self.dataset_kwargs
            )

            # Load all patterns from this file
            num_patterns = file_info['num_patterns']
            for local_idx in range(num_patterns):
                global_idx = file_info['start_idx'] + local_idx

                # Load sample and store at global index
                self.preloaded_data[global_idx] = dataset[local_idx]

                # Progress indicator
                if self.is_main_process and (local_idx + 1) % max(1, num_patterns // 10) == 0:
                    progress = (local_idx + 1) / num_patterns * 100
                    print(f"  Progress: {local_idx + 1}/{num_patterns} ({progress:.1f}%)", flush=True)

            # Close file
            dataset.close()

            if self.is_main_process:
                print(f"  ✓ Loaded {num_patterns} patterns from {file_info['path'].name}", flush=True)

    def verify_split_integrity(self):
        """
        Verify that train/val split has not been modified.

        Returns:
            bool: True if split is unchanged, raises AssertionError otherwise
        """
        assert np.array_equal(self.train_indices, self._original_train_indices), \
            "Train indices have been modified! Split integrity violated."
        assert np.array_equal(self.val_indices, self._original_val_indices), \
            "Val indices have been modified! Split integrity violated."

        # Also verify no overlap between train and val
        assert len(np.intersect1d(self.train_indices, self.val_indices)) == 0, \
            "Train and val indices overlap! Split integrity violated."

        return True

    def _get_file_indices(self, global_indices, file_idx):
        """Get indices that belong to a specific file."""
        file_info = self.file_info[file_idx]
        # Find which global indices map to this file
        mask = (global_indices >= file_info['start_idx']) & (global_indices < file_info['end_idx'])
        file_global_indices = global_indices[mask]
        # Convert to local indices within the file
        local_indices = file_global_indices - file_info['start_idx']
        return local_indices

    def _partition_for_rank(self, indices):
        """Partition indices for current DDP rank."""
        # Each rank gets every world_size-th index
        return indices[self.rank::self.world_size]

    def print_rank_allocation(self, split='train', epoch=0):
        """
        Print how patterns are allocated to this rank from each file.
        Useful for debugging DDP data distribution.

        Args:
            split: 'train' or 'val'
            epoch: Current epoch number (for file order)
        """
        split_indices = self.train_indices if split == 'train' else self.val_indices
        rank_indices = self._partition_for_rank(split_indices)

        # Get file order for this epoch
        file_order = list(range(len(self.file_paths)))
        if self.shuffle:
            epoch_rng = np.random.default_rng(self.random_seed + epoch)
            epoch_rng.shuffle(file_order)

        print(f"\n[Rank {self.rank}] {split.upper()} data allocation (Epoch {epoch}):", flush=True)
        print(f"  Total patterns for this rank: {len(rank_indices)}", flush=True)
        print("  File allocation:", flush=True)

        total_allocated = 0
        for file_idx in file_order:
            local_indices = self._get_file_indices(rank_indices, file_idx)
            num_patterns = len(local_indices)
            if num_patterns > 0:
                num_batches = num_patterns // self.batch_size
                file_name = self.file_info[file_idx]['path'].name
                print(f"    {file_name}: {num_patterns} patterns ({num_batches} batches)", flush=True)
                total_allocated += num_patterns

        print(f"  Total allocated: {total_allocated} patterns\n", flush=True)

    def iterate_batches(self, split='train', epoch=0, debug=False):
        """
        Iterate through files and yield batches with file metadata.

        IMPORTANT: Train/val split is FIXED and never changes across epochs.
        What DOES change per epoch:
        - File processing order (shuffled)
        - Pattern order within each file (shuffled)

        Args:
            split: 'train' or 'val'
            epoch: Current epoch number (for shuffling)
            debug: If True, print allocation information (default: False)

        Yields:
            tuple: (batch, file_metadata) where:
                - batch: Tuple of (diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale)
                - file_metadata: Dict with file progress information
        """
        # If data is pre-loaded, use memory-based iteration
        if self.preload_to_memory and self.preloaded_data is not None:
            yield from self._iterate_batches_preloaded(split, epoch, debug)
            return

        # Print allocation info in debug mode
        if debug:
            self.print_rank_allocation(split, epoch)

        # Get FIXED indices for this split (never changes across epochs)
        split_indices = self.train_indices if split == 'train' else self.val_indices

        # Partition for this rank (creates a view, doesn't modify original)
        rank_indices = self._partition_for_rank(split_indices)

        # Get batch limit for this split (for DDP equalization)
        if split == 'train':
            max_batches = self.train_batches_per_rank
        else:
            max_batches = self.val_batches_per_rank

        # Shuffle file order each epoch (NOT the indices themselves)
        file_order = list(range(len(self.file_paths)))
        if self.shuffle:
            epoch_rng = np.random.default_rng(self.random_seed + epoch)
            epoch_rng.shuffle(file_order)

        # IMPORTANT: All ranks must iterate through ALL files in the same order
        # to avoid DDP deadlocks. Ranks with no data from a file will skip it.
        # We count files that have data separately for metadata purposes.
        file_counter = 0
        total_files_with_data = sum(
            1 for file_idx in file_order
            if len(self._get_file_indices(rank_indices, file_idx)) > 0
        )

        # Track total batches yielded (for DDP equalization)
        batches_yielded = 0

        # Iterate through ALL files in the same order (critical for DDP)
        for file_idx in file_order:
            # Get indices for this file that belong to current rank
            local_indices = self._get_file_indices(rank_indices, file_idx)

            # Skip files where this rank has no data, but continue to next file
            # (all ranks iterate same file_order, just skip different files)
            if len(local_indices) == 0:
                continue

            file_counter += 1

            # Shuffle patterns within file (LOCAL shuffle only - doesn't affect global split)
            # This shuffles which patterns from THIS FILE are processed first
            if self.shuffle:
                epoch_rng = np.random.default_rng(self.random_seed + epoch + file_idx)
                epoch_rng.shuffle(local_indices)

            # Open file and create dataset
            dataset = PtychographyDataset(
                str(self.file_info[file_idx]['path']),
                **self.dataset_kwargs
            )

            # Load all patterns for this rank from this file and create batches
            num_batches = len(local_indices) // self.batch_size
            total_patterns = len(local_indices)
            file_name = self.file_info[file_idx]['path'].name

            for batch_idx in range(num_batches):
                # Check if we've hit the batch limit (for DDP equalization)
                if max_batches is not None and batches_yielded >= max_batches:
                    # Close file and stop iterating
                    dataset.close()
                    return

                start = batch_idx * self.batch_size
                end = start + self.batch_size
                batch_local_indices = local_indices[start:end]

                # Load batch
                batch = []
                for idx in batch_local_indices:
                    batch.append(dataset[int(idx)])

                # Collate batch (stack tensors)
                if len(batch) > 0:
                    collated = self._collate(batch)

                    # Create file metadata
                    file_metadata = {
                        'file_num': file_counter,
                        'total_files': total_files_with_data,
                        'file_name': file_name,
                        'batch_in_file': batch_idx + 1,
                        'total_batches_in_file': num_batches,
                        'patterns_processed': end,
                        'total_patterns': total_patterns
                    }

                    batches_yielded += 1
                    yield collated, file_metadata

            # Close file immediately after processing
            dataset.close()

    def _iterate_batches_preloaded(self, split='train', epoch=0, debug=False):
        """
        Iterate through pre-loaded data and yield batches.

        This is a simplified version of iterate_batches that serves data from memory.
        Much faster than file-based iteration but requires data to fit in RAM.

        Args:
            split: 'train' or 'val'
            epoch: Current epoch number (for shuffling)
            debug: If True, print allocation information (default: False)

        Yields:
            tuple: (batch, file_metadata) where batches are served from pre-loaded memory
        """
        # Print allocation info in debug mode
        if debug:
            self.print_rank_allocation(split, epoch)

        # Get FIXED indices for this split
        split_indices = self.train_indices if split == 'train' else self.val_indices

        # Partition for this rank
        rank_indices = self._partition_for_rank(split_indices)

        # Get batch limit for this split (for DDP equalization)
        if split == 'train':
            max_batches = self.train_batches_per_rank
        else:
            max_batches = self.val_batches_per_rank

        # Shuffle indices for this epoch (NOT the split itself)
        if self.shuffle:
            epoch_rng = np.random.default_rng(self.random_seed + epoch)
            rank_indices = rank_indices.copy()  # Don't modify original
            epoch_rng.shuffle(rank_indices)

        # Calculate number of batches (respecting the equalized limit)
        natural_num_batches = len(rank_indices) // self.batch_size
        if max_batches is not None:
            num_batches = min(natural_num_batches, max_batches)
        else:
            num_batches = natural_num_batches
        total_patterns = num_batches * self.batch_size  # Only count patterns actually used

        if self.is_main_process:
            print(f"\n  Serving {total_patterns} patterns from pre-loaded memory ({num_batches} batches)", flush=True)

        # Iterate through batches
        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_global_indices = rank_indices[start:end]

            # Collect batch from pre-loaded data
            batch = []
            for global_idx in batch_global_indices:
                batch.append(self.preloaded_data[global_idx])

            # Collate batch
            if len(batch) > 0:
                collated = self._collate(batch)

                # Create simplified metadata
                file_metadata = {
                    'file_num': 1,
                    'total_files': 1,
                    'file_name': 'preloaded_memory',
                    'batch_in_file': batch_idx + 1,
                    'total_batches_in_file': num_batches,
                    'patterns_processed': end,
                    'total_patterns': total_patterns
                }

                yield collated, file_metadata

    def _collate(self, batch):
        """Collate list of samples into batched tensors."""
        diff_amps = torch.stack([item[0] for item in batch])
        amp_patches = torch.stack([item[1] for item in batch])
        ph_patches = torch.stack([item[2] for item in batch])
        probes = torch.stack([item[3] for item in batch])
        probe_positions = torch.stack([item[4] for item in batch])
        normalizations = torch.tensor([item[5] for item in batch])
        scales = torch.tensor([item[6] for item in batch])

        return diff_amps, amp_patches, ph_patches, probes, probe_positions, normalizations, scales

    def get_num_batches(self, split='train'):
        """Get total number of batches for this rank in a split."""
        # If batch counts have been equalized, return the equalized count
        if split == 'train' and self.train_batches_per_rank is not None:
            return self.train_batches_per_rank
        elif split == 'val' and self.val_batches_per_rank is not None:
            return self.val_batches_per_rank

        # Otherwise calculate naturally
        split_indices = self.train_indices if split == 'train' else self.val_indices
        rank_indices = self._partition_for_rank(split_indices)
        return len(rank_indices) // self.batch_size
