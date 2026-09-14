"""
CUDA Prefetcher for DataLoader

Wraps a PyTorch DataLoader to enable asynchronous data transfer to GPU.
Preloads the next batch while the current batch is being processed,
hiding IO latency and improving GPU utilization.

Usage:
    train_loader = DataLoader(dataset, ...)
    train_prefetcher = CUDAPrefetcher(train_loader, device)
    
    for batch in train_prefetcher:
        # batch is already on GPU
        process(batch)
"""

import torch


class CUDAPrefetcher:
    """
    Prefetcher that asynchronously moves data to GPU using CUDA streams.
    
    Wraps a DataLoader and preloads the next batch while the current batch
    is being processed, hiding data transfer latency.
    
    Args:
        loader: PyTorch DataLoader to wrap
        device: CUDA device to transfer data to
    """
    
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None
        
    def preload(self):
        """Preload next batch asynchronously using CUDA stream."""
        try:
            self.next_batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return
        
        # Use CUDA stream for async transfer
        with torch.cuda.stream(self.stream):
            self.next_batch = self._move_to_device(self.next_batch)
    
    def _move_to_device(self, batch):
        """
        Recursively move batch to device.
        
        Handles:
        - Tensors
        - Lists/tuples of tensors
        - Nested structures
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            # Recursively handle lists/tuples
            moved = [self._move_to_device(item) for item in batch]
            # Preserve original type (list or tuple)
            return type(batch)(moved)
        elif isinstance(batch, dict):
            # Handle dictionaries
            return {key: self._move_to_device(value) for key, value in batch.items()}
        else:
            # Non-tensor items (e.g., strings, numbers) - return as-is
            return batch
    
    def __iter__(self):
        """Initialize iterator and preload first batch."""
        self.loader_iter = iter(self.loader)
        self.preload()
        return self
    
    def __next__(self):
        """Return current batch and preload next batch."""
        # Wait for current batch to be ready
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        
        # Preload next batch while current batch is being processed
        self.preload()
        
        return batch
    
    def __len__(self):
        """Return length of wrapped loader."""
        return len(self.loader)