import torch
from torch.utils.data import Dataset


class DummyPtychographyDataset(Dataset):
    """Synthetic zero-valued dataset for training throughput benchmarks."""

    def __init__(
        self,
        length: int,
        image_size: int = 256,
        normalization: float = 100000.0,
        scale: float = 10000.0,
    ):
        if length <= 0:
            raise ValueError("length must be positive")
        self.length = int(length)
        self.image_size = int(image_size)
        self.normalization = float(normalization)
        self.scale = float(scale)

        image_shape = (1, self.image_size, self.image_size)
        probe_shape = (1, 1, self.image_size, self.image_size)
        self._input = torch.zeros(image_shape, dtype=torch.float32)
        self._label = torch.zeros(image_shape, dtype=torch.float32)
        self._probe = torch.zeros(probe_shape, dtype=torch.complex64)
        self._probe_position = torch.zeros(2, dtype=torch.float32)
        self._normalization = torch.tensor(self.normalization, dtype=torch.float32)
        self._scale = torch.tensor(self.scale, dtype=torch.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)

        return (
            self._input.clone(),
            self._label.clone(),
            self._label.clone(),
            self._probe.clone(),
            self._probe_position.clone(),
            self._normalization.clone(),
            self._scale.clone(),
        )
