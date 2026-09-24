"""Global pytest configuration for fast, predictable CPU execution."""

import importlib
import os

# The test suite operates on small tensors, where large thread pools cost far
# more to launch and synchronize than the underlying operations. Set these
# before test modules import NumPy, SciPy, or PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

torch = importlib.import_module("torch")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
