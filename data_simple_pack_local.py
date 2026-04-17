"""Node-local variant of PtychographyDatasetPacked.

Reads from an explicit list of shard paths (typically staged onto node-local
tmpfs by ``scripts/stage_pack_to_local.py``), rather than globbing a directory.
All tensor decoding logic is inherited from PtychographyDatasetPacked.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from data_simple_pack import PtychographyDatasetPacked


class PtychographyDatasetPackedLocal(PtychographyDatasetPacked):

    def __init__(
        self,
        shard_paths,
        rank: int = 0,
        world_size: int = 1,
        debug: bool = False,
        scale: float = 100000.0,
        normalization_dict_path: Optional[str] = None,
        default_normalization: float = 100000.0,
        apply_noise: bool = True,
        max_probe_modes: int = 8,
        target_size: Optional[int] = 256,
    ):
        _ = world_size, debug

        shard_paths = [Path(p) for p in shard_paths]
        if not shard_paths:
            raise ValueError("shard_paths is empty")
        for p in shard_paths:
            if not p.is_file():
                raise ValueError(f"shard not found: {p}")

        self.pack_dir = shard_paths[0].parent
        self.shard_paths = shard_paths

        self.scale = scale
        self.apply_noise = apply_noise
        self.max_probe_modes = max_probe_modes
        self.target_size = target_size
        self.default_normalization = default_normalization
        self.fake_data = None

        self._open_shards: Dict[Path, h5py.File] = {}

        self._norm_override: Optional[dict] = None
        if normalization_dict_path:
            with open(normalization_dict_path, "rb") as f:
                m = pickle.load(f)
            if not isinstance(m, dict):
                raise ValueError("normalization file must be a dict")
            self._norm_override = m

        self.shard_offsets: List[int] = [0]
        self._obj_starts: List[np.ndarray] = []
        for p in self.shard_paths:
            with h5py.File(p, "r", libver="latest", swmr=True) as f:
                n_dp = f["n_dp"][:].astype(np.int64)
            starts = np.concatenate([[0], np.cumsum(n_dp[:-1])])
            self._obj_starts.append(starts)
            self.shard_offsets.append(int(self.shard_offsets[-1] + int(np.sum(n_dp))))

        self._len = self.shard_offsets[-1]
        print(
            f"[Rank {rank}] PtychographyDatasetPackedLocal: "
            f"{len(self.shard_paths)} shard(s) @ {self.pack_dir}, {self._len} patterns",
            flush=True,
        )
