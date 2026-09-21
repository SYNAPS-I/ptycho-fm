"""Dataset over `data_pack.py` shards: one global pattern index → one packed HDF5, one read."""

from __future__ import annotations

import bisect
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from ptychi.image_proc import extract_patches_fourier_shift
from torch import Tensor
from torch.utils.data import Dataset


def _pad_probe(probe: np.ndarray, max_modes: int) -> np.ndarray:
    t = max_modes
    if probe.shape[1] >= t:
        return probe
    z = np.zeros((probe.shape[0], t - probe.shape[1], probe.shape[2], probe.shape[3]), dtype=probe.dtype)
    return np.concatenate([probe, z], axis=1)


class PtychographyDatasetPacked(Dataset):
    """Globally indexed patterns over sorted ``packed_*.hdf5`` from ``data_pack.py``.

    Keeps one HDF5 handle per shard file that has been touched; opens on demand.
    Call :meth:`close` when the dataset is no longer needed to release file descriptors.
    Returns native spatial dimensions for diffraction patterns, object patches,
    and probes; only the probe mode axis is padded.
    """

    @staticmethod
    def find_packed_shards(directory: Path) -> list[Path]:
        directory = Path(directory)
        paths = sorted(directory.glob("packed_*.hdf5"))
        if not paths:
            raise ValueError(f"No packed_*.hdf5 under {directory}")
        return paths

    def __init__(
        self,
        pack_dir,
        rank: int = 0,
        world_size: int = 1,
        debug: bool = False,
        max_shards: int | None = None,
        scale: float = 100000.0,
        normalization_dict_path: str | None = None,
        default_normalization: float = 100000.0,
        apply_noise: bool = True,
        deterministic_noise: bool = False,
        noise_seed: int = 0,
        max_probe_modes: int = 8,
    ):
        _ = world_size, debug
        self.pack_dir = Path(pack_dir)
        if not self.pack_dir.is_dir():
            raise ValueError(f"pack_dir must be a directory: {pack_dir}")

        self.shard_paths = self.find_packed_shards(self.pack_dir)
        if max_shards is not None:
            self.shard_paths = self.shard_paths[: max_shards]

        self.scale = scale
        self.apply_noise = apply_noise
        self.deterministic_noise = deterministic_noise
        self.noise_seed = int(noise_seed)
        if self.noise_seed < 0:
            raise ValueError("noise_seed must be non-negative")
        self.max_probe_modes = max_probe_modes
        self.default_normalization = default_normalization
        self.fake_data = None

        self._open_shards: dict[Path, h5py.File] = {}

        self._norm_override: dict | None = None
        if normalization_dict_path:
            with open(normalization_dict_path, "rb") as f:
                m = pickle.load(f)
            if not isinstance(m, dict):
                raise ValueError("normalization file must be a dict")
            self._norm_override = m

        # shard_offsets[k] = pattern index where shard k starts; length n_shards+1
        self.shard_offsets: list[int] = [0]
        self._obj_starts: list[np.ndarray] = []
        for p in self.shard_paths:
            with h5py.File(p, "r", libver="latest", swmr=True) as f:
                n_dp = f["n_dp"][:].astype(np.int64)
            starts = np.concatenate([[0], np.cumsum(n_dp[:-1])])
            self._obj_starts.append(starts)
            self.shard_offsets.append(int(self.shard_offsets[-1] + int(np.sum(n_dp))))

        self._len = self.shard_offsets[-1]
        print(
            f"[Rank {rank}] PtychographyDatasetPacked: {len(self.shard_paths)} shard(s), {self._len} patterns",
            flush=True,
        )

    def _shard_file(self, path: Path) -> h5py.File:
        path = path.resolve()
        if path not in self._open_shards:
            self._open_shards[path] = h5py.File(path, "r", libver="latest", swmr=True)
        return self._open_shards[path]

    def close(self) -> None:
        for fh in self._open_shards.values():
            fh.close()
        self._open_shards.clear()

    def __len__(self) -> int:
        return self._len

    def _norm_for_slot(self, f: h5py.File, o: int, key: str | None) -> float:
        v = float(f["normalization"][o])
        if self._norm_override is not None and key is not None:
            v = float(self._norm_override.get(key, v if not np.isnan(v) else self.default_normalization))
        if np.isnan(v):
            v = self.default_normalization
        return v

    @staticmethod
    def _probe_xy_from_packed(
        py_full: np.ndarray,
        px_full: np.ndarray,
        i: int,
        n_dp: int,
        object_shape: tuple,
        pixel_height_m: float,
    ) -> Tensor:
        oh, ow = int(object_shape[0]), int(object_shape[1])
        py, px = py_full[:n_dp], px_full[:n_dp]
        if py.shape[0] != n_dp:
            raise ValueError("position length mismatch")
        ry, rx = float(py.max() - py.min()), float(px.max() - px.min())
        in_px = 0.1 * oh < ry < 10 * oh and 0.1 * ow < rx < 10 * ow
        ps = float(pixel_height_m)
        origin = ((np.array(object_shape, dtype=np.float32) / 2.0).round() + 0.5).astype(np.float32)
        py_i, px_i = float(py_full[i]), float(px_full[i])
        v = np.array([py_i, px_i], dtype=np.float32)
        if not in_px:
            v = v / ps
        return torch.from_numpy(v + origin)

    def __getitem__(self, idx: int):
        if self.fake_data is not None:
            return self.fake_data
        idx = int(idx)
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)

        si = bisect.bisect_right(self.shard_offsets, idx) - 1
        local = idx - self.shard_offsets[si]
        starts = self._obj_starts[si]
        o = bisect.bisect_right(starts, local) - 1
        pi = local - int(starts[o])

        path = self.shard_paths[si]
        f = self._shard_file(path)
        n_dp_o = int(f["n_dp"][o])
        if pi >= n_dp_o:
            raise IndexError(f"pattern {pi} >= n_dp {n_dp_o} for slot {o} in {path.name}")

        obj_key = None
        if "object_key" in f:
            raw = f["object_key"][o]
            obj_key = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        norm = self._norm_for_slot(f, o, obj_key)

        img = np.asarray(f["dp"][o, pi], dtype=np.float32)
        img = (img / norm) * self.scale
        if self.apply_noise:
            seed = self.noise_seed + idx if self.deterministic_noise else None
            img = np.random.default_rng(seed).poisson(img).astype(np.float32)
        amp = np.sqrt(np.float32(img))

        osh = tuple(int(x) for x in f["object"].shape[2:])
        py_a = np.asarray(f["probe_position_y_m"][o, :n_dp_o])
        px_a = np.asarray(f["probe_position_x_m"][o, :n_dp_o])
        pix = float(f["pixel_height_m"][o])
        xy = self._probe_xy_from_packed(py_a, px_a, pi, n_dp_o, osh, pix)

        obj = np.asarray(f["object"][o, 0])
        rh, rw = img.shape
        patch = extract_patches_fourier_shift(torch.from_numpy(obj), xy.unsqueeze(0), (rh, rw))[0]
        ap, ph = torch.abs(patch), torch.angle(patch)

        probe = np.asarray(f["probe"][o, 0])
        probe = probe[np.newaxis, ...]
        probe = _pad_probe(probe, self.max_probe_modes)

        amp = torch.from_numpy(np.asarray(amp))
        probe = torch.from_numpy(np.asarray(probe, dtype=np.complex64))

        if amp.dim() == 2:
            amp = amp.unsqueeze(0)
        if ap.dim() == 2:
            ap = ap.unsqueeze(0)
        if ph.dim() == 2:
            ph = ph.unsqueeze(0)
        if probe.dim() == 2:
            probe = probe.unsqueeze(0).unsqueeze(0)
        elif probe.dim() == 3:
            probe = probe.unsqueeze(0)

        # All real DataLoader fields are float32; the probe stays complex64.
        return (amp.float(), ap.float(), ph.float(), probe, xy.float(),
                torch.tensor(norm, dtype=torch.float32),
                torch.tensor(self.scale, dtype=torch.float32))
