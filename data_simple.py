import bisect
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from scipy.ndimage import zoom

from utils.ptychi_utils import extract_patches_fourier_shift


class PtychographyDatasetSimple(Dataset):
    """Many HDF5 pairs, one global index; open files per sample (no handle cache)."""

    @staticmethod
    def find_paired_files(directory: Path) -> list:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        out = []
        for dp in sorted(directory.rglob("*_dp.hdf5")):
            para = dp.with_name(f"{dp.stem[:-3]}_para{dp.suffix}")
            if para.is_file():
                out.append(dp)
        if not out:
            raise ValueError(f"No paired HDF5 files found in {directory}")
        print(f"PtychographyDatasetSimple: {len(out)} paired file(s) under {directory}", flush=True)
        return out

    @staticmethod
    def derive_object_name(file_path: Path, base_dir: Path) -> str:
        fp, bd = file_path.resolve(), Path(base_dir).resolve()
        if not fp.is_relative_to(bd):
            stem = file_path.stem
            return stem[:-3] if stem.endswith("_dp") else stem
        rel = fp.relative_to(bd).with_suffix("")
        name = rel.name
        if name.endswith("_dp"):
            rel = rel.with_name(name[:-3])
        return rel.as_posix()

    def __init__(
        self,
        file_paths,
        rank: int = 0,
        world_size: int = 1,
        debug: bool = False,
        max_files: Optional[int] = None,
        scale: float = 100000.0,
        normalization_dict_path: Optional[str] = None,
        default_normalization: float = 100000.0,
        apply_noise: bool = True,
        max_probe_modes: int = 8,
        target_size: Optional[int] = 256,
    ):
        _ = world_size, debug

        self.data_dir = Path(file_paths)
        if not self.data_dir.is_dir():
            raise ValueError(f"file_paths must be a directory, got: {file_paths}")

        self.file_paths = self.find_paired_files(self.data_dir)
        if max_files is not None:
            self.file_paths = self.file_paths[:max_files]

        self.scale = scale
        self.apply_noise = apply_noise
        self.max_probe_modes = max_probe_modes
        self.target_size = target_size
        self.fake_data = None

        norm_map: Optional[dict] = None
        if normalization_dict_path:
            with open(normalization_dict_path, "rb") as f:
                norm_map = pickle.load(f)
            if not isinstance(norm_map, dict):
                raise ValueError("normalization file must contain a dict")

        self._norm_by_path: Dict[Path, float] = {}
        for fp in self.file_paths:
            key = self.derive_object_name(fp, self.data_dir)
            if norm_map is None:
                self._norm_by_path[fp] = float(default_normalization)
            else:
                self._norm_by_path[fp] = float(norm_map.get(key, default_normalization))

        csv_counts: Dict[Path, int] = {}
        idx_csv = self.data_dir / "index.csv"
        if idx_csv.is_file():
            for _, row in pd.read_csv(idx_csv).iterrows():
                rel = Path(row["dp_path"])
                if rel.is_absolute():
                    raise ValueError(f"index.csv dp_path must be relative: {row['dp_path']}")
                csv_counts[(self.data_dir / rel).resolve()] = int(row["n_dps"])

        self.file_offsets = [0]
        total = 0
        for fp in self.file_paths:
            n = csv_counts.get(fp.resolve())
            if n is None:
                with h5py.File(fp, "r", libver="latest", swmr=True) as f:
                    n = int(f["dp"].shape[0])
            total += n
            self.file_offsets.append(total)

        self._len = total
        print(
            f"[Rank {rank}] PtychographyDatasetSimple: {len(self.file_paths)} files, {total} patterns",
            flush=True,
        )

    def close(self) -> None:
        pass

    def __len__(self) -> int:
        return self._len

    @staticmethod
    def _state_for_pair(dp_path: Path, para_path: Path, normalization: float) -> Dict[str, Any]:
        return {
            "dp_path": dp_path,
            "para_path": para_path,
            "normalization": normalization,
            "n_dp": None,
            "raw_shape": None,
            "object_shape": None,
            "pos_in_px": None,
            "pos_ps": None,
            "pos_origin": None,
        }

    def _layout(self, s: Dict[str, Any], d: h5py.File, p: h5py.File) -> None:
        if s["n_dp"] is not None:
            return
        if "dp" not in d:
            raise KeyError(f"no 'dp' in {s['dp_path'].name}")
        s["n_dp"] = int(d["dp"].shape[0])
        s["raw_shape"] = d["dp"].shape[1:]
        for k in ("object", "probe", "probe_position_x_m", "probe_position_y_m"):
            if k not in p:
                raise KeyError(f"{s['para_path'].name} missing {k}")
        s["object_shape"] = tuple(int(x) for x in p["object"].shape[1:])

    def _init_pos_rules(self, s: Dict[str, Any], d: h5py.File, p: h5py.File) -> None:
        if s["pos_in_px"] is not None:
            return
        self._layout(s, d, p)
        py = p["probe_position_y_m"][...]
        px = p["probe_position_x_m"][...]
        if py.shape[0] != s["n_dp"]:
            raise ValueError(f"positions {py.shape[0]} != dp {s['n_dp']}")
        oh, ow = s["object_shape"][0], s["object_shape"][1]
        ry, rx = float(py.max() - py.min()), float(px.max() - px.min())
        in_px = 0.1 * oh < ry < 10 * oh and 0.1 * ow < rx < 10 * ow
        ps = float(p["object"].attrs["pixel_height_m"])
        origin = ((np.array(s["object_shape"], dtype=np.float32) / 2.0).round() + 0.5).astype(np.float32)
        s["pos_in_px"], s["pos_ps"], s["pos_origin"] = in_px, ps, origin

    def _probe_xy(self, s: Dict[str, Any], d: h5py.File, p: h5py.File, i: int) -> Tensor:
        self._init_pos_rules(s, d, p)
        py, px = float(p["probe_position_y_m"][i]), float(p["probe_position_x_m"][i])
        v = np.array([py, px], dtype=np.float32)
        if not s["pos_in_px"]:
            v = v / s["pos_ps"]
        return torch.from_numpy(v + s["pos_origin"])

    def _pad_probe(self, probe: np.ndarray) -> np.ndarray:
        t = self.max_probe_modes
        if probe.shape[1] >= t:
            return probe
        z = np.zeros((probe.shape[0], t - probe.shape[1], probe.shape[2], probe.shape[3]), dtype=probe.dtype)
        return np.concatenate([probe, z], axis=1)

    def _probe_first_slice(self, probe: np.ndarray) -> np.ndarray:
        if probe.ndim != 4:
            raise ValueError(f"probe must be 4D, got {probe.shape}")
        return probe if probe.shape[0] == 1 else probe[:1]

    def _zero_pad(self, image: np.ndarray, size: int) -> np.ndarray:
        h, w = image.shape
        if h == size and w == size:
            return image
        if h > size or w > size:
            raise ValueError(f"({h},{w}) larger than target {size}")
        a, b = size - h, size - w
        pt, pb = a // 2, a - a // 2
        pl, pr = b // 2, b - b // 2
        return np.pad(image, ((pt, pb), (pl, pr)), mode="constant")

    def _upsample_probe(self, probe: np.ndarray, size: int) -> np.ndarray:
        _, m, h, w = probe.shape
        if h == size and w == size:
            return probe
        zh, zw = size / h, size / w
        out_r = np.zeros((1, m, size, size), np.float64)
        out_i = np.zeros((1, m, size, size), np.float64)
        for i in range(m):
            u = probe[0, i]
            out_r[0, i] = zoom(u.real, (zh, zw), order=1)
            out_i[0, i] = zoom(u.imag, (zh, zw), order=1)
        return (out_r + 1j * out_i).astype(probe.dtype)

    def _patch(self, s: Dict[str, Any], obj: np.ndarray, xy: Tensor) -> Tensor:
        rh, rw = s["raw_shape"]
        return extract_patches_fourier_shift(torch.from_numpy(obj), xy.unsqueeze(0), (rh, rw))[0]

    def _sample(self, s: Dict[str, Any], d: h5py.File, p: h5py.File, i: int):
        img = d["dp"][i]
        img = (img / s["normalization"]) * self.scale
        if self.apply_noise:
            img = np.random.default_rng().poisson(img)
        amp = np.sqrt(np.float32(img))
        ts = self.target_size
        if ts is not None and amp.shape[0] != ts:
            amp = self._zero_pad(amp, ts)

        xy = self._probe_xy(s, d, p, i)

        probe = p["probe"][...]
        probe = self._probe_first_slice(probe)
        probe = self._pad_probe(probe)
        if ts is not None:
            probe = self._upsample_probe(probe, ts)

        obj = p["object"][0]
        patch = self._patch(s, obj, xy)
        ap, ph = torch.abs(patch), torch.angle(patch)
        if ts is not None and ap.shape[0] != ts:
            ap = torch.from_numpy(self._zero_pad(ap.detach().cpu().numpy(), ts))
            ph = torch.from_numpy(self._zero_pad(ph.detach().cpu().numpy(), ts))
        return amp, ap, ph, probe, xy

    def __getitem__(self, idx: int):
        if self.fake_data is not None:
            return self.fake_data
        if not isinstance(idx, int):
            idx = int(idx)
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)

        fi = bisect.bisect_right(self.file_offsets, idx) - 1
        local = idx - self.file_offsets[fi]
        dp_path = self.file_paths[fi]
        stem = dp_path.stem[:-3] if dp_path.stem.endswith("_dp") else dp_path.stem
        para_path = dp_path.parent / f"{stem}_para{dp_path.suffix}"
        if not para_path.is_file():
            raise FileNotFoundError(para_path)
        norm = self._norm_by_path[dp_path]
        s = self._state_for_pair(dp_path, para_path, norm)
        with h5py.File(dp_path, "r", libver="latest", swmr=True) as d, h5py.File(
            para_path, "r", libver="latest", swmr=True
        ) as p:
            amp, ap, ph, probe, xy = self._sample(s, d, p, local)

        amp = torch.from_numpy(amp)
        probe = torch.from_numpy(probe)

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

        return amp, ap, ph, probe, xy, s["normalization"], self.scale