#!/usr/bin/env python3
"""
Print basic probe-position and pixel-size diagnostics for a paired *_dp / *_para dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def infer_para_path(dp_path: Path) -> Path:
    if dp_path.name.endswith("_dp.hdf5"):
        return dp_path.with_name(dp_path.name.replace("_dp.hdf5", "_para.hdf5"))
    if dp_path.name.endswith("_dp.h5"):
        return dp_path.with_name(dp_path.name.replace("_dp.h5", "_para.h5"))
    raise ValueError(f"Expected dp file name to end with '_dp.hdf5' or '_dp.h5', got: {dp_path.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dp_file", type=Path, help="Path to *_dp.hdf5")
    parser.add_argument("--para-file", type=Path, default=None, help="Override path to *_para.hdf5")
    args = parser.parse_args(argv)

    dp_file: Path = args.dp_file
    para_file: Path = args.para_file or infer_para_path(dp_file)

    print(f"DP file:   {dp_file}")
    print(f"Para file: {para_file}")

    with h5py.File(dp_file, "r") as dp_h5:
        print(f"\nDP file keys: {list(dp_h5.keys())}")
        print(f"DP shape: {dp_h5['dp'].shape}")
        raw_pattern_shape = dp_h5["dp"].shape[1:]
        print(f"Raw pattern shape: {raw_pattern_shape}")

    with h5py.File(para_file, "r") as para_h5:
        print(f"\nPara file keys: {list(para_h5.keys())}")

        obj = para_h5["object"]
        object_shape = obj[0].shape
        print(f"Object shape: {obj.shape} (per-slice: {object_shape})")
        print(f"Object attrs: {dict(obj.attrs)}")

        pixel_size_m = obj.attrs.get("pixel_height_m", None)
        print(f"\nPixel size (m): {pixel_size_m}")

        if "probe_position_indexes" in para_h5:
            pos_idx = para_h5["probe_position_indexes"][...]
            print(f"\nProbe position indexes shape: {pos_idx.shape}")
            print(f"Probe position indexes: min={pos_idx.min()}, max={pos_idx.max()}")
            if pos_idx.ndim == 2 and pos_idx.shape[1] >= 2:
                print(f"  Column 0: min={pos_idx[:, 0].min():.2f}, max={pos_idx[:, 0].max():.2f}")
                print(f"  Column 1: min={pos_idx[:, 1].min():.2f}, max={pos_idx[:, 1].max():.2f}")

        pos_x = para_h5["probe_position_x_m"][...]
        pos_y = para_h5["probe_position_y_m"][...]
        print(f"\nProbe positions X (labeled _m): min={pos_x.min():.6e}, max={pos_x.max():.6e}")
        print(f"Probe positions Y (labeled _m): min={pos_y.min():.6e}, max={pos_y.max():.6e}")

        pos_range_x = float(pos_x.max() - pos_x.min())
        pos_range_y = float(pos_y.max() - pos_y.min())
        print(f"\nPosition ranges (raw): X={pos_range_x:.2f}, Y={pos_range_y:.2f}")
        print(f"Object size: {object_shape[1]}x{object_shape[0]} (WxH)")
        print(f"If positions are pixels, X range ~{pos_range_x:.0f} vs object width {object_shape[1]}")
        print(f"If positions are pixels, Y range ~{pos_range_y:.0f} vs object height {object_shape[0]}")

        print("\n--- Test: treat positions as pixels directly ---")
        pos_origin = np.array(object_shape, dtype=np.float64) / 2.0
        pos_origin = np.round(pos_origin) + 0.5
        print(f"Position origin (object center): {pos_origin}")

        pos_x_final = pos_x + pos_origin[1]
        pos_y_final = pos_y + pos_origin[0]
        print(f"Final positions X (px): min={pos_x_final.min():.2f}, max={pos_x_final.max():.2f}")
        print(f"Final positions Y (px): min={pos_y_final.min():.2f}, max={pos_y_final.max():.2f}")

        patch_size = raw_pattern_shape[0]
        sy_min = pos_y_final.min() - (patch_size - 1) / 2
        sy_max = pos_y_final.max() + (patch_size + 1) / 2
        sx_min = pos_x_final.min() - (patch_size - 1) / 2
        sx_max = pos_x_final.max() + (patch_size + 1) / 2

        pad_top = max(-int(sy_min), 0)
        pad_bottom = max(int(sy_max) - object_shape[0], 0)
        pad_left = max(-int(sx_min), 0)
        pad_right = max(int(sx_max) - object_shape[1], 0)

        print(
            "Required padding if positions are pixels: "
            f"top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
