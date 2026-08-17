#!/usr/bin/env python
"""
Visualize one (<name>_dp.hdf5, <name>_para.hdf5) pair.

Produces a multi-panel PNG showing, for a single object:
  1. the full complex object (amplitude + phase) with the scan positions on top
  2. every probe mode (amplitude + phase) and how much power each mode carries
  3. a few measured diffraction patterns (log scale)
  4. the physics check: object patch -> |FFT(probe x patch)|^2 next to the
     measured pattern at the same scan position

Usage
-----
    python viz_object.py /flare/SYNAPS-I/aileen/test_data/scan807_dp.hdf5
    python viz_object.py .../scan807_dp.hdf5 --indices 0 500 2000 --out scan807.png

Either file of the pair can be given; the partner is found automatically.
Reads only what it needs from the big _dp file (a handful of frames).
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_pair(path: Path):
    """Return (dp_file, para_file) given either half of the pair."""
    stem = path.stem
    if stem.endswith("_dp"):
        base = stem[:-3]
    elif stem.endswith("_para"):
        base = stem[:-5]
    else:
        raise SystemExit(f"{path.name} is not a *_dp.hdf5 / *_para.hdf5 file")
    dp = path.with_name(f"{base}_dp.hdf5")
    para = path.with_name(f"{base}_para.hdf5")
    for f in (dp, para):
        if not f.exists():
            raise SystemExit(f"missing partner file: {f}")
    return base, dp, para


def crop_patch(obj, cy, cx, size):
    """Integer-pixel crop of `obj` centred on (cy, cx). Zero-padded at edges.

    The dataloader uses a sub-pixel Fourier shift; this is the nearest-integer
    version, which is fine for looking at things.
    """
    h = size // 2
    out = np.zeros((size, size), dtype=obj.dtype)
    y0, x0 = int(round(cy)) - h, int(round(cx)) - h
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + size, obj.shape[0]), min(x0 + size, obj.shape[1])
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = obj[ys:ye, xs:xe]
    return out


def simulate(patch, probe):
    """Forward model used by the model: sum over modes of |FFT(probe*patch)|^2."""
    psi = np.fft.fftshift(np.fft.fft2(probe * patch[None, ...]), axes=(-2, -1))
    return (np.abs(psi) ** 2).sum(axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path, help="path to *_dp.hdf5 or *_para.hdf5")
    ap.add_argument("--indices", type=int, nargs="*", default=None,
                    help="scan indices to show (default: 3 spread across the scan)")
    ap.add_argument("--max-modes", type=int, default=8,
                    help="cap on how many probe modes to draw (default 8)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG (default: <name>_viz.png next to the data)")
    args = ap.parse_args()

    name, dp_file, para_file = resolve_pair(args.file)
    out = args.out or dp_file.with_name(f"{name}_viz.png")

    # ---------------- read metadata + small arrays ----------------
    with h5py.File(para_file, "r") as f:
        obj = f["object"][0]                      # (H, W) complex
        probe = f["probe"][0]                     # (n_modes, 256, 256) complex
        pixel_size = float(f["object"].attrs["pixel_height_m"])
        pos_y = f["probe_position_y_m"][...]
        pos_x = f["probe_position_x_m"][...]
        para_keys = list(f.keys())

    with h5py.File(dp_file, "r") as f:
        dp_key = "dp" if "dp" in f else next(iter(f.keys()))
        n_patterns, dp_h, dp_w = f[dp_key].shape
        if args.indices:
            idxs = [i for i in args.indices if 0 <= i < n_patterns]
        else:
            idxs = list(np.linspace(0, n_patterns - 1, 3).astype(int))
        frames = np.stack([f[dp_key][i] for i in idxs]).astype(np.float32)

    n_modes = probe.shape[0]
    # positions in pixels, in the same convention the dataloader uses
    py = pos_y / pixel_size + round(obj.shape[0] / 2) + 0.5
    px = pos_x / pixel_size + round(obj.shape[1] / 2) + 0.5

    mode_power = (np.abs(probe) ** 2).sum(axis=(1, 2))
    mode_frac = mode_power / mode_power.sum()

    # ---------------- text summary ----------------
    print(f"\n=== {name} ===")
    print(f"  dp file            {dp_file}")
    print(f"  para file          {para_file}")
    print(f"  patterns           {n_patterns}  of shape {dp_h}x{dp_w}")
    print(f"  object             {obj.shape}  {obj.dtype}   pixel {pixel_size:.3e} m")
    print(f"  amplitude          min {np.abs(obj).min():.3f}  max {np.abs(obj).max():.3f}")
    print(f"  phase (rad)        min {np.angle(obj).min():.3f}  max {np.angle(obj).max():.3f}")
    print(f"  probe              {probe.shape}  {n_modes} mode(s)")
    print(f"  mode power frac    {np.array2string(mode_frac, precision=3)}")
    print(f"  scan extent (px)   y [{py.min():.0f}, {py.max():.0f}]  x [{px.min():.0f}, {px.max():.0f}]")
    print(f"  intensity          min {frames.min():.3g}  max {frames.max():.3g}  (sampled frames)")
    print(f"  _para keys         {para_keys}")

    # ---------------- figure ----------------
    n_show = min(n_modes, args.max_modes)
    n_idx = len(idxs)
    ncols = max(4, n_show, 2 * n_idx)
    nrows = 4
    fig = plt.figure(figsize=(3.1 * ncols, 3.4 * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.25)

    def show(ax, data, title, cmap="viridis", log=False):
        kw = {}
        if log:
            # clip to the top ~5 decades, otherwise the central beam is the only
            # thing visible and the fringes disappear into the floor
            d = np.log10(np.maximum(data, 0) + 1e-6)
            hi = d.max()
            kw = dict(vmin=max(hi - 5, d.min()), vmax=hi)
        else:
            d = data
        im = ax.imshow(d, cmap=cmap, **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        return im

    # --- row 1: the object, with scan positions ---
    half = ncols // 2
    ax = fig.add_subplot(gs[0, :half])
    show(ax, np.abs(obj), f"{name}: object AMPLITUDE  {obj.shape}", cmap="gray")
    ax.plot(px, py, ".", ms=1.2, color="tab:red", alpha=0.5)
    ax.plot(px[idxs], py[idxs], "o", ms=9, mfc="none", mec="yellow", mew=2)

    ax = fig.add_subplot(gs[0, half:])
    show(ax, np.angle(obj), "object PHASE (rad)  — red dots = scan positions", cmap="twilight")
    ax.plot(px, py, ".", ms=1.2, color="tab:red", alpha=0.5)
    ax.plot(px[idxs], py[idxs], "o", ms=9, mfc="none", mec="yellow", mew=2)

    # --- row 2: probe modes ---
    for m in range(n_show):
        ax = fig.add_subplot(gs[1, m])
        show(ax, np.abs(probe[m]), f"probe mode {m} |A|  ({mode_frac[m] * 100:.1f}% power)")
    if n_modes > n_show:
        fig.text(0.01, 0.52, f"(+{n_modes - n_show} more modes not drawn)", fontsize=8)

    # --- row 3: measured diffraction patterns ---
    for k, i in enumerate(idxs):
        ax = fig.add_subplot(gs[2, k])
        show(ax, frames[k], f"measured dp #{i}  (log10 intensity)", cmap="inferno", log=True)

    # --- row 4: physics check, simulated vs measured ---
    size = dp_h
    for k, i in enumerate(idxs):
        patch = crop_patch(obj, py[i], px[i], size)
        sim = simulate(patch, probe)
        meas = frames[k]
        # scale simulation to the measured total counts so the two are comparable
        sim = sim * (meas.sum() / max(sim.sum(), 1e-12))

        ax = fig.add_subplot(gs[3, 2 * k])
        show(ax, np.angle(patch), f"object patch #{i} PHASE", cmap="twilight")
        ax = fig.add_subplot(gs[3, 2 * k + 1])
        show(ax, sim, f"|FFT(probe x patch)|^2  #{i}", cmap="inferno", log=True)

    fig.suptitle(
        f"{name}   |   {n_patterns} patterns, {dp_h}x{dp_w}   |   object {obj.shape}   |   "
        f"{n_modes}-mode probe\n"
        "row1 full object + scan grid   row2 probe modes   row3 measured diffraction   "
        "row4 patch -> simulated diffraction (compare with row3)",
        fontsize=11,
    )
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\n  wrote {out}\n")


if __name__ == "__main__":
    main()
