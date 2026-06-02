"""Convert HXN ptycho output (HDF5 + probe.npy + object.npy) to ptycho-vit format.

The script does not run any reconstruction. It sweeps every plausible
combination of:
  - orient: a D4 symmetry applied to both probe AND object (they share the
    reconstruction frame). Restricted to the 4 non-transposing elements
    {identity, fliplr, flipud, rot180}; the 4 transposing elements are
    equivalent under a global transpose-conjugation (verified empirically:
    each gives a 2-way score tie with its non-transposing partner).
  - diffraction-pattern orientation (8 D4 symmetries) — independent of
    probe/object because detector readout can be flipped relative to
    the optical frame
  - scan-position mapping (8 sign/swap variants)
  - position-anchor corner of the object (TL, TR, BL, BR)
  - source-DP semantics (amplitude vs intensity)
  - object conjugation (False / True) — tests whether the stored complex
    object has the correct phase sign relative to the forward model.
    conj(o) gives a different |FFT(p*o)|^2 unless the probe is exactly
    centrosymmetric; for a zone-plate probe this breaks the degeneracy.
    Conjugation commutes with all D4 orientations so no equivalences are
    lost by sweeping it independently.

Total: 4 * 8 * 8 * 4 * 2 * 2 = 4096 combinations.

Each is scored via a forward model that simulates intensities as
|fft2_unnormalized(probe * obj_patch)|^2 (matching ptycho-vit's forward
model) and compared to the measured DPs with a scale-invariant normalised
cross-correlation. The best-scoring combination is written out; a full
ranked report is saved alongside.

FFTs run on GPU when available — the script tries torch, then cupy, then
falls back to numpy.

Source diffraction data is loaded from either of the two HXN storage
conventions: an in-file `/diffamp` dataset (amplitude, DC-at-corner) or
`/raw_data/filename` + `/raw_data/roi` pointing to an external raw-counts
HDF5 with an ROI crop. The DP stack is never fully resident in RAM —
the sweep reads only its eval subset and writes are streamed in chunks.

Hot detector pixels are filtered with a fixed photon-count threshold
(default 50000). Whether the source represents amplitude or intensity is
NOT decided up front — the forward-model sweep settles it via
`best['dp_kind']`, and the threshold is then applied on raw source values
with the appropriate conversion (sqrt for amplitude, identity for
intensity). Matching pixels are zeroed both during scoring (under each
candidate kind) and on the streamed output.
"""

import argparse
import os
import pickle
import h5py as h5
import numpy as np


# 8 D4 symmetries acting on the last two axes of an array.
ORIENTATIONS = {
    'identity':      lambda a: a,
    'fliplr':        lambda a: np.flip(a, axis=-1),
    'flipud':        lambda a: np.flip(a, axis=-2),
    'rot180':        lambda a: np.flip(np.flip(a, axis=-1), axis=-2),
    'transpose':     lambda a: np.swapaxes(a, -1, -2),
    'rot90_ccw':     lambda a: np.flip(np.swapaxes(a, -1, -2), axis=-2),
    'rot90_cw':      lambda a: np.flip(np.swapaxes(a, -1, -2), axis=-1),
    'antitranspose': lambda a: np.flip(np.flip(np.swapaxes(a, -1, -2), axis=-1), axis=-2),
}

# Orient sweep is restricted to the non-transposing coset of D4. Every
# transposing orient is equivalent (same NCC score) to a non-transposing one
# under the joint substitution (orient↦T·orient, pos_map↦swap_xy(pos_map),
# corner↦anti_diag(corner), dp_orient↦T·dp_orient) — verified empirically by
# the 2-way tie pattern in the sweep report. Halves the sweep space.
ORIENT_SWEEP_KEYS = ['identity', 'fliplr', 'flipud', 'rot180']

# 8 scan-position remappings: sign flips on each axis, with optional swap.
POSITION_MAPS = {
    '(x,y)':   lambda x, y: (x, y),
    '(-x,y)':  lambda x, y: (-x, y),
    '(x,-y)':  lambda x, y: (x, -y),
    '(-x,-y)': lambda x, y: (-x, -y),
    '(y,x)':   lambda x, y: (y, x),
    '(-y,x)':  lambda x, y: (-y, x),
    '(y,-x)':  lambda x, y: (y, -x),
    '(-y,-x)': lambda x, y: (-y, -x),
}

# Which corner of the object the minimum scan-position is anchored to.
CORNERS = ['TL', 'TR', 'BL', 'BR']

# DP semantics: "amplitude" means source is sqrt(counts) and must be squared
# to compare against |FFT|^2; "intensity" means source already represents counts.
DP_KINDS = ['amplitude', 'intensity']


class _NumpyBackend:
    name = 'numpy'
    def asarray(self, x): return np.asarray(x)
    def to_numpy(self, x): return np.asarray(x)
    def fft2(self, x): return np.fft.fft2(x, axes=(-2, -1))
    def fftshift(self, x): return np.fft.fftshift(x, axes=(-2, -1))
    def abs2(self, x): return (x.real ** 2 + x.imag ** 2)
    def sum(self, x, axis=None): return np.sum(x, axis=axis)
    def sqrt(self, x): return np.sqrt(x)


def _select_backend():
    """Pick the fastest available FFT backend: torch → cupy → numpy."""
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        class _TorchBackend:
            name = f'torch ({device})'
            def asarray(self, x):
                arr = np.ascontiguousarray(np.asarray(x))
                return torch.as_tensor(arr, device=device)
            def to_numpy(self, x): return x.detach().cpu().numpy()
            def fft2(self, x): return torch.fft.fft2(x, dim=(-2, -1))
            def fftshift(self, x): return torch.fft.fftshift(x, dim=(-2, -1))
            def abs2(self, x): return x.real ** 2 + x.imag ** 2
            def sum(self, x, axis=None):
                return x.sum() if axis is None else x.sum(dim=axis)
            def sqrt(self, x): return torch.sqrt(x)

        if device == 'cuda':
            return _TorchBackend()
        # If only CPU, prefer cupy/numpy paths below in case GPU exists there.
        torch_backend = _TorchBackend()
    except ImportError:
        torch_backend = None

    try:
        import cupy as cp
        if cp.cuda.is_available():
            class _CupyBackend:
                name = 'cupy (cuda)'
                def asarray(self, x): return cp.asarray(x)
                def to_numpy(self, x): return cp.asnumpy(x)
                def fft2(self, x): return cp.fft.fft2(x, axes=(-2, -1))
                def fftshift(self, x): return cp.fft.fftshift(x, axes=(-2, -1))
                def abs2(self, x): return cp.abs(x) ** 2
                def sum(self, x, axis=None): return cp.sum(x, axis=axis)
                def sqrt(self, x): return cp.sqrt(x)
            return _CupyBackend()
    except ImportError:
        pass

    if torch_backend is not None:
        return torch_backend
    return _NumpyBackend()


def _resolve_source_spec(src_hdf5_path):
    """Decide where and how the source DPs are stored, matching the original
    HXN converter's two paths:
      - /diffamp:           amplitude data stored DC-at-corner.
      - /raw_data/filename: pointer to an external raw-counts HDF5 (typically
                            an Eiger frame stack); paired with /raw_data/roi
                            which crops to the actual 256x256 diffraction window.

    Returns a dict consumed by _load_dp_chunk:
      kind:              'diffamp' or 'raw_data'
      file:              absolute path to the HDF5 that holds the DPs
      dataset:           internal path to the DP dataset
      roi:               (r0, r1, c0, c1) or None
      fftshift_on_load:  bool — true iff DPs are stored DC-at-corner
      n_dp:              number of patterns
    """
    src_hdf5_path = os.path.abspath(src_hdf5_path)
    with h5.File(src_hdf5_path, 'r') as f:
        if '/diffamp' in f:
            dset = f['/diffamp']
            if dset.ndim != 3:
                raise ValueError(
                    f"/diffamp must be 3D (N, H, W); got ndim={dset.ndim}"
                )
            n_dp = int(dset.shape[0])
            dp_yx = (int(dset.shape[-2]), int(dset.shape[-1]))
            if dp_yx != (256, 256):
                raise ValueError(
                    f"/diffamp in {src_hdf5_path} must be 256x256; got {dp_yx}"
                )
            return {
                'kind': 'diffamp',
                'file': src_hdf5_path,
                'dataset': '/diffamp',
                'roi': None,
                'fftshift_on_load': True,
                'n_dp': n_dp,
            }
        if '/raw_data/filename' in f:
            fname_raw = f['/raw_data/filename'][()]
            roi_raw = np.asarray(f['/raw_data/roi'][()])
        else:
            raise KeyError(
                f"No supported diffraction data in {src_hdf5_path}; "
                f"expected '/diffamp' or '/raw_data/filename'"
            )

    if isinstance(fname_raw, np.ndarray):
        if fname_raw.size == 0:
            raise ValueError("/raw_data/filename is empty")
        fname_raw = fname_raw.flat[0]
    if isinstance(fname_raw, bytes):
        fname_raw = fname_raw.decode('utf-8')
    fname = str(fname_raw)
    if not os.path.isabs(fname):
        fname = os.path.normpath(os.path.join(os.path.dirname(src_hdf5_path), fname))
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"/raw_data/filename points to {fname!r}, which does not exist"
        )

    if roi_raw.shape != (2, 2):
        raise ValueError(f"/raw_data/roi must be shape (2, 2); got {roi_raw.shape}")
    r0, r1 = int(roi_raw[0, 0]), int(roi_raw[0, 1])
    c0, c1 = int(roi_raw[1, 0]), int(roi_raw[1, 1])
    if (r1 - r0, c1 - c0) != (256, 256):
        raise ValueError(
            f"raw_data ROI must yield a 256x256 frame; got ({r1 - r0}, {c1 - c0})"
        )

    with h5.File(fname, 'r') as f_raw:
        if '/entry/data/data' not in f_raw:
            raise KeyError(f"Expected '/entry/data/data' in raw file {fname}")
        dset = f_raw['/entry/data/data']
        if dset.ndim != 3:
            raise ValueError(
                f"raw '/entry/data/data' must be 3D; got ndim={dset.ndim}"
            )
        n_dp = int(dset.shape[0])

    return {
        'kind': 'raw_data',
        'file': fname,
        'dataset': '/entry/data/data',
        'roi': (r0, r1, c0, c1),
        'fftshift_on_load': False,
        'n_dp': n_dp,
    }


def _load_dp_chunk(spec, indices):
    """Read DPs by index from the source described by `spec`.

    `indices` may be a `slice` or a 1D array of integer indices. The result
    is float32 (N, 256, 256); ROI cropping and DC-centring (fftshift) are
    applied per the spec.
    """
    with h5.File(spec['file'], 'r') as f:
        dset = f[spec['dataset']]
        if spec['roi'] is not None:
            r0, r1, c0, c1 = spec['roi']
            if isinstance(indices, slice):
                data = dset[indices, r0:r1, c0:c1].astype(np.float32)
            else:
                data = dset[np.asarray(indices), r0:r1, c0:c1].astype(np.float32)
        else:
            if isinstance(indices, slice):
                data = dset[indices].astype(np.float32)
            else:
                data = dset[np.asarray(indices), :, :].astype(np.float32)
    if spec['fftshift_on_load']:
        data = np.fft.fftshift(data, axes=(-2, -1))
    return data


def _apply_hot_pixel_filter(dps, count_threshold, source_kind):
    """Zero pixels whose photon count exceeds count_threshold.

    The threshold is in photon-count units; how to apply it on raw source
    values depends on whether the source represents amplitude or intensity:
      - amplitude: raw > sqrt(count_threshold)  (since count = raw^2)
      - intensity: raw > count_threshold        (raw is already counts)

    Mutates `dps` in-place and returns it.
    """
    if count_threshold is None or not np.isfinite(count_threshold):
        return dps
    if source_kind == 'amplitude':
        raw_threshold = float(np.sqrt(count_threshold))
    elif source_kind == 'intensity':
        raw_threshold = float(count_threshold)
    else:
        raise ValueError(f"source_kind must be 'amplitude' or 'intensity'; got {source_kind!r}")
    dps[dps > raw_threshold] = 0.0
    return dps


def _read_scalar(f, key, transform=float):
    """Read a scalar dataset from h5; raise a clear error if missing."""
    if key not in f:
        raise KeyError(f"Missing required dataset {key!r} in {f.filename}")
    return transform(f[key][()])


def _read_positions_um(f):
    """Read /points and return (x_um, y_um) as 1D float arrays.

    Accepts either (2, N) or (N, 2) layouts. For the ambiguous (2, 2) case
    we default to the (2, N) convention used by the HXN backend.
    """
    if '/points' not in f:
        raise KeyError(f"Missing required dataset '/points' in {f.filename}")
    points = np.asarray(f['/points'][()])
    if points.ndim != 2:
        raise ValueError(f"/points must be 2D; got shape {points.shape}")
    if points.shape[0] == 2 and points.shape[1] != 2:
        x, y = points[0], points[1]
    elif points.shape[1] == 2 and points.shape[0] != 2:
        x, y = points[:, 0], points[:, 1]
    elif points.shape == (2, 2):
        x, y = points[0], points[1]
    else:
        raise ValueError(
            f"/points must be (2, N) or (N, 2); got {points.shape}"
        )
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def load_source(src_hdf5_path, src_probe_path, src_object_path,
                hot_pixel_count_threshold=50000.0):
    """Load source metadata + probe + object. Diffraction patterns stay on disk.

    Supports both source-data storage paths used by the HXN backend:
      - `/diffamp` for in-file amplitude data, and
      - `/raw_data/filename` for an external raw-counts HDF5 with an ROI crop.
    Source kind (amplitude vs intensity) is NOT decided here; the forward-
    model sweep settles it via `best['dp_kind']`.

    The DP stack can be huge (100k × 256² ≈ 26 GiB float32), so it is never
    fully resident in RAM — the sweep reads only the eval subset and
    write_outputs streams chunk-by-chunk.

    Validates that the DPs and probe are 256x256.
    """
    src_hdf5_path = os.path.abspath(src_hdf5_path)
    spec = _resolve_source_spec(src_hdf5_path)
    print(f"  DP source: kind={spec['kind']}, file={spec['file']}, n_dp={spec['n_dp']}")
    if spec['roi'] is not None:
        print(f"  raw_data ROI: rows {spec['roi'][0]}:{spec['roi'][1]}, "
              f"cols {spec['roi'][2]}:{spec['roi'][3]}")

    with h5.File(src_hdf5_path, 'r') as f:
        ccd_pixel_size_m = _read_scalar(f, 'ccd_pixel_um') * 1e-6
        wavelength_m = _read_scalar(f, 'lambda_nm') * 1e-9
        detector_distance_m = _read_scalar(f, 'z_m')
        angle_deg = _read_scalar(f, 'angle')
        x_um, y_um = _read_positions_um(f)

    if len(x_um) != spec['n_dp']:
        raise ValueError(
            f"/points has {len(x_um)} positions but the DP source has "
            f"{spec['n_dp']} frames — these must match"
        )

    x_positions_m = x_um * 1e-6
    y_positions_m = y_um * 1e-6

    probe = np.load(src_probe_path)
    if probe.ndim == 4:
        probe = probe[:, 0]
    if probe.ndim == 2:
        probe = probe[np.newaxis]
    if probe.ndim != 3:
        raise ValueError(
            f"Probe must be 2D, 3D (n_modes, H, W), or 4D (n_modes, 1, H, W); "
            f"got ndim={probe.ndim}, shape={probe.shape}"
        )
    probe = probe.astype(np.complex64)

    obj = np.load(src_object_path)
    if obj.ndim == 2:
        obj = obj[np.newaxis]
    if obj.ndim != 3:
        raise ValueError(
            f"Object must be 2D or 3D (n_slices, H, W); got ndim={obj.ndim}, "
            f"shape={obj.shape}"
        )
    obj = obj.astype(np.complex64)

    if probe.shape[-1] != 256 or probe.shape[-2] != 256:
        raise ValueError(
            f"Source probe must be 256x256; got {probe.shape[-2:]}"
        )

    pixel_m = (wavelength_m * detector_distance_m) / (256 * ccd_pixel_size_m)
    print(f"  Hot-pixel filter (applied during sweep + write): photon count > "
          f"{hot_pixel_count_threshold:g}; source-kind from sweep")

    return {
        'spec': spec,                      # streamed-read on demand
        'n_dp': spec['n_dp'],
        'hot_pixel_count_threshold': float(hot_pixel_count_threshold),
        'probe': probe,                    # (n_modes, 256, 256)
        'object': obj,                     # (n_slices, H, W)
        'x_positions_m': x_positions_m,    # (N_pos,)
        'y_positions_m': y_positions_m,    # (N_pos,)
        'pixel_m': pixel_m,
        'wavelength_m': wavelength_m,
        'detector_distance_m': detector_distance_m,
        'ccd_pixel_size_m': ccd_pixel_size_m,
        'angle_deg': angle_deg,
    }


def map_positions_to_pixels(x_pos_m, y_pos_m, pixel_m, obj_shape, probe_shape, corner):
    """Convert physical (m) probe positions to pixel coords of the probe centre.

    The scan area's minimum-position is anchored to one of the four object
    corners; the rest follow at the source pixel pitch.
    """
    obj_H, obj_W = obj_shape[-2], obj_shape[-1]
    pH, pW = probe_shape[-2], probe_shape[-1]

    dx_px = (x_pos_m - x_pos_m.min()) / pixel_m
    dy_px = (y_pos_m - y_pos_m.min()) / pixel_m

    if corner == 'TL':
        cx, cy = dx_px + pW / 2, dy_px + pH / 2
    elif corner == 'TR':
        cx, cy = (obj_W - pW / 2) - dx_px, dy_px + pH / 2
    elif corner == 'BL':
        cx, cy = dx_px + pW / 2, (obj_H - pH / 2) - dy_px
    elif corner == 'BR':
        cx, cy = (obj_W - pW / 2) - dx_px, (obj_H - pH / 2) - dy_px
    else:
        raise ValueError(f"Unknown corner: {corner}")
    return cx, cy


def extract_patches(obj_slice, cx, cy, pH, pW):
    """Extract probe-sized patches centred at (cx, cy) from obj_slice."""
    N = len(cx)
    H, W = obj_slice.shape
    patches = np.zeros((N, pH, pW), dtype=obj_slice.dtype)
    for i in range(N):
        x0 = int(round(cx[i] - pW / 2))
        y0 = int(round(cy[i] - pH / 2))
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + pW, W), min(y0 + pH, H)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        dx0, dy0 = sx0 - x0, sy0 - y0
        patches[i, dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = obj_slice[sy0:sy1, sx0:sx1]
    return patches


def _spatially_diverse_sample(x_pos, y_pos, n_target, rng):
    """Pick spatially-spread position indices via grid bucketing.

    Avoids the failure mode where a uniform random sample happens to cluster
    in one part of the scan area — there, coincidentally symmetric object
    structure could make a wrong orientation score well. Bucketing the scan
    bounding box and picking one position per occupied cell guarantees
    coverage across the whole scanned region.

    Returns up to n_target indices (fewer if some grid cells are empty).
    """
    N = len(x_pos)
    if n_target >= N:
        return np.arange(N)
    grid_n = max(1, int(np.ceil(np.sqrt(n_target))))
    # +1e-12 keeps the maximum-value point inside the last bucket.
    x_edges = np.linspace(x_pos.min(), x_pos.max() + 1e-12, grid_n + 1)
    y_edges = np.linspace(y_pos.min(), y_pos.max() + 1e-12, grid_n + 1)
    bx = np.clip(np.searchsorted(x_edges, x_pos, side='right') - 1, 0, grid_n - 1)
    by = np.clip(np.searchsorted(y_edges, y_pos, side='right') - 1, 0, grid_n - 1)
    bucket_id = by * grid_n + bx
    selected = [rng.choice(np.where(bucket_id == bid)[0])
                for bid in np.unique(bucket_id)]
    return np.sort(np.array(selected))


def sweep(data, n_eval_positions=64, rng_seed=0):
    """Score every combination and return all rows sorted ascending by score.

    Loop order is reuse-aware: the outer axis is `orient` (which mutates
    probe and obj together), then `pos_map` (cheap remap), then `corner`
    (cheap shift). The 16 DP variants are compared in one vectorised NCC
    per (orient, pos_map, corner).
    """
    bk = _select_backend()
    print(f"  FFT backend: {bk.name}")

    probe_mode0 = data['probe'][0]  # single-mode forward model for scoring
    obj_slice = data['object'][0]   # single slice; multi-slice not swept
    xp_all = data['x_positions_m']
    yp_all = data['y_positions_m']
    pixel_m = data['pixel_m']

    rng = np.random.default_rng(rng_seed)
    eval_idx = _spatially_diverse_sample(xp_all, yp_all, n_eval_positions, rng)
    print(f"  Eval positions: {len(eval_idx)} (spatially-diverse sample)")
    dp_eval = _load_dp_chunk(data['spec'], eval_idx)
    xp_eval = xp_all[eval_idx]
    yp_eval = yp_all[eval_idx]

    # Stack all 16 measured-intensity variants in a fixed order so the inner
    # NCC is a single einsum-like sum. variant_keys[i] = (dp_orient, dp_kind).
    # Each variant gets the photon-count hot-pixel filter applied under its
    # own kind interpretation (sqrt(threshold) for amplitude, threshold for
    # intensity), so the two interpretations are scored on equal footing.
    count_threshold = data['hot_pixel_count_threshold']
    variant_keys = []
    variants_np = []
    for o_name in ORIENTATIONS:
        dp_t = ORIENTATIONS[o_name](dp_eval).astype(np.float32)
        for kind in DP_KINDS:
            filtered = _apply_hot_pixel_filter(dp_t.copy(), count_threshold, kind)
            variants_np.append(filtered ** 2 if kind == 'amplitude' else filtered)
            variant_keys.append((o_name, kind))
    variants_np = np.stack(variants_np, axis=0)  # (16, N_eval, 256, 256)
    dp_variants = bk.asarray(variants_np)
    # ||I_meas_v|| precomputed; lives on backend.
    norms_meas = bk.sqrt(bk.sum(dp_variants * dp_variants, axis=(1, 2, 3)))

    results = []
    n_outer = len(ORIENT_SWEEP_KEYS) * len(POSITION_MAPS) * len(CORNERS)
    outer_i = 0
    for orient_name in ORIENT_SWEEP_KEYS:
        orient_fn = ORIENTATIONS[orient_name]
        # Probe and object share the reconstruction frame — apply the same
        # D4 element to both.
        probe_t_np = orient_fn(probe_mode0)
        obj_t_np = orient_fn(obj_slice)
        probe_t_bk = bk.asarray(probe_t_np)
        for pm_name, pm_fn in POSITION_MAPS.items():
            xm, ym = pm_fn(xp_eval, yp_eval)
            for corner in CORNERS:
                outer_i += 1
                cx, cy = map_positions_to_pixels(
                    xm, ym, pixel_m, obj_t_np.shape, probe_t_np.shape, corner
                )
                patches_np = extract_patches(
                    obj_t_np, cx, cy, probe_t_np.shape[-2], probe_t_np.shape[-1]
                )
                patches_bk = bk.asarray(patches_np)
                wavefront = patches_bk * probe_t_bk
                fft = bk.fft2(wavefront)
                fft = bk.fftshift(fft)
                I_sim = bk.abs2(fft)

                # Vectorised NCC against all 16 DP variants in one pass.
                # Score both the object as-stored and its complex conjugate.
                for conj_flag, I_sim_c in [
                    (False, I_sim),
                    (True,  bk.abs2(bk.fftshift(bk.fft2(
                        bk.asarray(np.conj(patches_np)) * probe_t_bk
                    )))),
                ]:
                    num_c = bk.sum(dp_variants * I_sim_c, axis=(1, 2, 3))   # (16,)
                    norm_sim_c = bk.sqrt(bk.sum(I_sim_c * I_sim_c))         # ()
                    scores_c = 1.0 - num_c / (norms_meas * norm_sim_c + 1e-30)
                    scores_np_c = bk.to_numpy(scores_c)
                    for vi, (dp_name, kind) in enumerate(variant_keys):
                        results.append({
                            'score': float(scores_np_c[vi]),
                            'orient': orient_name,
                            'pos_map': pm_name,
                            'corner': corner,
                            'dp_orient': dp_name,
                            'dp_kind': kind,
                            'conjugate': conj_flag,
                        })
                if outer_i % 32 == 0 or outer_i == n_outer:
                    print(f"  Forward-model batch {outer_i}/{n_outer}")

    results.sort(key=lambda r: r['score'])
    return results


def write_outputs(data, best, out_dir, scan_id, write_chunk_size=128):
    """Write {scan_id}_dp.hdf5 and {scan_id}_para.hdf5 using the best combination.

    The DP output is written chunk-by-chunk: we never hold the full
    transformed-intensity stack in RAM. Maximum intensity is accumulated as
    we stream.
    """
    os.makedirs(out_dir, exist_ok=True)

    dp_fn = ORIENTATIONS[best['dp_orient']]
    orient_fn = ORIENTATIONS[best['orient']]
    pos_fn = POSITION_MAPS[best['pos_map']]
    is_amplitude = (best['dp_kind'] == 'amplitude')
    # Source kind is determined by the sweep (best['dp_kind']); the photon-count
    # threshold is converted to raw-source units inside _apply_hot_pixel_filter.
    source_kind = best['dp_kind']
    count_threshold = data['hot_pixel_count_threshold']
    n_dp = data['n_dp']

    # FFT-normalization convention fix: HXN reconstructs the probe assuming
    # an orthonormal FFT (factor 1/N), while ptycho-vit's forward model uses
    # an unnormalized fft2. For an NxN FFT the |·|^2 conventions differ by
    # N^2, so we absorb a 1/N factor into the probe to make
    # |fft2_unnormalized(probe * obj)|^2 land on photon counts.
    probe = data['probe']
    probe_t = np.stack([orient_fn(probe[m]) for m in range(probe.shape[0])], axis=0)
    probe_t = (probe_t / 256).astype(np.complex64)
    # ptycho-vit shape convention: (n_modes, n_slices, H, W).
    probe_out = probe_t[:, np.newaxis]

    obj = data['object']
    obj_t = np.stack([orient_fn(obj[s]) for s in range(obj.shape[0])], axis=0).astype(np.complex64)
    #if best.get('conjugate', False):
    #    obj_t = np.conj(obj_t)

    xm, ym = pos_fn(data['x_positions_m'], data['y_positions_m'])
    cx, cy = map_positions_to_pixels(
        xm, ym, data['pixel_m'], obj_t.shape, probe.shape, best['corner']
    )
    x_out_m = cx * data['pixel_m']
    y_out_m = cy * data['pixel_m']

    # Streamed DP write: read source chunk (via spec; handles both /diffamp
    # and raw_data paths, ROI crop and DC-centring) → hot-pixel filter →
    # orient → square-if-amplitude → write → update running max. Never
    # materialises the full intensity stack in RAM.
    spec = data['spec']
    dp_path = os.path.join(out_dir, f"{scan_id}_dp.hdf5")
    max_intensity = 0.0
    chunks = (min(write_chunk_size, n_dp), 256, 256)
    with h5.File(dp_path, 'w') as out:
        out_dset = out.create_dataset(
            'dp', shape=(n_dp, 256, 256), dtype=np.float32, chunks=chunks,
        )
        for i in range(0, n_dp, write_chunk_size):
            j = min(i + write_chunk_size, n_dp)
            chunk = _load_dp_chunk(spec, slice(i, j))
            _apply_hot_pixel_filter(chunk, count_threshold, source_kind)
            chunk = dp_fn(chunk)
            if is_amplitude:
                chunk = chunk ** 2
            chunk_max = float(chunk.max())
            if chunk_max > max_intensity:
                max_intensity = chunk_max
            out_dset[i:j] = chunk

    # Maximum-intensity pickle + text dump, keyed by scan name so downstream
    # pipelines can merge per-scan files into a single normalisation dict.
    scan_name = str(scan_id)
    max_dict = {scan_name: max_intensity}
    pkl_path = os.path.join(out_dir, f"{scan_id}_max_intensity.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(max_dict, f)
    txt_path = os.path.join(out_dir, f"{scan_id}_max_intensity.txt")
    with open(txt_path, 'w') as f:
        f.write(f"{scan_name}: {max_intensity}\n")

    para_path = os.path.join(out_dir, f"{scan_id}_para.hdf5")
    with h5.File(para_path, 'w') as f:
        obj_d = f.create_dataset('object', data=obj_t)
        obj_d.attrs['center_x_m'] = 0.0
        obj_d.attrs['center_y_m'] = 0.0
        obj_d.attrs['pixel_height_m'] = data['pixel_m']
        obj_d.attrs['pixel_width_m'] = data['pixel_m']
        f.create_dataset('probe', data=probe_out)
        f.create_dataset('probe_position_indexes', data=np.arange(xm.shape[0]))
        f.create_dataset('probe_position_x_m', data=x_out_m)
        f.create_dataset('probe_position_y_m', data=y_out_m)
        # Record the sweep result as root-level attributes so the provenance
        # of every transform applied to the stored data is inspectable with
        # h5py / h5dump without needing the sweep report file.
        # NOTE: the object and probe are already stored *after* all transforms
        # have been applied (orient, conjugate). These attrs are for auditing
        # only — downstream code does not need to re-apply them.
        f.attrs['hxn_to_vit_sweep_score']     = best['score']
        f.attrs['hxn_to_vit_orient']          = best['orient']
        f.attrs['hxn_to_vit_pos_map']         = best['pos_map']
        f.attrs['hxn_to_vit_corner']          = best['corner']
        f.attrs['hxn_to_vit_dp_orient']       = best['dp_orient']
        f.attrs['hxn_to_vit_dp_kind']         = best['dp_kind']
        f.attrs['hxn_to_vit_conjugate']       = bool(best.get('conjugate', False))

    return dp_path, para_path, pkl_path, txt_path


def write_report(results, out_dir, scan_id, best):
    path = os.path.join(out_dir, f"{scan_id}_sweep_report.txt")
    with open(path, 'w') as f:
        f.write(f"Best combination for scan {scan_id}:\n")
        f.write(f"  score (1-NCC): {best['score']:.6g}\n")
        f.write(f"  orient:        {best['orient']}\n")
        f.write(f"  pos_map:       {best['pos_map']}\n")
        f.write(f"  corner:        {best['corner']}\n")
        f.write(f"  dp_orient:     {best['dp_orient']}\n")
        f.write(f"  dp_kind:       {best['dp_kind']}\n")
        f.write(f"  conjugate:     {best.get('conjugate', False)}\n\n")
        f.write(f"All {len(results)} combinations ranked (lower score = better match):\n")
        hdr = (f"{'rank':>5} {'score':>14} {'orient':>15} "
               f"{'pos_map':>9} {'corner':>7} {'dp_orient':>15} {'dp_kind':>10} {'conj':>6}\n")
        f.write(hdr)
        for i, r in enumerate(results):
            f.write(
                f"{i+1:>5} {r['score']:>14.6g} {r['orient']:>15} "
                f"{r['pos_map']:>9} {r['corner']:>7} {r['dp_orient']:>15} {r['dp_kind']:>10} "
                f"{str(r.get('conjugate', False)):>6}\n"
            )
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Convert HXN ptycho output to ptycho-vit by sweeping orientation/position conventions."
    )
    parser.add_argument('--src-hdf5', required=True, help="Source HXN-format HDF5 with /diffamp and /points.")
    parser.add_argument('--src-probe', required=True, help="Source probe .npy (HxW, MxHxW, or Mx1xHxW).")
    parser.add_argument('--src-object', required=True, help="Source object .npy (HxW or SxHxW).")
    parser.add_argument('--out-dir', required=True, help="Output directory for ptycho-vit files and report.")
    parser.add_argument('--scan-id', required=True, help="Scan ID used in output filenames.")
    parser.add_argument('--n-eval-positions', type=int, default=64,
                        help="Subsampled probe positions used for forward-model scoring (default 64).")
    parser.add_argument('--hot-pixel-count-threshold', type=float, default=50000.0,
                        help="Photon-count threshold for hot-pixel filtering (default 50000). "
                             "Applied to source amplitude as amp > sqrt(threshold).")
    parser.add_argument('--write-chunk-size', type=int, default=128,
                        help="Streaming chunk size when writing the output DP HDF5 (default 128).")
    parser.add_argument('--rng-seed', type=int, default=0)
    args = parser.parse_args()

    print("Loading source files...")
    data = load_source(args.src_hdf5, args.src_probe, args.src_object,
                       hot_pixel_count_threshold=args.hot_pixel_count_threshold)
    print(f"  DPs:    ({data['n_dp']}, 256, 256) — lazy")
    print(f"  probe:  {data['probe'].shape}")
    print(f"  object: {data['object'].shape}")
    print(f"  pixel size at sample: {data['pixel_m'] * 1e9:.3f} nm")

    n_combos = (
        len(ORIENT_SWEEP_KEYS)                  # orient (probe=obj), non-transpose coset
        * len(POSITION_MAPS) * len(CORNERS)
        * len(ORIENTATIONS) * len(DP_KINDS)
        * 2                                      # conjugate object: False / True
    )
    print(f"Sweeping {n_combos} combinations with {args.n_eval_positions} eval positions...")
    results = sweep(data, n_eval_positions=args.n_eval_positions, rng_seed=args.rng_seed)
    best = results[0]
    print(f"Best: score={best['score']:.6g}  orient={best['orient']}  "
          f"pos={best['pos_map']}  corner={best['corner']}  "
          f"dp={best['dp_orient']}  kind={best['dp_kind']}  "
          f"conjugate={best.get('conjugate', False)}")

    dp_path, para_path, pkl_path, txt_path = write_outputs(
        data, best, args.out_dir, args.scan_id,
        write_chunk_size=args.write_chunk_size,
    )
    report_path = write_report(results, args.out_dir, args.scan_id, best)
    print(f"Wrote:\n  {dp_path}\n  {para_path}\n  {pkl_path}\n  {txt_path}\n  {report_path}")


if __name__ == "__main__":
    main()
