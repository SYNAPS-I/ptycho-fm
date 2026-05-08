"""Tiled-backed ptychography dataset.

Mirrors :class:`PtychographyDataset` but reads from a single per-run Tiled
container rather than a paired ``*_dp.hdf5`` / ``*_para.hdf5`` pair on disk.
The container layout is the one written by holoptycho when its ``fine_tune``
config flag is set (see holoptycho/AGENTS.md):

    <run>/diffraction/dp                       (nz, H, W) uint8 amplitude
    <run>/diffraction/probe_position_x_m       (nz,) float64 meters
    <run>/diffraction/probe_position_y_m       (nz,) float64 meters
    <run>/final/object                          (1, H_obj, W_obj) complex
    <run>/final/probe                           (N_modes, H, W) complex
    <run>.metadata['x_pixel_m']                 float, sample-plane pixel size

The diffraction data is stored as **amplitude** (= ``sqrt(intensity)``,
rounded to uint8) rather than raw intensity, because Tiled doesn't accept
compressed ``write_block`` payloads and uint8 storage halves the on-the-wire
upload volume. The 1-count quantization is below the Poisson noise floor for
ML training, so the loss is negligible. Concretely this means the loader
**skips the sqrt step** that the h5 path applies — the bytes coming off Tiled
are already amplitude.

The ``__getitem__`` contract (return order, shapes, dtypes) is identical to
:class:`PtychographyDataset`, so a downstream training loop can swap data
sources without other changes.
"""

import numpy as np
import torch
from typing import Optional

from data import PtychographyDataset


class TiledPtychographyDataset(PtychographyDataset):
    """Tiled-backed sibling of :class:`PtychographyDataset`.

    Args:
        tiled_uri: Tiled URI ending in ``/<run_uid>/`` — one holoptycho run.
        scale, normalization_dict_path, default_normalization, apply_noise,
        cache_object, max_probe_modes, target_size, object_name: same
        semantics as :class:`PtychographyDataset`.
    """

    def __init__(
        self,
        tiled_uri: str,
        scale: float = 100000.0,
        normalization_dict_path: Optional[str] = None,
        default_normalization: float = 100000.0,
        apply_noise: bool = True,
        cache_object: bool = True,
        max_probe_modes: int = 8,
        target_size: Optional[int] = 256,
        object_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        # Skip PtychographyDataset.__init__ — its h5-specific path expects a
        # local file. Set up the parent's attributes by hand, then run the
        # Tiled-specific loaders.
        torch.utils.data.Dataset.__init__(self)
        import os
        from tiled.client import from_uri

        self.tiled_uri = tiled_uri
        self.scale = scale
        self.normalization_dict_path = normalization_dict_path
        self.default_normalization = default_normalization
        self.apply_noise = apply_noise
        self.cache_object = cache_object
        self.max_probe_modes = max_probe_modes
        self.target_size = target_size

        self._cached_object = None
        self._cached_probe_positions = None
        self._cached_probe = None

        # Auth precedence: explicit kwarg > TILED_API_KEY env var > cached
        # `tiled login` credentials. Passing api_key=None to from_uri is the
        # signal to fall back to the cached credentials, so we only forward
        # the kwarg when something resolved to a non-empty string.
        resolved_api_key = api_key or os.environ.get("TILED_API_KEY") or None
        from_uri_kwargs = {"api_key": resolved_api_key} if resolved_api_key else {}
        self._run = from_uri(tiled_uri, **from_uri_kwargs)
        # holoptycho stamps `fine_tunable: true` into run metadata iff the
        # iterative branch will populate `final/probe` and `final/object` —
        # the supervised targets this loader requires. Reject vit-only runs
        # (or runs that predate the metadata flag) with a clear message
        # rather than failing later with a cryptic KeyError on `final/`.
        run_meta = dict(self._run.metadata or {})
        if not run_meta.get("fine_tunable", False):
            raise ValueError(
                f"Tiled run {tiled_uri} is not marked fine_tunable in its "
                "metadata. ptycho-vit's training loader needs final/probe "
                "and final/object as supervised targets; only runs created "
                "with recon_mode='iterative' or 'both' produce them. Re-run "
                "holoptycho on this scan with one of those recon modes."
            )
        diffraction = self._run["diffraction"]
        final = self._run["final"]
        self._dp_node = diffraction["dp"]
        self._x_pos_node = diffraction["probe_position_x_m"]
        self._y_pos_node = diffraction["probe_position_y_m"]
        self._object_node = final["object"]
        self._probe_node = final["probe"]

        # Mirror PtychographyDataset's basic-info derivation.
        dp_shape = self._dp_node.shape
        self.num_patterns = int(dp_shape[0])
        self._raw_pattern_shape = tuple(int(s) for s in dp_shape[1:])
        if self.target_size is not None:
            self.pattern_shape = (self.target_size, self.target_size)
        else:
            self.pattern_shape = self._raw_pattern_shape
        self.object_shape = tuple(int(s) for s in self._object_node[0].shape)

        # object_name: prefer caller, else fall back to the run's metadata.
        if object_name is not None:
            self.object_name = object_name
        else:
            md = dict(self._run.metadata or {})
            self.object_name = str(md.get("scan_id") or md.get("run_uid") or "")

        self._load_normalization()

    # ------------------------------------------------------------------
    # Cache / load methods — overridden to read from Tiled instead of h5
    # ------------------------------------------------------------------

    def _cache_positions(self):
        if self._cached_probe_positions is not None:
            return

        # Pixel size lives on the run-container metadata in the Tiled layout
        # (holoptycho writes it as run_metadata['x_pixel_m']) rather than as
        # an attribute on the object array as in the h5 schema.
        md = dict(self._run.metadata or {})
        self.pixel_size_m = float(md["x_pixel_m"])

        pos_y = np.asarray(self._y_pos_node[...])
        pos_x = np.asarray(self._x_pos_node[...])
        positions_raw = np.column_stack([pos_y, pos_x])

        # Auto-detect meters vs pixels — same heuristic as the parent class.
        pos_range_y = float(pos_y.max() - pos_y.min())
        pos_range_x = float(pos_x.max() - pos_x.min())
        obj_h, obj_w = self.object_shape
        positions_likely_pixels = (
            0.1 * obj_h < pos_range_y < 10 * obj_h
            and 0.1 * obj_w < pos_range_x < 10 * obj_w
        )

        if positions_likely_pixels:
            self._cached_probe_positions = torch.from_numpy(
                positions_raw.astype(np.float32)
            )
        else:
            self._cached_probe_positions = torch.from_numpy(
                (positions_raw / self.pixel_size_m).astype(np.float32)
            )

        if self._cached_probe_positions.shape[0] != self.num_patterns:
            raise ValueError(
                f"Mismatch in number of patterns: {self.num_patterns} "
                f"diffraction patterns vs "
                f"{self._cached_probe_positions.shape[0]} probe positions"
            )

        self._pos_origin_coords = torch.tensor(
            self.object_shape, dtype=torch.float32
        ) / 2.0
        self._pos_origin_coords = self._pos_origin_coords.round() + 0.5
        self._cached_probe_positions = (
            self._cached_probe_positions + self._pos_origin_coords
        )

    def _cache_object_data(self):
        if self._cached_object is not None:
            return
        # Object: (1, H, W) complex; loader uses [0].
        self._cached_object = np.asarray(self._object_node[0])

        # Probe: holoptycho writes (N_modes, H, W); _normalize_probe_shape
        # accepts (A, N, H, W) only, so wrap to (1, N, H, W) here.
        probe = np.asarray(self._probe_node[...])
        if probe.ndim == 3:
            probe = probe[np.newaxis, ...]
        probe = self._normalize_probe_shape(probe)
        probe = self._pad_probe(probe, target_modes=self.max_probe_modes)
        if self.target_size is not None:
            probe = self._upsample_probe(probe, self.target_size)
        self._cached_probe = probe

    def _load_pattern(self, pattern_idx: int):
        """Tiled-side equivalent of PtychographyDataset._load_hdf5_pattern.

        holoptycho writes ``dp`` as **uint8 amplitude** (= ``sqrt(intensity)``,
        rounded) to halve the on-the-wire write volume vs raw uint16
        intensity, since Tiled does not accept compressed write_block payloads.
        That means we already have ``A = sqrt(I)``, so we skip the h5 path's
        own ``sqrt`` step.

        The h5 path does ``sqrt((I/N)*scale)`` — algebraically that's
        ``A * sqrt(scale/N)``, which is what we apply here.

        Poisson augmentation lives in intensity space and isn't meaningful on
        pre-sqrt'd amplitudes; experimental data (the only kind that lands in
        Tiled today) is set to ``apply_noise=False`` anyway, so we issue a
        warning and skip if asked to apply it.
        """
        self._cache_positions()
        if self.cache_object:
            self._cache_object_data()

        amp_u8 = np.asarray(self._dp_node[pattern_idx])
        if self.apply_noise:
            import warnings
            warnings.warn(
                "TiledPtychographyDataset: apply_noise=True on pre-sqrt'd "
                "uint8 amplitude; skipping Poisson (use experimental data "
                "with apply_noise=False, or fall back to PtychographyDataset "
                "for simulated h5 data).",
                stacklevel=2,
            )
        diffraction_amp = amp_u8.astype(np.float32) * float(
            np.sqrt(self.scale / self.normalization)
        )

        if (
            self.target_size is not None
            and diffraction_amp.shape[0] != self.target_size
        ):
            diffraction_amp = self._zero_pad_to_target(
                diffraction_amp, self.target_size
            )

        probe_position = self._cached_probe_positions[pattern_idx]

        if self._cached_probe is not None:
            probe = self._cached_probe
        else:
            probe = np.asarray(self._probe_node[...])
            if probe.ndim == 3:
                probe = probe[np.newaxis, ...]
            probe = self._normalize_probe_shape(probe)
            probe = self._pad_probe(probe, target_modes=self.max_probe_modes)
            if self.target_size is not None:
                probe = self._upsample_probe(probe, self.target_size)

        if self._cached_object is not None:
            full_object = self._cached_object
        else:
            full_object = np.asarray(self._object_node[0])

        patch = self._extract_patch(full_object, probe_position)
        amplitude_patch = torch.abs(patch)
        phase_patch = torch.angle(patch)

        if (
            self.target_size is not None
            and amplitude_patch.shape[0] != self.target_size
        ):
            amplitude_patch = torch.from_numpy(
                self._zero_pad_to_target(
                    amplitude_patch.detach().cpu().numpy(), self.target_size
                )
            )
            phase_patch = torch.from_numpy(
                self._zero_pad_to_target(
                    phase_patch.detach().cpu().numpy(), self.target_size
                )
            )

        return diffraction_amp, amplitude_patch, phase_patch, probe, probe_position

    # ------------------------------------------------------------------
    # Item access — same return contract as PtychographyDataset
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        if idx >= self.num_patterns:
            raise IndexError(
                f"Index {idx} out of range for dataset with "
                f"{self.num_patterns} patterns"
            )

        diffraction_amp, amplitude_patch, phase_patch, probe, probe_position = (
            self._load_pattern(idx)
        )

        diffraction_amp = (
            torch.from_numpy(diffraction_amp)
            if isinstance(diffraction_amp, np.ndarray)
            else diffraction_amp
        )
        amplitude_patch = (
            amplitude_patch
            if isinstance(amplitude_patch, torch.Tensor)
            else torch.from_numpy(amplitude_patch)
        )
        phase_patch = (
            phase_patch
            if isinstance(phase_patch, torch.Tensor)
            else torch.from_numpy(phase_patch)
        )
        probe = torch.from_numpy(probe) if probe is not None else None
        probe_position = (
            probe_position
            if isinstance(probe_position, torch.Tensor)
            else torch.from_numpy(probe_position)
        )

        if diffraction_amp.dim() == 2:
            diffraction_amp = diffraction_amp.unsqueeze(0)
        if amplitude_patch.dim() == 2:
            amplitude_patch = amplitude_patch.unsqueeze(0)
        if phase_patch.dim() == 2:
            phase_patch = phase_patch.unsqueeze(0)
        if probe is not None and probe.dim() == 2:
            probe = probe.unsqueeze(0).unsqueeze(0)
        if probe is not None and probe.dim() == 3:
            probe = probe.unsqueeze(0)

        return (
            diffraction_amp,
            amplitude_patch,
            phase_patch,
            probe,
            probe_position,
            self.normalization,
            self.scale,
        )
