"""Tests for TiledPtychographyDataset.

The class reads from a Tiled run container with a specific shape:

    <run>/diffraction/dp                       (nz, H, W) uint8 amplitude
    <run>/diffraction/probe_position_x_m       (nz,) float64
    <run>/diffraction/probe_position_y_m       (nz,) float64
    <run>/final/object                          (1, H_obj, W_obj) complex
    <run>/final/probe                           (N_modes, H, W) complex
    <run>.metadata['x_pixel_m']                 float

Rather than spin up an in-memory Tiled server, these tests replace
``tiled.client.from_uri`` with a fake whose subscript / metadata semantics
match what the dataset reads. That isolates the test to the schema mapping
and ``__getitem__`` contract — exactly what could regress.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

# Add parent directory to path so `import data_tiled` works.
sys.path.insert(0, str(Path(__file__).parent.parent))


class _FakeArray:
    """Numpy-backed stand-in for a Tiled array node."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    @property
    def shape(self):
        return self._arr.shape

    def __getitem__(self, idx):
        return self._arr[idx]


class _FakeContainer:
    """Dict-of-_FakeArray stand-in for a Tiled container node."""

    def __init__(self, children: dict, metadata: dict | None = None):
        self._children = children
        self.metadata = metadata or {}

    def __getitem__(self, key):
        return self._children[key]


def _build_fake_run(num_patterns=8, pattern_size=128, object_size=512, n_modes=4):
    """Construct a fake Tiled run with the schema TiledPtychographyDataset reads."""
    rng = np.random.default_rng(seed=0)

    # uint8 amplitudes (= sqrt of intensity) — what holoptycho writes to
    # halve the on-the-wire write volume.
    dp = (rng.random((num_patterns, pattern_size, pattern_size)) * 255).astype(
        np.uint8
    )
    # Positions in meters: range chosen so the meters-vs-pixels heuristic in
    # _cache_positions classifies them as meters (range ~ 10x pixel_size_m).
    pos_x_m = np.linspace(0.0, 1e-5, num_patterns).astype(np.float64)
    pos_y_m = np.linspace(0.0, 1e-5, num_patterns).astype(np.float64)
    obj = (
        rng.random((1, object_size, object_size))
        + 1j * rng.random((1, object_size, object_size))
    ).astype(np.complex64)
    probe = (
        rng.random((n_modes, pattern_size, pattern_size))
        + 1j * rng.random((n_modes, pattern_size, pattern_size))
    ).astype(np.complex64)

    diffraction = _FakeContainer({
        "dp": _FakeArray(dp),
        "probe_position_x_m": _FakeArray(pos_x_m),
        "probe_position_y_m": _FakeArray(pos_y_m),
    })
    final = _FakeContainer({
        "object": _FakeArray(obj),
        "probe": _FakeArray(probe),
    })
    return _FakeContainer(
        {"diffraction": diffraction, "final": final},
        metadata={
            "scan_id": "test_scan_42",
            "x_pixel_m": 1e-7,
            "fine_tune": True,
        },
    )


def test_tiled_dataset_returns_expected_tuple():
    """TiledPtychographyDataset.__getitem__ matches the 7-tuple contract of
    PtychographyDataset and shapes line up with the requested target_size."""
    fake_run = _build_fake_run(num_patterns=8, pattern_size=128, object_size=512)

    with patch("tiled.client.from_uri", return_value=fake_run):
        from data_tiled import TiledPtychographyDataset

        ds = TiledPtychographyDataset(
            tiled_uri="http://fake/run-uid/",
            apply_noise=False,  # deterministic for assertions
            max_probe_modes=8,
            target_size=128,
        )

    assert len(ds) == 8

    sample = ds[0]
    assert len(sample) == 7, f"expected 7-tuple, got {len(sample)}"
    diff_amp, amp_patch, ph_patch, probe, probe_pos, normalization, scale = sample

    # diff_amp: torch tensor, shape (1, target_size, target_size), float
    assert isinstance(diff_amp, torch.Tensor)
    assert diff_amp.shape == (1, 128, 128)
    assert diff_amp.dtype == torch.float32

    # amplitude / phase patches: shape (1, target_size, target_size)
    assert amp_patch.shape == (1, 128, 128)
    assert ph_patch.shape == (1, 128, 128)

    # probe: padded to max_probe_modes, shape (1, max_probe_modes, H, W)
    assert probe.shape == (1, 8, 128, 128)

    # probe_pos: 2 coordinates
    assert probe_pos.shape == (2,)

    # normalization & scale: scalars from the loader
    assert isinstance(normalization, float)
    assert isinstance(scale, float)


def test_tiled_dataset_reads_pixel_size_from_metadata():
    """The pixel size needed for meters-to-pixels conversion is read from
    container metadata, not from an array attribute as in the h5 schema."""
    fake_run = _build_fake_run()

    with patch("tiled.client.from_uri", return_value=fake_run):
        from data_tiled import TiledPtychographyDataset

        ds = TiledPtychographyDataset(
            tiled_uri="http://fake/run-uid/",
            apply_noise=False,
            target_size=128,
        )
        # Trigger position cache (lazy in __init__).
        _ = ds[0]

    assert ds.pixel_size_m == 1e-7


def test_tiled_dataset_object_name_from_metadata():
    """When object_name is not supplied, fall back to scan_id from metadata."""
    fake_run = _build_fake_run()

    with patch("tiled.client.from_uri", return_value=fake_run):
        from data_tiled import TiledPtychographyDataset

        ds = TiledPtychographyDataset(
            tiled_uri="http://fake/run-uid/",
            apply_noise=False,
            target_size=128,
        )

    assert ds.object_name == "test_scan_42"
