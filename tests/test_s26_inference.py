"""S26 raw-data preprocessing and integer-grid stitching checks."""

import h5py
import numpy as np
import pytest
import torch

from ptycho_fm.utils.inference import make_inference_dataloader
from scripts.s26.run_flyscan_inference import (
    RawH5FlyscanDataset,
    compute_raw_data_max,
    get_crop_sizes,
    stitch_grid_patches,
)


def constant_patches(values, shape=(2, 2)):
    return torch.stack([torch.full(shape, value) for value in values])


@pytest.mark.parametrize(
    "scan_pattern, expected_values",
    [
        ("raster", [[1, 2], [3, 4]]),
        ("zigzag", [[1, 2], [4, 3]]),
    ],
)
def test_grid_scan_order(scan_pattern, expected_values):
    amplitude = constant_patches([1.0, 2.0, 3.0, 4.0])

    stitched_amplitude, stitched_phase = stitch_grid_patches(
        amplitude,
        amplitude * 10,
        ny=2,
        nx=2,
        step_size=2,
        scan_pattern=scan_pattern,
        flip_patch_y=False,
    )

    expected = (
        torch.tensor(expected_values, dtype=torch.float32)
        .repeat_interleave(2, 0)
        .repeat_interleave(2, 1)
    )
    torch.testing.assert_close(stitched_amplitude, expected)
    torch.testing.assert_close(stitched_phase, expected * 10)


def test_grid_overlap_is_averaged():
    amplitude = constant_patches([1.0, 3.0])

    stitched_amplitude, _ = stitch_grid_patches(
        amplitude,
        amplitude,
        ny=1,
        nx=2,
        step_size=1,
        scan_pattern="raster",
        flip_patch_y=False,
    )

    torch.testing.assert_close(
        stitched_amplitude, torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    )


def test_grid_patch_vertical_flip():
    patch = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    stitched, _ = stitch_grid_patches(
        patch,
        patch,
        ny=1,
        nx=1,
        step_size=1,
        flip_patch_y=True,
    )

    torch.testing.assert_close(stitched, torch.tensor([[3.0, 4.0], [1.0, 2.0]]))


def test_raw_h5_crop_binning_and_loader_settings(tmp_path):
    path = tmp_path / "scan.h5"
    frames = np.arange(16, dtype=np.uint16).reshape(1, 4, 4)
    with h5py.File(path, "w") as file:
        file.create_dataset("/entry/data/data", data=frames)

    assert (
        compute_raw_data_max(
            path,
            (1, 2, 0, 3),
            dataset_path="/entry/data/data",
            binning=2,
        )
        == 46.0
    )

    dataset = RawH5FlyscanDataset(
        path,
        (1, 2, 0, 3),
        normalization=2.0,
        scale=2.0,
        binning=2,
    )
    loader = make_inference_dataloader(dataset, batch_size=1)
    diffraction, amplitude, phase, probe, _position, normalization, scale = dataset[0]

    assert dataset.pattern_shape == (2, 1)
    torch.testing.assert_close(
        diffraction.square().squeeze(), torch.tensor([14.0, 46.0])
    )
    assert amplitude.shape == phase.shape == diffraction.shape == (1, 2, 1)
    assert probe.shape == (1, 1, 2, 1)
    assert probe.is_complex()
    assert normalization.item() == 2.0
    assert scale.item() == 2.0
    assert loader.num_workers == 0
    assert not loader.pin_memory
    dataset.close()


def test_crop_size_range_is_inclusive():
    assert get_crop_sizes(9, None, 1) == [9]
    assert get_crop_sizes(9, (2, 6), 2) == [2, 4, 6]
