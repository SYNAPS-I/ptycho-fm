"""Full-patch illumination weighting and compatibility of plotting entry points."""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ptycho_fm.training import Trainer
from ptycho_fm.utils.config import load_config
from ptycho_fm.utils.stitching import ProbeWeightedStitcher, probe_power
from scripts import run_inference_and_stitch as inference


class PatchDataset(Dataset):
    pattern_shape = (4, 6)
    object_shape = (6, 10)

    def __init__(self, metadata=False):
        self.metadata = metadata
        self.positions = torch.tensor([[2.5, 3.5], [2.5, 5.5]])
        self.probes = torch.zeros(2, 2, 3, 4, 6, dtype=torch.complex64)
        self.probes[:, 0, 0] = 0.1j
        self.probes[1, 0, 1] = 0.2
        self.probes[:, 0, :, 0, 0] = 0  # An unilluminated corner.
        self.probes[:, 1] = 100  # Higher OPR components are not forward illumination.

    def __len__(self):
        return 2

    def get_probe_positions(self):
        return self.positions

    def __getitem__(self, index):
        patch = torch.full((1, *self.pattern_shape), 2.0 + 6 * index)
        sample = (patch, patch * 2, -patch * 2, self.probes[index],
                  self.positions[index], torch.tensor(1.0), torch.tensor(1.0))
        return (*sample, {"index": index}) if self.metadata else sample

    def close(self):
        pass


class PatchModel(torch.nn.Module):
    def forward(self, diffraction, *_args):
        return diffraction, diffraction, -diffraction


def expected_object(dataset):
    numerator = torch.zeros(dataset.object_shape)
    denominator = torch.zeros_like(numerator)
    for index, left in enumerate((1, 3)):
        power = torch.full(dataset.pattern_shape, 0.01 if index == 0 else 0.05)
        power[0, 0] = 0
        numerator[1:5, left:left + 6] += (2 + 6 * index) * power
        denominator[1:5, left:left + 6] += power
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-20), 0)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("pad", [0, 32])
def test_full_patch_weighted_overlap_and_probe_scale(scale, pad):
    dataset = PatchDataset()
    stitcher = ProbeWeightedStitcher(dataset.object_shape, 2, pad=pad)
    for index in range(len(dataset)):
        patch = dataset[index][0]
        stitcher.add(dataset.positions[index:index + 1], (patch, -patch),
                     probe_power(dataset.probes[index:index + 1] * scale))
    amplitude, phase = stitcher.result()
    expected = expected_object(dataset)
    torch.testing.assert_close(amplitude, expected)
    torch.testing.assert_close(phase, -expected)
    assert amplitude[4, 8] == 8  # Retain the outermost illuminated patch pixels.
    assert amplitude[1, 1] == 0  # No contribution from higher OPR components.


def test_dark_probe_returns_finite_zero_objects():
    stitcher = ProbeWeightedStitcher((8, 8), 1)
    stitcher.add(torch.tensor([[3.125, 3.75]]), (torch.ones(1, 4, 4),),
                 torch.zeros(1, 4, 4))
    result, = stitcher.result()
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_fractional_positions_and_boundary_preserve_constant_object():
    positions = torch.tensor([[0.125, 0.75], [4.25, 5.875]])
    power = torch.linspace(0.1, 1, 24).reshape(1, 4, 6).expand(2, -1, -1)
    stitcher = ProbeWeightedStitcher((6, 8), 1)
    stitcher.add(positions, (torch.full_like(power, 3.0),), power)
    result, = stitcher.result()
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) > 0
    torch.testing.assert_close(result[result != 0], torch.full_like(result[result != 0], 3))


@pytest.mark.parametrize("batch_size,metadata", [(1, False), (2, True)])
def test_training_plot_uses_weighted_predictions_and_ground_truth(
    tmp_path, monkeypatch, batch_size, metadata,
):
    dataset = PatchDataset(metadata=metadata)
    images = []
    original_imshow = matplotlib.axes.Axes.imshow

    def capture(self, image, *args, **kwargs):
        images.append(torch.as_tensor(np.asarray(image)).clone())
        return original_imshow(self, image, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", capture)
    trainer = Trainer(PatchModel(), "supervised", "weighted", torch.device("cpu"), str(tmp_path))
    with pytest.warns(UserWarning, match="central_crop.*ignored"):
        trainer.generate_test_plot(DataLoader(dataset, batch_size=batch_size), 0,
                                   "object.png", central_crop=999, object_crop=0)
    expected = expected_object(dataset)
    for actual, reference in zip(images, (expected * 2, -expected * 2, expected, -expected), strict=True):
        torch.testing.assert_close(actual, reference)
    assert (tmp_path / "runweighted" / "object.png").is_file()


@pytest.mark.parametrize("crop", [-1, 999])
def test_inference_ignores_legacy_crop(tmp_path, monkeypatch, crop):
    dataset = PatchDataset()
    monkeypatch.setattr(inference, "predict_object_patches",
                        lambda model, batch, device: (batch[0].squeeze(1), -batch[0].squeeze(1)))
    with pytest.warns(UserWarning, match="central_crop.*ignored"):
        amplitude, phase = inference.run_inference_and_stitch(
            PatchModel(), DataLoader(dataset, batch_size=1), dataset.positions,
            dataset.object_shape, central_crop=crop,
        )
    torch.testing.assert_close(amplitude, expected_object(dataset))
    torch.testing.assert_close(phase, -expected_object(dataset))


@pytest.mark.parametrize("crop", [64, -1, 99999, None])
def test_old_config_crop_warns_without_size_validation(tmp_path, crop):
    path = tmp_path / "config.yaml"
    path.write_text(f"training:\n  test_plot_central_crop: {crop if crop is not None else 'null'}\n")
    with pytest.warns(UserWarning, match="test_plot_central_crop.*ignored"):
        config = load_config(path)
    assert config.training.test_plot_central_crop == crop


def test_new_config_does_not_warn(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("training: {}\n")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        load_config(path)
    assert not recorded


def test_cli_ignores_both_legacy_crop_options(tmp_path, monkeypatch):
    dataset = PatchDataset()
    data_path = tmp_path / "sample_dp.hdf5"
    data_path.touch()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training: {test_plot_central_crop: 999}\n")
    monkeypatch.setattr("sys.argv", [
        "run_inference_and_stitch.py", str(data_path), "--config", str(config_path),
        "--checkpoint", "unused.pth", "--output-dir", str(tmp_path),
        "--central-crop", "-1", "--object-crop", "0", "--normalization-value", "1",
        "--device", "cpu",
    ])
    monkeypatch.setattr(inference, "resolve_model_and_size", lambda config: (PatchModel(), 4))
    monkeypatch.setattr(inference, "load_checkpoint", lambda *_args: None)
    monkeypatch.setattr(inference, "build_dataloader",
                        lambda *_args, **_kwargs: (dataset, DataLoader(dataset, batch_size=1)))
    monkeypatch.setattr(inference, "predict_object_patches",
                        lambda model, batch, device: (batch[0].squeeze(1), -batch[0].squeeze(1)))
    with pytest.warns(UserWarning, match="ignored") as recorded:
        inference.main()
    assert len(recorded) == 2
    np.testing.assert_allclose(np.load(tmp_path / "pred_amp_object_sample.npy"), expected_object(dataset))
