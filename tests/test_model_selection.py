"""Current-model inference and checkpoint compatibility checks."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ptycho_fm.model.model import (
    PtychoFM,
    PtychoFMCoupledAmpPh,
    PtychoFMInference,
    PtychoFMReIm,
)
from scripts.run_inference_and_stitch import (
    build_dataloader,
    load_checkpoint,
    load_normalization_map,
    resolve_model_and_size,
    resolve_normalization,
    run_inference_and_stitch,
    save_stitched_outputs,
)

from .test_data_refactor import make_pair


@pytest.fixture
def config():
    return {
        "model": {
            "encoder_type": "custom",
            "encoder": {
                "img_size": 32,
                "patch_size": 4,
                "embed_dim": 16,
                "depth": 1,
                "num_heads": 2,
                "dropout": 0.0,
            },
            "decoder": {
                "base_channels": 4,
                "num_stages": 2,
                "use_batchnorm": False,
            },
        },
    }


def test_inference_checkpoint_round_trip(config, tmp_path):
    trained_model = PtychoFM(config=config["model"]).eval()
    checkpoint = tmp_path / "model.pth"
    torch.save(trained_model.state_dict(), checkpoint)

    model, img_size = resolve_model_and_size(config)
    load_checkpoint(model, checkpoint, torch.device("cpu"))
    model.eval()
    assert img_size == 32

    inputs = (
        torch.rand(2, 1, img_size, img_size),
        torch.randn(2, 1, 2, img_size, img_size, 2),
        torch.ones(2),
        torch.ones(2),
    )
    with torch.no_grad():
        _expected_diff, expected_amp, expected_phase = trained_model(*inputs)
        actual_amp, actual_phase = model(inputs[0])

    for result, reference in zip(
        (actual_amp, actual_phase),
        (expected_amp, expected_phase),
        strict=True,
    ):
        assert result.shape == (2, 1, img_size, img_size)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference)


@pytest.mark.parametrize(
    "mode, expected_type",
    [
        (None, PtychoFMInference),
        ("real_imag", PtychoFMReIm),
        ("amp_phase", PtychoFMCoupledAmpPh),
    ],
)
def test_inference_model_selection(config, mode, expected_type):
    config["model"]["coupled_decoder_mode"] = mode

    model, img_size = resolve_model_and_size(config)

    assert isinstance(model, expected_type)
    assert img_size == 32


@pytest.mark.parametrize("budget", [0, 512])
def test_inference_data_settings(config, monkeypatch, budget):
    class DatasetStub:
        def __init__(self, path, **kwargs):
            self.settings = kwargs

        def __len__(self):
            return 1

    monkeypatch.setattr("ptycho_fm.utils.inference.PtychographyDataset", DatasetStub)
    config["data"] = {"max_OPR_modes": 3, "cache_memory_budget_mb": budget}
    dataset, loader = build_dataloader("unused.hdf5", config, 25.0, batch_size=2)

    assert "target_size" not in dataset.settings
    assert dataset.settings["max_OPR_modes"] == 3
    assert dataset.settings["cache_memory_budget_mb"] == budget
    assert dataset.normalization == 25.0
    assert loader.dataset is dataset
    assert loader.num_workers == 0
    assert not loader.pin_memory


def test_polaris_model_config_uses_current_schema():
    config_path = Path(__file__).resolve().parents[1] / "configs/polaris/config.yaml"
    with config_path.open() as f:
        model_config = yaml.safe_load(f)["model"]

    assert model_config["encoder_type"] == "custom"
    assert model_config["encoder"]["img_size"] == 256
    assert model_config["encoder"]["embed_dim"] == model_config["decoder"]["latent_dim"]


def test_native_loader_model_and_stitching(config, tmp_path):
    path, *_ = make_pair(tmp_path, opr=2, pattern_shape=(32, 32))
    config["data"] = {"scale": 7.0, "max_OPR_modes": 3, "cache_memory_budget_mb": 1}
    dataset, loader = build_dataloader(str(path), config, 10.0, batch_size=2)
    model = PtychoFM(config=config["model"])
    diff, amp, phase, probe, _, norm, scale = next(iter(loader))
    assert probe.shape == (2, 3, 8, 32, 32)
    outputs = model(diff, torch.view_as_real(probe), norm, scale)
    loss = sum(
        (output - target).square().mean()
        for output, target in zip(outputs, (diff, amp, phase), strict=True)
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    inference_model, _ = resolve_model_and_size(config)
    inference_model.load_state_dict(model.state_dict())
    stitched = run_inference_and_stitch(
        inference_model,
        loader,
        dataset.get_probe_positions(),
        dataset.object_shape,
        central_crop=8,
        pad=1,
        device=torch.device("cpu"),
    )
    for image in stitched:
        assert image.shape == dataset.object_shape
        assert torch.isfinite(image).all()
    assert dataset._cached_probe_positions is None


@pytest.mark.parametrize("mode", ["real_imag", "amp_phase"])
def test_all_current_models_stitch_to_amp_and_phase(config, tmp_path, mode):
    path, *_ = make_pair(tmp_path, opr=2, pattern_shape=(32, 32))
    config["model"]["coupled_decoder_mode"] = mode
    config["data"] = {"scale": 7.0, "max_OPR_modes": 3}
    dataset, loader = build_dataloader(path, config, 10.0, batch_size=2)
    model, _ = resolve_model_and_size(config)

    stitched = run_inference_and_stitch(
        model,
        loader,
        dataset.get_probe_positions(),
        dataset.object_shape,
        central_crop=8,
        pad=1,
        device=torch.device("cpu"),
    )

    assert len(stitched) == 2
    for image in stitched:
        assert image.shape == dataset.object_shape
        assert torch.isfinite(image).all()


def test_normalization_resolution_and_output_format(tmp_path):
    path, _para_path, diffraction, *_ = make_pair(tmp_path, name="coins")
    normalization_path = tmp_path / "normalization.pkl"
    normalization_path.write_bytes(pickle.dumps({"coins": 17.0}))
    normalization_map = load_normalization_map(normalization_path)

    assert resolve_normalization(path, normalization_value=11.0) == 11.0
    assert resolve_normalization(path, normalization_map=normalization_map) == 17.0
    assert resolve_normalization(path) == float(diffraction.max())

    amp_path, phase_path = save_stitched_outputs(
        tmp_path / "results",
        "coins",
        torch.ones(32, 40),
        torch.full((32, 40), 2.0),
        object_crop=4,
    )
    assert amp_path.name == "pred_amp_object_coins.npy"
    assert phase_path.name == "pred_ph_object_coins.npy"
    amp = np.load(amp_path)
    phase = np.load(phase_path)
    assert amp.shape == phase.shape == (24, 32)
    assert amp.dtype == phase.dtype == np.float32
    np.testing.assert_array_equal(amp, 1.0)
    np.testing.assert_array_equal(phase, 2.0)
