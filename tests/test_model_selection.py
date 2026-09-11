"""Current-model inference and checkpoint compatibility checks."""

from pathlib import Path

import pytest
import torch
import yaml

from ptycho_fm.model.model import PtychoFM
from scripts.run_inference_and_stitch import (
    build_dataloader,
    load_checkpoint,
    resolve_model_and_size,
    run_inference_and_stitch,
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
        expected = trained_model(*inputs)
        actual = model(*inputs)

    for result, reference in zip(actual, expected, strict=True):
        assert result.shape == (2, 1, img_size, img_size)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference)


@pytest.mark.parametrize("budget", [0, 512])
def test_inference_data_settings(config, monkeypatch, budget):
    class DatasetStub:
        def __init__(self, path, **kwargs):
            self.settings = kwargs

        def __len__(self):
            return 1

    monkeypatch.setattr(
        "scripts.run_inference_and_stitch.PtychographyDataset", DatasetStub
    )
    config["data"] = {"max_OPR_modes": 3, "cache_memory_budget_mb": budget}
    dataset, loader = build_dataloader("unused.hdf5", config, 25.0, batch_size=2)

    assert "target_size" not in dataset.settings
    assert dataset.settings["max_OPR_modes"] == 3
    assert dataset.settings["cache_memory_budget_mb"] == budget
    assert dataset.normalization == 25.0
    assert loader.dataset is dataset


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
    loss = sum((output - target).square().mean()
               for output, target in zip(outputs, (diff, amp, phase), strict=True))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    stitched = run_inference_and_stitch(
        model, loader, dataset.get_probe_positions(), dataset.object_shape,
        central_crop=8, pad=1, device=torch.device("cpu"),
    )
    for image in stitched:
        assert image.shape == dataset.object_shape
        assert torch.isfinite(image).all()
    assert dataset._cached_probe_positions is None
