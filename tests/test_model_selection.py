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
)


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


@pytest.mark.parametrize("target_size", [None, 64])
def test_inference_data_size(config, monkeypatch, target_size):
    class DatasetStub:
        def __init__(self, path, **kwargs):
            self.target_size = kwargs["target_size"]

        def __len__(self):
            return 1

    monkeypatch.setattr(
        "scripts.run_inference_and_stitch.PtychographyDataset", DatasetStub
    )
    config["data"] = {"target_size": target_size}
    dataset, loader = build_dataloader("unused.hdf5", config, 25.0, batch_size=2)

    assert dataset.target_size == (32 if target_size is None else target_size)
    assert dataset.normalization == 25.0
    assert loader.dataset is dataset


def test_polaris_model_config_uses_current_schema():
    config_path = Path(__file__).resolve().parents[1] / "configs/polaris/config.yaml"
    with config_path.open() as f:
        model_config = yaml.safe_load(f)["model"]

    assert model_config["encoder_type"] == "custom"
    assert model_config["encoder"]["img_size"] == 256
    assert model_config["encoder"]["embed_dim"] == model_config["decoder"]["latent_dim"]
