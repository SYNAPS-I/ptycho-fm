"""Config lookup for the installed training entry point."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from ptycho_fm import train


@pytest.mark.parametrize(
    ("output_norm", "scale_keys", "offset_keys"),
    [
        (
            {"amp_scale": 0.4, "ph_scale": 2.5, "amp_offset": 0.9},
            ("amp_scale", "ph_scale"),
            ("amp_offset",),
        ),
        (
            {
                "real_scale": 0.2,
                "imag_scale": 0.3,
                "real_offset": 1.1,
                "imag_offset": -0.1,
            },
            ("real_scale", "imag_scale"),
            ("real_offset", "imag_offset"),
        ),
    ],
)
def test_finetuning_restores_configured_output_norm(
    output_norm, scale_keys, offset_keys
):
    model = SimpleNamespace(
        **{
            key: torch.nn.Parameter(torch.tensor(99.0), requires_grad=False)
            for key in output_norm
        }
    )

    overridden = train.override_output_norm_from_config(
        model, {"output_norm": output_norm}
    )

    assert overridden == output_norm
    for key in scale_keys:
        assert getattr(model, key).item() == pytest.approx(math.log(output_norm[key]))
    for key in offset_keys:
        assert getattr(model, key).item() == pytest.approx(output_norm[key])


def test_default_config_contains_merged_training_options():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with config_path.open() as stream:
        config = yaml.safe_load(stream)

    assert {
        "encoder_lr",
        "amp_decoder_lr",
        "ph_decoder_lr",
        "coupled_decoder_lr",
        "combined_loss",
        "probe_aware_loss",
        "q_dependent_loss",
        "finetune_from_model",
    } <= config["training"].keys()
    assert {"decoder_type", "coupled_decoder_mode", "output_norm"} <= config[
        "model"
    ].keys()
    assert config["data"]["apply_noise"] is True
    assert config["data"]["test_apply_noise"] is False


@pytest.mark.parametrize("config_option", ["default", "relative", "absolute"])
def test_training_reads_config_from_working_directory(tmp_path, monkeypatch, config_option):
    config_path = tmp_path / ("config.yaml" if config_option == "default" else "custom.yaml")
    config_path.write_text("marker: selected config\n")
    args = ["ptycho-fm-train"]
    if config_option != "default":
        args += ["--config", str(config_path) if config_option == "absolute" else config_path.name]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", args)

    class ConfigRead(Exception):
        pass

    def read_config(stream):
        assert stream.name == str(config_path)
        assert stream.read() == "marker: selected config\n"
        # Stop before distributed initialization or training.
        raise ConfigRead

    monkeypatch.setattr(train.yaml, "safe_load", read_config)
    with pytest.raises(ConfigRead):
        train.main()
