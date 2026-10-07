"""Config lookup for the installed training entry point."""

import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from ptycho_fm import train
from ptycho_fm.utils import config as config_utils


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
    assert config["trainer"]["run_name"] is None
    assert "run_num" not in config["trainer"]
    assert config["mlflow"]["enabled"] is True
    assert "wandb" not in config
    assert {
        "dataset_name",
        "notes",
        "log_parameters",
        "log_metrics",
        "log_artifacts",
        "log_system_metrics",
        "log_every_n_batches",
    } <= config["tracking"].keys()
    assert "# wandb:" in config_path.read_text()


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

    monkeypatch.setattr(config_utils.yaml, "safe_load", read_config)
    with pytest.raises(ConfigRead):
        train.main()


def test_run_name_is_generated_once_and_legacy_run_num_is_supported():
    config = {"trainer": {"run_name": None}, "paths": {"model_save_path": "/models"}}
    moment = datetime(2026, 9, 23, 14, 35, 12, tzinfo=UTC)
    assert config_utils.resolve_run_name(config, generate=True, now=moment) == "20260923-143512"
    assert config_utils.run_directory(config) == Path("/models/run20260923-143512")
    assert config_utils.resolve_run_name(config, generate=True) == "20260923-143512"

    legacy = {"trainer": {"run_num": 7}}
    assert config_utils.resolve_run_name(legacy) == "7"


def test_resume_requires_an_explicit_run_name():
    with pytest.raises(ValueError, match="required when resuming"):
        config_utils.resolve_run_name({"trainer": {"run_name": None}})


def test_launcher_run_name_defaults_only_an_unnamed_fresh_run():
    config = {
        "training": {"resume_from_checkpoint": False},
        "trainer": {"run_name": None},
    }
    train.apply_launcher_run_name(config, "20260924-191504")
    assert config["trainer"]["run_name"] == "20260924-191504"

    explicit = {
        "training": {"resume_from_checkpoint": False},
        "trainer": {"run_name": "deliberate-name"},
    }
    train.apply_launcher_run_name(explicit, "20260924-191504")
    assert explicit["trainer"]["run_name"] == "deliberate-name"

    resume = {
        "training": {"resume_from_checkpoint": True},
        "trainer": {"run_name": "existing-run"},
    }
    train.apply_launcher_run_name(resume, "20260924-191504")
    assert resume["trainer"]["run_name"] == "existing-run"
