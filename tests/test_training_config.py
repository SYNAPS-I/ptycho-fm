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


@pytest.mark.parametrize('spelling', ['hyphen', 'underscore'])
def test_cli_paths_override_inherited_config_before_training(tmp_path, monkeypatch, spelling):
    base = {
        'data': {
            'data_path': '/old/data',
            'test_path': '/old/test/scan_dp.hdf5',
            'normalization_dict_path': '/old/data/train_norm.pkl',
            'test_normalization': '/old/test/test_norm.pkl',
        },
        'training': {'finetune_from_model': '/old/model.pth'},
        'paths': {'model_save_path': '/old/models'},
    }
    (tmp_path / 'base.yaml').write_text(yaml.safe_dump(base))
    (tmp_path / 'config.yaml').write_text('extends: base.yaml\n')
    overrides = {
        'data_path': 'new/data',
        'test_path': 'new/test/scan_dp.hdf5',
        'finetune_from_model': 'new/model.pth',
        'model_save_path': 'new/models',
    }
    argv = ['ptycho-fm-train']
    for key, value in overrides.items():
        flag = key.replace('_', '-') if spelling == 'hyphen' else key
        argv.extend(['--' + flag, value])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', argv)

    class ConfigReady(Exception):
        pass

    def check_config(config):
        assert config['data'] == {
            'data_path': overrides['data_path'],
            'test_path': overrides['test_path'],
            'normalization_dict_path': 'new/data/normalization.pkl',
            'test_normalization': 'new/data/normalization.pkl',
        }
        assert config['training']['finetune_from_model'] == overrides['finetune_from_model']
        assert config['paths']['model_save_path'] == overrides['model_save_path']
        # Stop before distributed initialization, data loading, or tracking.
        raise ConfigReady

    monkeypatch.setattr(train, 'resolve_tracking_config', check_config)
    with pytest.raises(ConfigReady):
        train.main()
    assert config_utils.load_config(tmp_path / 'base.yaml') == base


@pytest.mark.parametrize('override', [None, 'data_path', 'test_path', 'finetune_from_model', 'model_save_path'])
def test_path_overrides_are_independent_and_optional(override):
    config = {
        'data': {
            'data_path': '/old/data',
            'test_path': '/old/test/scan_dp.hdf5',
            'normalization_dict_path': '/old/data/norm.pkl',
            'test_normalization': '/old/test/test_norm.pkl',
        },
        'training': {'finetune_from_model': '/old/model.pth'},
        'paths': {'model_save_path': '/old/models'},
    }
    args = SimpleNamespace(
        data_path=None, test_path=None, finetune_from_model=None, model_save_path=None
    )
    if override is not None:
        setattr(args, override, '/new/location')
    train.apply_cli_path_overrides(config, args)

    assert config['data']['data_path'] == ('/new/location' if override == 'data_path' else '/old/data')
    assert config['data']['test_path'] == ('/new/location' if override == 'test_path' else '/old/test/scan_dp.hdf5')
    assert config['data']['normalization_dict_path'] == (
        '/new/location/normalization.pkl' if override == 'data_path' else '/old/data/norm.pkl'
    )
    assert config['data']['test_normalization'] == (
        '/new/location/normalization.pkl' if override == 'data_path' else '/old/test/test_norm.pkl'
    )
    assert config['training']['finetune_from_model'] == (
        '/new/location' if override == 'finetune_from_model' else '/old/model.pth'
    )
    assert config['paths']['model_save_path'] == (
        '/new/location' if override == 'model_save_path' else '/old/models'
    )


@pytest.mark.parametrize('normalization', ['norm.pkl', 'metadata/norm.pkl', '/old/norm.pkl', None])
@pytest.mark.parametrize('test_path', [None, '/new/test/scan_dp.hdf5'])
def test_normalization_paths_always_use_cli_training_directory(normalization, test_path):
    config = {'data': {
        'normalization_dict_path': normalization, 'test_normalization': normalization,
    }}
    args = SimpleNamespace(
        data_path='/new/train', test_path=test_path,
        finetune_from_model=None, model_save_path=None,
    )
    train.apply_cli_path_overrides(config, args)

    assert config['data']['normalization_dict_path'] == '/new/train/normalization.pkl'
    assert config['data']['test_normalization'] == '/new/train/normalization.pkl'


def test_cli_paths_can_supply_missing_settings_and_expand_home():
    home = Path.home()
    config = {}
    args = SimpleNamespace(
        data_path='~/data', test_path='~/test/scan_dp.hdf5',
        finetune_from_model='~/model.pth', model_save_path='~/models',
    )
    train.apply_cli_path_overrides(config, args)

    assert config == {
        'data': {
            'data_path': str(home / 'data'),
            'test_path': str(home / 'test/scan_dp.hdf5'),
            'normalization_dict_path': str(home / 'data/normalization.pkl'),
            'test_normalization': str(home / 'data/normalization.pkl'),
        },
        'training': {'finetune_from_model': str(home / 'model.pth')},
        'paths': {'model_save_path': str(home / 'models')},
    }


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
