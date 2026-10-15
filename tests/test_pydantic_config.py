"""Compatibility of the typed API with existing YAML and runtime consumers."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from pydantic import BaseModel, ValidationError

from ptycho_fm import train
from ptycho_fm.config import PtychoFMConfig
from ptycho_fm.model.model import PtychoFM, PtychoFMCoupledAmpPh, PtychoFMInference, PtychoFMReIm
from ptycho_fm.utils.config import deep_merge_config, load_config, resolve_run_name, run_directory
from ptycho_fm.utils.inference import load_config as load_inference_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / 'config.yaml', *sorted((ROOT / 'configs').rglob('*.yaml'))]


def legacy_load(path):
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop('extends', None)
    return deep_merge_config(legacy_load(path.parent / parent), raw) if parent else raw


def assert_same_values(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_values(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_same_values(left, right)
    elif isinstance(expected, str) and isinstance(actual, float):
        # PyYAML treats forms such as 1e-6 as strings; Pydantic parses them.
        assert actual == float(expected)
    else:
        assert actual == expected


@pytest.mark.parametrize('path', CONFIGS, ids=lambda path: str(path.relative_to(ROOT)))
def test_repository_yaml_preserves_keys_and_values(path):
    config = load_config(path)
    assert isinstance(config, BaseModel)
    assert isinstance(config.model.encoder, BaseModel)
    raw = config.model_dump(exclude_unset=True)
    assert_same_values(raw, legacy_load(path))
    # YAML/JSON/checkpoint consumers must receive plain builtins, not models.
    assert yaml.safe_load(yaml.safe_dump(raw)) == raw
    assert PtychoFMConfig.model_validate_json(config.model_dump_json(exclude_unset=True)) == config


def test_aliases_schema_and_empty_config(tmp_path):
    path = tmp_path / 'empty.yaml'
    path.write_text('')
    assert train.load_yaml_config is load_config
    assert load_inference_config is load_config
    assert load_config(path).model_dump(exclude_unset=True) == {}
    assert 'model' in PtychoFMConfig.model_json_schema()['properties']


def test_inheritance_validates_only_final_values_and_preserves_extensions(tmp_path):
    (tmp_path / 'base.yaml').write_text('training: {batch_size: replace-me}\n')
    path = tmp_path / 'child.yaml'
    path.write_text('extends: base.yaml\ntraining: {batch_size: 4, learning_rate: 1e-5}\n'
                    'model: {encoder: {custom_option: 7}}\n_dashboard_runtime: integration-v1\n')
    config = load_config(path)
    assert config.training.batch_size == 4
    assert config.training.learning_rate == 1e-5
    assert config.model.encoder.model_extra == {'custom_option': 7}
    assert config.model_dump(exclude_unset=True)['_dashboard_runtime'] == 'integration-v1'
    path.write_text('training: {batch_size: invalid}\n')
    with pytest.raises(ValidationError, match=r'training.batch_size'):
        load_config(path)
    with pytest.raises(ValidationError):
        config.training.batch_size = 'invalid'


def test_omitted_null_and_legacy_values_are_distinct():
    config = PtychoFMConfig.model_validate({
        'data': {'test_path': None, 'max_OPR_modes': 3},
        'training': {'batch_size': 4, 'validation_plot_num_objects': None},
        'model': {'encoder_type': 'pretrained', 'scaling': {'log_scale_amp': -2.0}},
        'trainer': {'run_num': 7}, 'paths': {'model_save_path': '/models'},
    })
    raw = config.model_dump(exclude_unset=True)
    assert raw['data'] == {'test_path': None, 'max_OPR_modes': 3}
    assert 'valid_batch_size' not in raw['training']
    assert 'encoder_lr' not in raw['training']
    assert 'encoder' not in raw['model']
    assert 'output_norm' not in raw['model']
    assert config.training.validation_plot_num_objects is None
    assert resolve_run_name(config) == '7'
    assert config.trainer.run_name == '7'
    assert run_directory(config) == Path('/models/run7')
    assert config.model_dump(exclude_unset=True)['trainer']['run_name'] == '7'


def test_nullable_backend_and_scheduler_options():
    raw = {'tracking': None, 'wandb': None, 'mlflow': {'tags': None},
           'training': {'lr_scheduler': {'kwargs': None}}}
    assert PtychoFMConfig.model_validate(raw).model_dump(exclude_unset=True) == raw


def test_typed_cli_overrides_missing_sections_and_launcher():
    config = PtychoFMConfig()
    args = SimpleNamespace(data_path='~/data', test_path='~/test.hdf5',
                           finetune_from_model='~/weights.pth', model_save_path='~/models')
    train.apply_cli_path_overrides(config, args)
    train.apply_launcher_run_name(config, 'launch-name')
    raw = config.model_dump(exclude_unset=True)
    assert raw['data']['data_path'] == str(Path.home() / 'data')
    assert raw['data']['normalization_dict_path'] == str(Path.home() / 'data/normalization.pkl')
    assert raw['data']['test_normalization'] == raw['data']['normalization_dict_path']
    assert raw['data']['test_path'] == str(Path.home() / 'test.hdf5')
    assert raw['training']['finetune_from_model'] == str(Path.home() / 'weights.pth')
    assert raw['paths']['model_save_path'] == str(Path.home() / 'models')
    assert raw['trainer']['run_name'] == 'launch-name'
    train.apply_launcher_run_name(config, 'other-name')
    assert config.trainer.run_name == 'launch-name'
    config.trainer.run_name = None
    config.training.resume_from_checkpoint = True
    train.apply_launcher_run_name(config, 'other-name')
    assert config.trainer.run_name is None


@pytest.mark.parametrize('model_class', [PtychoFM, PtychoFMInference, PtychoFMReIm, PtychoFMCoupledAmpPh])
def test_model_checkpoint_and_export_boundary_preserves_defaults(model_class, tmp_path):
    raw = {'encoder': {'img_size': 32, 'patch_size': 4, 'embed_dim': 16,
                       'depth': 1, 'num_heads': 2, 'dropout': 0.0},
           'decoder': {'base_channels': 4, 'num_stages': 2, 'use_batchnorm': False},
           'scaling': {'log_scale_amp': -2.0}}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'model': raw}))
    original = model_class(config=raw).eval()
    # This is also the migration required by the downstream ONNX exporter.
    model_config = load_config(path).model.model_dump(exclude_unset=True)
    restored = model_class(config=model_config).eval()
    checkpoint = tmp_path / 'weights.pth'
    torch.save(original.state_dict(), checkpoint)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    x = torch.rand(1, 1, 32, 32)
    args = (x,) if model_class is PtychoFMInference else (
        x, torch.randn(1, 1, 1, 32, 32, 2), torch.ones(1), torch.ones(1))
    with torch.no_grad():
        for actual, expected in zip(restored(*args), original(*args), strict=True):
            torch.testing.assert_close(actual, expected)
