"""Analytical accounting is checked against current modules and known costs."""

import copy

import pytest

from ptycho_fm.model.model import PtychoFM
from ptycho_fm.utils.flops import (
    PtychoFMFlopsCalculator,
    batch_norm2d_flops,
    decoder256_flops,
)


@pytest.fixture
def model_config():
    return {
        'encoder': {'img_size': 16, 'patch_size': 4, 'embed_dim': 16,
                    'num_heads': 2, 'depth': 1, 'dropout': 0.0},
        'decoder': {'base_channels': 4, 'num_stages': 2, 'dropout': 0.0},
    }


@pytest.mark.parametrize('batchnorm', [True, False])
@pytest.mark.parametrize('cls_token', [True, False])
def test_parameters_match_current_model(model_config, batchnorm, cls_token):
    model_config['decoder']['use_batchnorm'] = batchnorm
    model_config['encoder']['use_cls_token'] = cls_token
    model = PtychoFM(model_config)
    calc = PtychoFMFlopsCalculator(model_config, batch_size=1)
    assert round(calc.param_count() * 1e6) == sum(p.numel() for p in model.parameters())


def test_bn_training_statistics_and_batch_dependence():
    # Two channels, eight values/channel: (8*8+10)*2 operations.
    assert batch_norm2d_flops(2, 2, 2, 2) * 1e12 == pytest.approx(148)
    assert batch_norm2d_flops(2, 2, 2, 2, training=False) * 1e12 == pytest.approx(68)
    assert batch_norm2d_flops(2, 2, 2, 2) < 2 * batch_norm2d_flops(1, 2, 2, 2)


def test_custom_activation_counts_two_extra_multiplies():
    settings = {'batch': 2, 'grid0': 4, 'num_stages': 2, 'base_channels': 4, 'latent_dim': 16}
    difference = (decoder256_flops(**settings, output_activation='custom')
                  - decoder256_flops(**settings, output_activation='tanh'))
    assert difference * 1e12 == pytest.approx(2 * 2 * 16 * 16)


def test_training_cost_uses_actual_batch(model_config):
    calc = PtychoFMFlopsCalculator(model_config, batch_size=1)
    assert calc.training_tflops(3) == 3 * calc.flops_analytical(batch_size=3)
    assert calc.training_tflops(3) < 3 * calc.training_tflops(1)


@pytest.mark.parametrize('change', [
    {'encoder_type': 'pretrained'}, {'decoder_type': 'resnet'},
    {'coupled_decoder_mode': 'real_imag'}, {'coupled_decoder_mode': 'amp_phase'},
])
def test_unsupported_models_are_explicit(model_config, change):
    model_config.update(change)
    with pytest.raises(NotImplementedError):
        PtychoFMFlopsCalculator(model_config, batch_size=1)


def test_config_not_mutated_and_defaults_match_model(model_config):
    original = copy.deepcopy(model_config)
    calc = PtychoFMFlopsCalculator(model_config, batch_size=2)
    assert calc.flops_analytical() > 0
    assert model_config == original
    with pytest.raises(ValueError, match='positive integer'):
        calc.training_tflops(0)
