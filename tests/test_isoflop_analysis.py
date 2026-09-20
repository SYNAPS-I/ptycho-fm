"""Collector robustness and numerical equivalence to manuscript fits."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptycho_fm.analysis.isoflop import (
    find_row_for_target,
    fit_budget,
    historical_token_compute,
    validate_data,
)

FIXTURES = Path(__file__).parent / 'fixtures'


def test_historical_fits_match_original_plotter():
    data = pd.read_csv(FIXTURES / 'isoflop_points.csv')
    validate_data(data)
    for reference in json.loads((FIXTURES / 'isoflop_optima_legacy.json').read_text()):
        group = data[data.target_flops == reference['target_flops']]
        actual = fit_budget(group, reference['metric'])
        for key, value in reference.items():
            if key != 'metric':
                assert actual[key] == pytest.approx(value, rel=1e-10, abs=1e-12)
        tokens = group.assign(tokens=group['iter'] * 131072)
        token_fit = fit_budget(tokens, reference['metric'], length_column='tokens')
        assert token_fit['optimum_length'] == pytest.approx(actual['optimum_iterations'] * 131072)


def test_target_tolerance_nonfinite_and_resumed_duplicates():
    rows = [
        {'iter': '1', 'val_loss': 'nan', 'tflops_consumed': '100'},
        {'iter': '2', 'val_loss': '1', 'tflops_consumed': '103'},
        {'iter': '2', 'val_loss': '2', 'tflops_consumed': '103'},
        {'iter': '3', 'val_loss': '0.5', 'tflops_consumed': 'inf'},
    ]
    assert find_row_for_target(rows, 100)['val_loss'] == '2'
    assert find_row_for_target(rows, 100, tolerance=0.01) is None


def test_sparse_and_nonconvex_slices_are_explicit():
    data = pd.DataFrame({'params_m': [1, 10, 100], 'iter': [100, 10, 1],
                         'target_flops': [1000] * 3, 'val_loss': [1, 10, 1]})
    assert fit_budget(data.iloc[:2], 'val_loss')['fit_status'] == 'insufficient_distinct_points'
    result = fit_budget(data, 'val_loss')
    assert result['fit_status'] == 'non_convex'
    assert not result['vertex_in_observed_range']
    assert np.isnan(result['optimum_params_m'])


def test_historical_compute_scales_with_tokens_not_batch_history():
    model = {'encoder': {'img_size': 16, 'patch_size': 4, 'embed_dim': 16,
                         'num_heads': 2, 'depth': 1},
             'decoder': {'num_stages': 2, 'base_channels': 4}}
    assert historical_token_compute(model, 100) * 2 == historical_token_compute(model, 200)


def test_collector_prefers_saved_config_and_preserves_logged_counts(tmp_path):
    import yaml

    from ptycho_fm.analysis.isoflop import collect_points

    run = tmp_path / 'run1'
    run.mkdir()
    config = {'trainer': {'run_num': 1}, 'paths': {'model_save_path': str(tmp_path)},
              'model': {'encoder_type': 'unsupported'}, 'wandb': {'run_name': 'mutable'}}
    source = tmp_path / 'experiment.yaml'
    source.write_text(yaml.safe_dump(config))
    config['wandb']['run_name'] = 'saved'
    (run / 'config.yaml').write_text(yaml.safe_dump(config))
    (run / 'logs.txt').write_text('iter,val_loss,params_m,tflops_consumed,tokens_seen,accounting_version\n'
                                '2,0.5,42,100,2000,old_record\n')
    points = collect_points({source: [1e14]})
    assert len(points) == 1
    point = points[0]
    assert point['label'] == 'saved'
    assert point['params_m'] == 42
    assert point['parameter_count_source'] == 'logged'
    assert point['accounting_version'] == 'old_record'
    assert point['train_loss'] is None
    assert point['tokens_seen'] == 2000
    assert point['relative_compute_error'] == 0


def test_missing_metrics_are_independent_but_not_all_missing():
    data = pd.DataFrame({'target_flops': [1e12], 'params_m': [10], 'iter': [1],
                         'train_loss': [1.0], 'val_loss': [None]})
    validate_data(data)
    data['train_loss'] = None
    with pytest.raises(ValueError, match='At least one finite'):
        validate_data(data)


def test_flat_loss_does_not_claim_an_optimum():
    data = pd.DataFrame({'params_m': [1, 10, 100], 'iter': [100, 10, 1],
                         'target_flops': [1000] * 3, 'val_loss': [2, 2, 2]})
    fit = fit_budget(data, 'val_loss')
    assert fit['fit_status'] == 'degenerate'
    assert np.isnan(fit['optimum_params_m'])
