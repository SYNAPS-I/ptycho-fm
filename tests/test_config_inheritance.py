"""Configuration inheritance must work independently of the launch directory."""

import pytest

from ptycho_fm.utils.config import deep_merge_config, load_config


def test_relative_parents_and_caller_path(tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    (configs / 'base.yaml').write_text('model: {encoder: {depth: 12, embed_dim: 512}}\nitems: [1, 2]\n')
    (configs / 'child.yaml').write_text('extends: base.yaml\nmodel: {encoder: {depth: 6}}\nitems: [3]\n')
    monkeypatch.chdir(tmp_path)
    assert load_config('configs/child.yaml') == {
        'model': {'encoder': {'depth': 6, 'embed_dim': 512}}, 'items': [3],
    }


def test_merge_does_not_mutate_or_alias_inputs():
    base = {'model': {'values': [1], 'depth': 12}}
    override = {'model': {'depth': 6}}
    result = deep_merge_config(base, override)
    result['model']['values'].append(2)
    assert base == {'model': {'values': [1], 'depth': 12}}
    assert override == {'model': {'depth': 6}}


def test_cycle_through_resolved_paths(tmp_path):
    (tmp_path / 'a.yaml').write_text('extends: b.yaml\n')
    (tmp_path / 'b.yaml').write_text('extends: ./a.yaml\n')
    with pytest.raises(ValueError, match='Circular'):
        load_config(tmp_path / 'a.yaml')


@pytest.mark.parametrize('text', ['[1, 2]', 'extends: [base.yaml]', 'extends: ""'])
def test_invalid_config(tmp_path, text):
    path = tmp_path / 'config.yaml'
    path.write_text(text)
    with pytest.raises((TypeError, ValueError)):
        load_config(path)
