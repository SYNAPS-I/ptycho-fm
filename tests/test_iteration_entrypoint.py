"""Real entry-point continuation, cooldown and two-rank collective coverage."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from ptycho_fm.utils.flops import PtychoFMFlopsCalculator
from tests.test_data_refactor import make_pair

ROOT = Path(__file__).resolve().parents[1]


def run_training(config, path, ranks, worker):
    path.write_text(yaml.safe_dump(config))
    command = [sys.executable]
    if ranks > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nproc-per-node', str(ranks)]
    command += [str(worker), '--config', str(path)]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'MPLBACKEND': 'Agg', 'PYTHONPATH': str(ROOT)}
    result = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]
    run_dir = Path(config['paths']['model_save_path']) / ('run' + config['trainer']['run_num'])
    return (torch.load(run_dir / 'checkpoint_model.pth', weights_only=True),
            torch.load(run_dir / 'checkpoint.state', weights_only=True),
            pd.read_csv(run_dir / 'logs.txt'))


@pytest.mark.integration
def test_entrypoint_resume_and_cooldown(tmp_path):
    ranks = 2
    for name in ('a', 'b'):
        make_pair(tmp_path / 'data', name, modes=1, pattern_shape=(16, 16))
    worker = tmp_path / 'worker.py'
    worker.write_text('import torch\nfrom ptycho_fm.train import main\ntorch.manual_seed(123)\nmain()\n')
    config = yaml.safe_load((ROOT / 'config.yaml').read_text())
    config['data'].update(data_path=str(tmp_path / 'data'), packed=False, test_path=None,
                          normalization_dict_path=None, num_workers=0, pin_memory=False,
                          use_cuda_prefetcher=False, persistent_workers=False,
                          train_split=0.75, sharding_strategy='dynamic', drop_last=False,
                          apply_noise=True, deterministic_noise=True, noise_seed=17,
                          scale=2, default_normalization=2, max_probe_modes=1)
    config['model']['encoder'].update(img_size=16, patch_size=4, embed_dim=16,
                                      num_heads=2, depth=1, mlp_ratio=2, dropout=0.1)
    config['model']['decoder'].update(base_channels=4, latent_dim=None, num_stages=2)
    config['paths']['model_save_path'] = str(tmp_path / 'models')
    config['wandb']['enabled'] = False
    config['mlflow'] = {'enabled': False}
    config['training'].update(batch_size=2, epochs=2, log_every=1, track_flops=True,
                              loss_function='weighted', platform='slurm')
    config['training']['lr_scheduler'] = {
        'enabled': True, 'scheduler_class': 'warmup-stable', 'kwargs': {'warmup_steps': 2}}
    first_step_flops = PtychoFMFlopsCalculator(config['model'], batch_size=2).training_tflops(2) * ranks * 1e12
    config['training']['log_at_flops'] = [first_step_flops / 4, first_step_flops * 3 / 4]
    config['trainer']['run_num'] = 'whole'
    whole_model, whole_state, whole_log = run_training(config, tmp_path / 'whole.yaml', ranks, worker)
    config['trainer']['run_num'] = 'split'
    config['training']['stop_at_last_flop_target'] = True
    _, first_state, _ = run_training(config, tmp_path / 'first.yaml', ranks, worker)
    assert first_state['rank_runtime'][0]['progress']['samples_in_epoch'] == 2
    assert first_state['rank_runtime'][0]['progress']['iters'] == 1
    assert whole_log['targets_crossed'].iloc[0] == 2
    config['training'].update(stop_at_last_flop_target=False, resume_from_checkpoint=True)
    resumed_model, resumed_state, resumed_log = run_training(config, tmp_path / 'resume.yaml', ranks, worker)
    for key, value in whole_model.items():
        torch.testing.assert_close(resumed_model[key], value, rtol=0, atol=0)
    pd.testing.assert_frame_equal(whole_log, resumed_log, rtol=0, atol=0)
    for whole_rank, resumed_rank in zip(whole_state['rank_runtime'], resumed_state['rank_runtime'], strict=True):
        assert whole_rank['progress'] == resumed_rank['progress']
        assert torch.equal(whole_rank['torch_rng'], resumed_rank['torch_rng'])
    assert resumed_state['rank_runtime'][0]['progress']['samples_seen'] == 12
    assert resumed_log['world_size'].eq(ranks).all()

    config = copy.deepcopy(config)
    config['trainer']['run_num'] = 'cooldown'
    config['training']['resume_from_checkpoint'] = False
    config['training']['lr_scheduler'] = {
        'enabled': True, 'scheduler_class': 'cooldown',
        'kwargs': {'cooldown_steps': 2, 'branch_from': str(tmp_path / 'models/runsplit/state_iters_1.pth')}}
    _, cooldown_state, cooldown_log = run_training(config, tmp_path / 'cooldown.yaml', ranks, worker)
    progress = cooldown_state['rank_runtime'][0]['progress']
    assert progress['optimizer_steps'] == 3
    assert progress['iters'] == 3
    assert progress['tflops_consumed'] > first_state['rank_runtime'][0]['progress']['tflops_consumed']
    assert cooldown_log['lr'].iloc[-1] == 0


def test_plain_mse_trains_with_float32_batches(tmp_path):
    make_pair(tmp_path / 'data', 'a', modes=1, pattern_shape=(16, 16))
    make_pair(tmp_path / 'data', 'b', modes=1, pattern_shape=(16, 16))
    worker = tmp_path / 'worker.py'
    worker.write_text('from ptycho_fm.train import main\nmain()\n')
    config = yaml.safe_load((ROOT / 'config.yaml').read_text())
    config['data'].update(data_path=str(tmp_path / 'data'), packed=False, test_path=None,
                          normalization_dict_path=None, num_workers=0, pin_memory=False,
                          use_cuda_prefetcher=False, persistent_workers=False,
                          train_split=0.75, sharding_strategy='dynamic', drop_last=False,
                          apply_noise=False, scale=2.0, default_normalization=3.2,
                          max_probe_modes=1)
    config['model']['encoder'].update(img_size=16, patch_size=4, embed_dim=16,
                                      num_heads=2, depth=1, mlp_ratio=2)
    config['model']['decoder'].update(base_channels=4, latent_dim=None, num_stages=2)
    config['paths']['model_save_path'] = str(tmp_path / 'models')
    config['wandb']['enabled'] = False
    config['mlflow'] = {'enabled': False}
    config['training'].update(batch_size=2, epochs=1, log_every=1, track_flops=False,
                              loss_function='mse', platform='slurm')
    config['trainer']['run_num'] = 'mse'
    _, state, logs = run_training(config, tmp_path / 'mse.yaml', 1, worker)
    assert state['rank_runtime'][0]['progress']['optimizer_steps'] == 3
    assert logs['val_loss'].notna().all()
