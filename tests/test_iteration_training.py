"""Exercise the real Trainer loop, scheduler, and checkpoint continuation."""

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from ptycho_fm.training import Trainer
from ptycho_fm.utils.data import ResumeSampler
from ptycho_fm.utils.progress import TrainingProgress
from ptycho_fm.utils.schedulers import build_scheduler


class Cost:
    def training_tflops(self, batch):
        return (batch + 10) * 1e-12


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(1, 1, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, probe, normalization, scale):
        result = self.layer(self.dropout(x))
        return result, result, result


def data():
    x = torch.linspace(0.1, 1.0, 9 * 8 * 8).reshape(9, 1, 8, 8)
    probe = torch.ones(9, 1, 1, 8, 8, dtype=torch.complex64)
    return TensorDataset(x, x * 0.5, x * 0.2, probe, torch.zeros(9, 2),
                         torch.ones(9), torch.ones(9))


def make_loader(dataset, resume=0):
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=7)
    if resume:
        sampler = ResumeSampler(sampler, 0, resume)
    return DataLoader(dataset, batch_size=2, sampler=sampler,
                      generator=torch.Generator().manual_seed(9))


def metrics():
    return {'training_loss': [], 'train_amp_loss': [], 'train_ph_loss': [],
            'validation_loss': [], 'val_amp_loss': [], 'val_ph_loss': [],
            'best_val_loss': float('inf')}


def setup(tmp_path, max_iters=None, skip=None):
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler, unit = build_scheduler(optimizer, {
        'enabled': True, 'scheduler_class': 'warmup-stable', 'kwargs': {'warmup_steps': 3},
    })
    progress = TrainingProgress({'log_every': 2, 'max_iters': max_iters}, calculator=Cost())
    trainer = Trainer(model, 'unsupervised', 'test', torch.device('cpu'), str(tmp_path),
                      progress=progress,
                      skip_batch_if_grad_norm_greater_than=skip,
                      resume_context={'world_size': 1, 'batch_size': 2})
    return trainer, optimizer, scheduler, unit


def test_interrupted_resume_matches_uninterrupted_weights_rng_and_compute(tmp_path):
    dataset = data()
    torch.manual_seed(123)
    whole, whole_opt, whole_sched, unit = setup(tmp_path / 'whole')
    whole.train(make_loader(dataset), nn.MSELoss(), whole_opt, metrics(), epoch=0,
                scheduler=whole_sched, scheduler_unit=unit)
    expected = copy.deepcopy(whole.model.state_dict())
    expected_rng = torch.get_rng_state()

    torch.manual_seed(123)
    first, first_opt, first_sched, unit = setup(tmp_path / 'split', max_iters=2)
    first_metrics = metrics()
    first.train(make_loader(dataset), nn.MSELoss(), first_opt, first_metrics, epoch=0,
                scheduler=first_sched, scheduler_unit=unit)
    assert first.progress.samples_in_epoch == 4
    first.save_model_and_states_checkpoint(0, first_metrics, first_opt, scheduler=first_sched)
    resumed, resumed_opt, resumed_sched, unit = setup(tmp_path / 'split')
    resumed.model.load_state_dict(torch.load(tmp_path / 'split/runtest/checkpoint_model.pth', weights_only=True))
    epoch, saved_metrics, *_ = resumed.load_state_checkpoint(resumed_opt, resumed_sched)
    resumed.train(make_loader(dataset, resumed.progress.samples_in_epoch), nn.MSELoss(),
                  resumed_opt, saved_metrics, epoch=epoch,
                  scheduler=resumed_sched, scheduler_unit=unit)
    for key, value in expected.items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert resumed.progress.state_dict() == whole.progress.state_dict()
    assert resumed_opt.param_groups[0]['lr'] == whole_opt.param_groups[0]['lr']
    assert resumed.progress.epoch == 1
    assert resumed.progress.samples_in_epoch == 0
    assert resumed.progress.samples_seen == 9
    assert resumed.progress.tflops_consumed == pytest.approx(59e-12)


def test_skipped_updates_consume_compute_but_not_scheduler_steps(tmp_path):
    trainer, opt, scheduler, unit = setup(tmp_path, skip=0.0)
    original = copy.deepcopy(trainer.model.state_dict())
    initial_step = scheduler.last_epoch
    trainer.train(make_loader(data()), nn.MSELoss(), opt, metrics(), epoch=0,
                  scheduler=scheduler, scheduler_unit=unit)
    assert trainer.progress.iters == 5
    assert trainer.progress.optimizer_steps == 0
    assert scheduler.last_epoch == initial_step
    assert trainer.progress.tflops_consumed == pytest.approx(59e-12)
    for key, value in original.items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value, rtol=0, atol=0)


def test_batch_metrics_use_completed_iteration_steps(tmp_path):
    class CapturingLogger:
        log_every_n_batches = 2

        def __init__(self):
            self.calls = []

        def log_metrics(self, metrics, step=None):
            self.calls.append((metrics, step))

    trainer, optimizer, scheduler, unit = setup(tmp_path)
    logger = CapturingLogger()
    trainer.experiment_logger = logger
    trainer.train(
        make_loader(data()),
        nn.MSELoss(),
        optimizer,
        metrics(),
        epoch=0,
        scheduler=scheduler,
        scheduler_unit=unit,
    )

    batch_calls = [
        (payload, step)
        for payload, step in logger.calls
        if "train_batch_loss" in payload
    ]
    assert [step for _, step in batch_calls] == [2, 4]


def test_null_batch_interval_logs_epoch_after_final_iteration(tmp_path):
    class CapturingLogger:
        log_every_n_batches = None

        def __init__(self):
            self.calls = []

        def log_metrics(self, metrics, step=None):
            self.calls.append((metrics, step))

    trainer, optimizer, scheduler, unit = setup(tmp_path)
    trainer.progress = TrainingProgress()
    logger = CapturingLogger()
    trainer.experiment_logger = logger
    trainer.train(
        make_loader(data()),
        nn.MSELoss(),
        optimizer,
        metrics(),
        epoch=0,
        scheduler=scheduler,
        scheduler_unit=unit,
    )

    assert len(logger.calls) == 1
    payload, step = logger.calls[0]
    assert "train_epoch_loss" in payload
    assert step == 5


def test_targets_crossed_together_and_optional_stopping():
    progress = TrainingProgress({'log_at_flops': [5, 10], 'stop_at_last_flop_target': True}, Cost())
    due, crossed = progress.record(batch_size=2, global_samples=4, tflops=24e-12,
                                   updated=True, losses=(1, 2, 3))
    assert due and crossed == [5, 10] and progress.stopped
    progress.stop_at_last_target = False
    assert not progress.stopped


def test_checkpoint_rejects_changed_optimizer_and_context(tmp_path):
    trainer, opt, scheduler, _ = setup(tmp_path)
    trainer.save_model_and_states_checkpoint(0, metrics(), opt, scheduler=scheduler)
    trainer.resume_context['batch_size'] = 3
    with pytest.raises(ValueError, match='settings differ'):
        trainer.load_state_checkpoint(opt, scheduler)
    trainer.resume_context['batch_size'] = 2
    changed = torch.optim.Adam([{'params': [p]} for p in trainer.model.parameters()])
    with pytest.raises(ValueError, match='Optimizer parameters changed'):
        trainer.load_state_checkpoint(changed, scheduler)


def test_checkpoint_allows_changed_num_workers(tmp_path):
    trainer, opt, scheduler, _ = setup(tmp_path)
    trainer.resume_context['data'] = {'max_shards': 10, 'num_workers': 2}
    trainer.save_model_and_states_checkpoint(
        0, metrics(), opt, scheduler=scheduler
    )

    trainer.resume_context['data']['num_workers'] = 4
    trainer.load_state_checkpoint(opt, scheduler)

    trainer.resume_context['data']['max_shards'] = 11
    with pytest.raises(ValueError, match='settings differ'):
        trainer.load_state_checkpoint(opt, scheduler)
