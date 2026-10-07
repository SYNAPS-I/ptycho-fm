"""Serializable iteration progress, FLOP targets, and offline scalar logs."""

import csv
import math
from pathlib import Path

from ptycho_fm.utils.flops import ACCOUNTING_VERSION


class TrainingProgress:
    def __init__(self, training=None, calculator=None):
        config = training or {}
        self.log_every = config.get('log_every', 0)
        self.max_iters = config.get('max_iters')
        for name, value in [('log_every', self.log_every), ('max_iters', self.max_iters)]:
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)
                                      or value < (1 if name == 'max_iters' else 0)):
                raise ValueError(f'{name} must be a positive integer (log_every may be zero)')
        self.targets = sorted({float(v) for v in config.get('log_at_flops', [])})
        if any(not math.isfinite(v) or v <= 0 for v in self.targets):
            raise ValueError('log_at_flops must contain finite positive FLOP targets')
        if self.targets and calculator is None:
            raise ValueError('log_at_flops requires a supported FLOP calculator')
        self.stop_at_last_target = bool(config.get('stop_at_last_flop_target', False))
        if self.stop_at_last_target and not self.targets:
            raise ValueError('stop_at_last_flop_target requires log_at_flops')
        self.enabled = bool(self.log_every or self.targets or self.max_iters or calculator)
        self.calculator = calculator
        self.iters = 0
        self.optimizer_steps = 0
        self.epoch = 0
        self.iters_in_epoch = 0
        self.samples_in_epoch = 0  # rank-local consumed samples, excluding prefetch
        self.samples_seen = 0  # global processed samples, including sampler padding
        self.tflops_consumed = 0.0
        self.next_target = 0
        self.last_log_iter = 0
        self.interval_sums = [0.0, 0.0, 0.0]
        self.interval_batches = 0
        self.last_validation_iter = -1
        self.max_updates = None

    @property
    def stopped(self):
        return ((self.max_iters is not None and self.iters >= self.max_iters)
                or (self.max_updates is not None and self.optimizer_steps >= self.max_updates)
                or (self.stop_at_last_target and self.next_target >= len(self.targets)))

    def record(self, *, batch_size, global_samples, tflops, updated, losses):
        self.iters += 1
        self.iters_in_epoch += 1
        self.samples_in_epoch += batch_size
        self.samples_seen += global_samples
        self.optimizer_steps += int(updated)
        self.tflops_consumed += tflops
        self.interval_batches += 1
        self.interval_sums = [total + value for total, value in zip(self.interval_sums, losses, strict=True)]
        crossed = []
        while (self.next_target < len(self.targets)
               and self.tflops_consumed * 1e12 >= self.targets[self.next_target]):
            crossed.append(self.targets[self.next_target])
            self.next_target += 1
        due = bool(crossed) if self.targets else bool(
            self.log_every and self.iters - self.last_log_iter >= self.log_every)
        return due or (self.enabled and self.stopped), crossed

    def finish_epoch(self):
        self.epoch += 1
        self.iters_in_epoch = 0
        self.samples_in_epoch = 0

    def interval_metrics(self):
        if not self.interval_batches:
            return {}
        return dict(zip(('train_loss', 'train_amp_loss', 'train_ph_loss'),
                        (v / self.interval_batches for v in self.interval_sums), strict=True))

    def mark_logged(self):
        self.last_log_iter = self.iters
        self.last_validation_iter = self.iters
        self.interval_batches = 0
        self.interval_sums = [0.0, 0.0, 0.0]

    def state_dict(self):
        keys = ('iters', 'optimizer_steps', 'epoch', 'iters_in_epoch', 'samples_in_epoch',
                'samples_seen', 'tflops_consumed', 'last_log_iter', 'interval_sums',
                'interval_batches', 'last_validation_iter', 'max_updates')
        return {**{key: getattr(self, key) for key in keys},
                'accounting_version': ACCOUNTING_VERSION if self.calculator else None}

    def load_state_dict(self, state):
        expected = ACCOUNTING_VERSION if self.calculator else None
        if state.get('accounting_version') != expected:
            raise ValueError('Checkpoint compute accounting differs; explicit conversion is required')
        for key in self.state_dict():
            if key != 'accounting_version':
                setattr(self, key, state[key])
        self.next_target = sum(v <= self.tflops_consumed * 1e12 for v in self.targets)


def append_logs(path, row):
    """Write a fixed schema, refusing incompatible existing logs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fields = list(row)
    if exists:
        with path.open(newline='') as stream:
            fields = next(csv.reader(stream))
        if set(fields) != set(row):
            raise ValueError(f'Log schema differs at {path}; migrate the existing log before resuming')
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
