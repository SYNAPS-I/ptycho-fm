"""Update-based warmup/stable and square-root cooldown schedules."""

import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, config):
    """Return (scheduler, step_unit); existing Torch schedulers remain epoch-based."""
    if not config.get('enabled', False):
        return None, 'epoch'
    name = config.get('scheduler_class')
    if not name:
        raise ValueError('lr_scheduler.scheduler_class is required')
    kwargs = dict(config.get('kwargs') or {})
    if name == 'warmup-stable':
        warmup = kwargs.pop('warmup_steps', 0)
        if not isinstance(warmup, int) or warmup < 0:
            raise ValueError('warmup_steps must be a non-negative integer')
        def factor(step):
            return (step + 1) / warmup if step < warmup else 1.0
        scheduler = LambdaLR(optimizer, factor)
        unit = 'update'
    elif name == 'cooldown':
        steps = kwargs.pop('cooldown_steps', None)
        kwargs.pop('branch_from', None)
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError('cooldown_steps must be a positive integer')
        def factor(step):
            return 1.0 - math.sqrt(min((step + 1) / steps, 1.0))
        scheduler = LambdaLR(optimizer, factor)
        unit = 'update'
    else:
        from torch.optim import lr_scheduler
        cls = getattr(lr_scheduler, name, None)
        if cls is None:
            raise ValueError(f'Unknown lr scheduler class: {name}')
        return cls(optimizer=optimizer, **kwargs), 'epoch'
    if kwargs:
        raise ValueError(f'Unexpected {name} options: {sorted(kwargs)}')
    return scheduler, unit
