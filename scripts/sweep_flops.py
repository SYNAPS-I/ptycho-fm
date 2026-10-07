"""Report integer iteration counts and attained compute for selected sweep models."""

import argparse
import csv
import math
import sys
from pathlib import Path

from ptycho_fm.utils.config import load_config
from ptycho_fm.utils.flops import PtychoFMFlopsCalculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', type=Path, nargs='+', required=True)
    parser.add_argument('--budgets', type=float, nargs='+', required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--batch-size', type=int, help='Per rank; defaults to each training config')
    parser.add_argument('--min-iters', type=int, default=4000)
    parser.add_argument('--max-iters', type=int, default=400000)
    args = parser.parse_args()
    if args.world_size < 1 or any(not math.isfinite(v) or v <= 0 for v in args.budgets):
        parser.error('world-size and budgets must be positive')
    writer = csv.DictWriter(sys.stdout, fieldnames=['config', 'target_flops', 'params_m', 'iter', 'actual_flops', 'global_batch_size'])
    writer.writeheader()
    for path in args.configs:
        config = load_config(path)
        batch = args.batch_size or config['training']['batch_size']
        calc = PtychoFMFlopsCalculator(config['model'], batch_size=batch)
        per_step = calc.training_tflops(batch) * 1e12 * args.world_size
        for budget in sorted(args.budgets):
            iterations = math.ceil(budget / per_step)
            if args.min_iters <= iterations <= args.max_iters:
                writer.writerow({'config': str(path), 'target_flops': budget,
                                 'params_m': calc.param_count(), 'iter': iterations,
                                 'actual_flops': iterations * per_step,
                                 'global_batch_size': batch * args.world_size})


if __name__ == '__main__':
    main()
