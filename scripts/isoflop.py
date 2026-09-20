"""Collect isoFLOP points from run logs; optionally create fitted loss plots."""

import argparse
from pathlib import Path

import pandas as pd

from ptycho_fm.analysis.isoflop import (
    METRICS,
    collect_points,
    plot_fit_diagnostic,
    plot_metric,
    validate_data,
)
from ptycho_fm.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='+', type=Path, required=True)
    parser.add_argument('--targets', nargs='+', type=float, help='Defaults to each config training.log_at_flops')
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument('--loss-metric', choices=['train_loss', 'val_loss'], default='val_loss')
    parser.add_argument('--summary-csv', type=Path, default=Path('isoflop_points.csv'))
    parser.add_argument('--output-dir', type=Path, help='Also plot both available loss metrics')
    parser.add_argument('--fit-space', choices=['loglog', 'log_params', 'params'], default='loglog',
                        help='Optional diagnostic plot space; optima always use canonical log-log fits')
    args = parser.parse_args()
    selection = {path: args.targets or load_config(path)['training'].get('log_at_flops', []) for path in args.configs}
    data = pd.DataFrame(collect_points(selection, args.tolerance, args.loss_metric))
    if data.empty:
        raise ValueError('No isoFLOP points found within the requested tolerance')
    validate_data(data)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.summary_csv, index=False)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        for metric, (column, label) in METRICS.items():
            if data[column].notna().any():
                if args.fit_space != 'loglog':
                    plot_fit_diagnostic(data, column, args.fit_space, args.output_dir / f'diagnostic_{metric}_{args.fit_space}.png')
                summaries.append(plot_metric(data, metric, column, label, args.output_dir / f'isoflop_{metric}.png', 200))
        pd.concat(summaries, ignore_index=True).to_csv(args.output_dir / 'isoflop_optima.csv', index=False)


if __name__ == '__main__':
    main()
