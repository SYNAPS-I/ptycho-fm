"""Plot IsoFLOP curves and estimate compute-optimal training configurations.

For each fixed compute budget, the fit is quadratic in log10(model size) and
log10(loss). This makes the fitted curve a parabola in the log-log coordinates
used for the IsoFLOP analysis in Hoffmann et al. (2022):
https://arxiv.org/abs/2203.15556

The fitted vertex is constrained to the range of model sizes represented in
the input data. This avoids reporting an extrapolated optimum when a compute
slice does not contain an interior loss valley.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ptycho_fm.analysis.isoflop import METRICS, plot_metric, validate_data

DEFAULT_INPUT = Path("workspace/paper/isoflop_points.csv")
DEFAULT_OUTPUT_DIR = Path("workspace/paper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create training- and validation-loss IsoFLOP figures."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input points CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    validate_data(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for metric_name, (loss_column, y_label) in METRICS.items():
        if loss_column not in data or data[loss_column].dropna().empty:
            continue
        output_path = args.output_dir / f"isoflop_{metric_name}.png"
        summary = plot_metric(
            data,
            metric_name,
            loss_column,
            y_label,
            output_path,
            args.dpi,
        )
        summaries.append(summary)
        print(f"Wrote {output_path}")

    summary_path = args.output_dir / "isoflop_optima.csv"
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
