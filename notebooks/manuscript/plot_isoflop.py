#!/usr/bin/env python3
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

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_INPUT = Path("workspace/paper/isoflop_points.csv")
DEFAULT_OUTPUT_DIR = Path("workspace/paper")
METRICS = {
    "validation": ("val_loss", "Validation loss"),
    "training": ("train_loss", "Training loss"),
}
REQUIRED_COLUMNS = {
    "target_flops",
    "params_m",
    "iter",
    "train_loss",
    "val_loss",
}


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


def validate_data(data: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    numeric = list(REQUIRED_COLUMNS)
    if data[numeric].isna().any().any():
        raise ValueError("Required numeric columns contain missing values")
    if (data[numeric] <= 0).any().any():
        raise ValueError("FLOPs, model sizes, iterations, and losses must be positive")

    counts = data.groupby("target_flops").size()
    if (counts < 3).any():
        bad = counts[counts < 3].index.tolist()
        raise ValueError(f"At least three points are needed per FLOP budget: {bad}")


def quadratic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit log10(y) as a quadratic function of log10(x), returning R-squared."""
    log_x = np.log10(x)
    log_y = np.log10(y)
    coefficients = np.polyfit(log_x, log_y, deg=2)
    predicted = np.polyval(coefficients, log_x)
    residual_sum = np.square(log_y - predicted).sum()
    total_sum = np.square(log_y - log_y.mean()).sum()
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
    return coefficients, float(r_squared)


def fit_budget(group: pd.DataFrame, loss_column: str) -> dict[str, object]:
    """Fit one compute slice and extract its observed-range optimum."""
    group = group.sort_values("params_m")
    params = group["params_m"].to_numpy(dtype=float)
    iterations = group["iter"].to_numpy(dtype=float)
    losses = group[loss_column].to_numpy(dtype=float)

    model_coefficients, model_r_squared = quadratic_fit(params, losses)
    iteration_coefficients, iteration_r_squared = quadratic_fit(iterations, losses)

    curvature, linear, _ = model_coefficients
    if curvature <= 0:
        raise ValueError(
            f"Non-convex fit for {group['target_flops'].iloc[0]:.0e} FLOPs "
            f"using {loss_column}"
        )

    log_params = np.log10(params)
    unconstrained_log_vertex = -linear / (2.0 * curvature)
    optimum_log_params = float(
        np.clip(unconstrained_log_vertex, log_params.min(), log_params.max())
    )
    vertex_in_range = bool(log_params.min() <= unconstrained_log_vertex <= log_params.max())

    # Model size and iteration count are paired by the fixed-compute slice. Map
    # the model-space optimum through the observed relation in log-log space.
    optimum_log_iterations = np.interp(
        optimum_log_params, log_params, np.log10(iterations)
    )
    optimum_log_loss = np.polyval(model_coefficients, optimum_log_params)

    return {
        "target_flops": float(group["target_flops"].iloc[0]),
        "n_points": len(group),
        "model_coefficients": model_coefficients,
        "iteration_coefficients": iteration_coefficients,
        "optimum_params_m": 10.0**optimum_log_params,
        "optimum_iterations": 10.0**optimum_log_iterations,
        "fitted_min_loss": 10.0**optimum_log_loss,
        "unconstrained_params_m": 10.0**unconstrained_log_vertex,
        "vertex_in_observed_range": vertex_in_range,
        "model_fit_r2_log_loss": model_r_squared,
        "iteration_fit_r2_log_loss": iteration_r_squared,
    }


def format_flops(value: float) -> str:
    coefficient, exponent = f"{value:.0E}".split("E")
    return f"{coefficient}E{int(exponent)}"


def plot_metric(
    data: pd.DataFrame,
    metric_name: str,
    loss_column: str,
    y_label: str,
    output_path: Path,
    dpi: int,
) -> pd.DataFrame:
    budgets = np.sort(data["target_flops"].unique())
    colors = sns.color_palette("mako_r", n_colors=len(budgets))
    summaries: list[dict[str, object]] = []

    style = {
        "font.size": 13,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "legend.title_fontsize": 14,
        "axes.linewidth": 1.6,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

        for budget, color in zip(budgets, colors, strict=True):
            group = data[data["target_flops"] == budget].sort_values("params_m")
            fit = fit_budget(group, loss_column)
            summaries.append({"metric": metric_name, **fit})

            params = group["params_m"].to_numpy(dtype=float)
            iterations = group["iter"].to_numpy(dtype=float)
            losses = group[loss_column].to_numpy(dtype=float)

            model_grid = np.geomspace(params.min(), params.max(), 300)
            iteration_grid = np.geomspace(iterations.min(), iterations.max(), 300)
            model_curve = 10.0 ** np.polyval(
                fit["model_coefficients"], np.log10(model_grid)
            )
            iteration_curve = 10.0 ** np.polyval(
                fit["iteration_coefficients"], np.log10(iteration_grid)
            )

            label = format_flops(budget)
            axes[0].plot(model_grid, model_curve, color=color, linewidth=2.2, label=label)
            axes[0].scatter(params, losses, color=color, s=58, zorder=3)
            axes[1].plot(iteration_grid, iteration_curve, color=color, linewidth=2.2)
            axes[1].scatter(iterations, losses, color=color, s=58, zorder=3)

        axes[0].set_xscale("log")
        axes[1].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,)))
        axes[0].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda value, _: f"{value:g}")
        )
        axes[0].yaxis.set_minor_locator(
            ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
        )
        axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())
        axes[0].set_xlabel("Model size (M)")
        axes[1].set_xlabel("Training iterations")
        axes[0].set_ylabel(y_label)
        axes[0].legend(title="FLOPs", loc="upper left", frameon=True)

        for axis in axes:
            axis.tick_params(which="major", width=1.5, length=6)
            axis.tick_params(which="minor", width=1.1, length=3)
            axis.margins(x=0.05)

        figure.tight_layout(w_pad=2.0)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)

    summary = pd.DataFrame(summaries)
    return summary.drop(columns=["model_coefficients", "iteration_coefficients"])


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    validate_data(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for metric_name, (loss_column, y_label) in METRICS.items():
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
