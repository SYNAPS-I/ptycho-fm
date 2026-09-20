"""Collect isoFLOP points and share constrained fits across CLI and notebooks."""

import csv
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ptycho_fm.utils.config import load_config
from ptycho_fm.utils.flops import PtychoFMFlopsCalculator

METRICS = {'validation': ('val_loss', 'Validation loss'), 'training': ('train_loss', 'Training loss')}
REQUIRED_COLUMNS = {'target_flops', 'params_m', 'iter'}


def validate_data(data):
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f'Input CSV is missing columns: {sorted(missing)}')
    for column in REQUIRED_COLUMNS:
        values = pd.to_numeric(data[column], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f'{column} must contain finite positive values')
    if not {'train_loss', 'val_loss'}.intersection(data.columns):
        raise ValueError('At least one loss column is required')
    if not any(data[c].notna().any() for c in ('train_loss', 'val_loss') if c in data):
        raise ValueError('At least one finite positive loss is required')
    for column in ('train_loss', 'val_loss'):
        if column in data:
            values = pd.to_numeric(data[column].dropna(), errors='raise').to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f'{column} must contain finite positive values or missing entries')


def to_float(row, key):
    try:
        value = float(row.get(key))
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def find_row_for_target(rows, target_tflops, tolerance=0.05, loss_metric='val_loss'):
    if not math.isfinite(target_tflops) or target_tflops <= 0 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Target must be finite and positive, tolerance finite and non-negative')
    # A resumed run can repeat a logged iteration; its latest row is authoritative.
    unique = {row.get('iter', i): row for i, row in enumerate(rows)}
    candidates = []
    for row in unique.values():
        loss, compute = to_float(row, loss_metric), to_float(row, 'tflops_consumed')
        iteration = to_float(row, 'iter')
        if loss is None or loss <= 0 or compute is None or compute <= 0 or iteration is None or iteration <= 0:
            continue
        error = abs(compute - target_tflops) / target_tflops
        if error <= tolerance:
            candidates.append((error, -iteration, row))
    return min(candidates, key=lambda item: item[:2])[2] if candidates else None


def collect_points(config_flops, tolerance=0.05, loss_metric='val_loss'):
    points = []
    for config_name, targets in config_flops.items():
        path = Path(config_name).expanduser().resolve()
        config = load_config(path)
        run_num = config['trainer']['run_num']
        run_dir = Path(config['paths']['model_save_path']).expanduser() / f'run{run_num}'
        saved = run_dir / 'config.yaml'
        if saved.is_file():
            config = load_config(saved)
        log_path = run_dir / 'logs.txt'
        if not log_path.is_file():
            warnings.warn(f'Missing training log: {log_path}', stacklevel=2)
            continue
        with log_path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for target in targets:
            row = find_row_for_target(rows, float(target) / 1e12, tolerance, loss_metric)
            if row is None:
                warnings.warn(f'No {loss_metric} row within tolerance of {target:g} FLOPs in {log_path}', stacklevel=2)
                continue
            params = to_float(row, 'params_m')
            param_source = 'logged'
            if params is None:
                # Explicitly label reconstruction; never imply these are legacy counts.
                params = PtychoFMFlopsCalculator(config['model'], batch_size=1).param_count()
                param_source = 'current_model_reconstruction'
            points.append({
                'config': str(path), 'label': config.get('wandb', {}).get('run_name') or path.stem,
                'run_num': run_num, 'target_flops': float(target), 'params_m': params,
                'train_loss': to_float(row, 'train_loss'), 'val_loss': to_float(row, 'val_loss'),
                'iter': int(float(row['iter'])), 'tflops_consumed': float(row['tflops_consumed']),
                'relative_compute_error': abs(float(row['tflops_consumed']) * 1e12 - target) / target,
                'accounting_version': row.get('accounting_version') or 'legacy_unspecified',
                'parameter_count_source': param_source,
                **{key: to_float(row, key) for key in ('samples_seen', 'tokens_seen', 'world_size', 'batch_size_per_rank')},
            })
    return points


def quadratic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit log10(y) as a quadratic function of log10(x), returning R-squared."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(np.unique(x)) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all() or (x <= 0).any() or (y <= 0).any():
        raise ValueError('Quadratic fitting requires at least three distinct positive finite x values and positive finite losses')
    log_x = np.log10(x)
    log_y = np.log10(y)
    coefficients = np.polyfit(log_x, log_y, deg=2)
    predicted = np.polyval(coefficients, log_x)
    residual_sum = np.square(log_y - predicted).sum()
    total_sum = np.square(log_y - log_y.mean()).sum()
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
    return coefficients, float(r_squared)

def fit_budget(group: pd.DataFrame, loss_column: str, length_column: str = "iter") -> dict[str, object]:
    """Fit one compute slice and extract its observed-range optimum."""
    group = group.dropna(subset=[loss_column]).sort_values("params_m")
    if group.empty:
        raise ValueError(f'No valid {loss_column} points')
    params = group["params_m"].to_numpy(dtype=float)
    iterations = group[length_column].to_numpy(dtype=float)
    losses = group[loss_column].to_numpy(dtype=float)

    result = {
        'target_flops': float(group['target_flops'].iloc[0]), 'n_points': len(group),
        'model_coefficients': None, 'iteration_coefficients': None,
        'optimum_params_m': np.nan, 'optimum_iterations': np.nan,
        'optimum_length': np.nan, 'fitted_min_loss': np.nan,
        'unconstrained_params_m': np.nan, 'vertex_in_observed_range': False,
        'model_fit_r2_log_loss': np.nan, 'iteration_fit_r2_log_loss': np.nan,
        'fit_status': 'insufficient_distinct_points',
    }
    if len(np.unique(params)) < 3 or len(np.unique(iterations)) < 3:
        return result
    model_coefficients, model_r_squared = quadratic_fit(params, losses)
    iteration_coefficients, iteration_r_squared = quadratic_fit(iterations, losses)
    curvature, linear, _ = model_coefficients
    degenerate = abs(curvature) <= 100 * np.finfo(float).eps * max(1.0, np.max(np.abs(model_coefficients)))
    if curvature <= 0 or degenerate:
        return {**result, 'model_coefficients': model_coefficients,
                'iteration_coefficients': iteration_coefficients,
                'model_fit_r2_log_loss': model_r_squared,
                'iteration_fit_r2_log_loss': iteration_r_squared,
                'fit_status': 'degenerate' if degenerate else 'non_convex'}

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
        "optimum_iterations": 10.0 ** np.interp(optimum_log_params, log_params, np.log10(group['iter'].to_numpy(dtype=float))),
        "optimum_length": 10.0**optimum_log_iterations,
        "fit_status": 'interior' if vertex_in_range else 'boundary',
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
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import ticker

    data = data.dropna(subset=[loss_column])
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

            if fit['model_coefficients'] is None:
                axes[0].scatter(params, losses, color=color, s=58, label=format_flops(budget))
                axes[1].scatter(iterations, losses, color=color, s=58)
                continue
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
    return summary.drop(columns=["model_coefficients", "iteration_coefficients"], errors="ignore")


def historical_token_compute(model_config, tokens):
    """Token-normalized historical estimate; does not reconstruct rank batches.

    Normalize the current analytical training cost at a one-pattern reference
    batch by patch tokens per image, then multiply by documented token totals.
    Historical sampling/hardware metadata do not change this estimate.
    """
    calc = PtychoFMFlopsCalculator(model_config, batch_size=1)
    tokens_per_image = calc.grid_after_vit() ** 2
    if not math.isfinite(tokens) or tokens < 0:
        raise ValueError('Token count must be finite and non-negative')
    return calc.training_tflops(1) * 1e12 * tokens / tokens_per_image


def plot_fit_diagnostic(data, loss_column, fit_space, output_path):
    """Incoming alternative fit spaces, without claiming canonical optima."""
    import matplotlib.pyplot as plt

    if fit_space not in {'params', 'log_params', 'loglog'}:
        raise ValueError(f'Unknown fit space: {fit_space}')
    figure, axis = plt.subplots(figsize=(7, 5))
    for budget, group in data.dropna(subset=[loss_column]).groupby('target_flops'):
        x = group['params_m'].to_numpy(dtype=float)
        y = group[loss_column].to_numpy(dtype=float)
        points = axis.scatter(x, y, label=format_flops(budget))
        if len(np.unique(x)) < 3:
            continue
        fit_x = x if fit_space == 'params' else np.log10(x)
        fit_y = np.log10(y) if fit_space == 'loglog' else y
        coefficients = np.polyfit(fit_x, fit_y, 2)
        grid = np.linspace(fit_x.min(), fit_x.max(), 300)
        fitted = np.polyval(coefficients, grid)
        axis.plot(grid if fit_space == 'params' else 10**grid,
                  10**fitted if fit_space == 'loglog' else fitted,
                  color=points.get_facecolor()[0])
    axis.set(xlabel='Parameters (millions)', ylabel=loss_column,
             xscale='linear' if fit_space == 'params' else 'log',
             yscale='log' if fit_space == 'loglog' else 'linear',
             title=f'Diagnostic quadratic fit: {fit_space}')
    axis.legend(title='FLOPs')
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
