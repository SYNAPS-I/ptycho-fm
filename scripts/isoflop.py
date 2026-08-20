#!/usr/bin/env python
"""Plot isoFLOP loss versus model size and training iterations.

Reads each sweep config, finds its run directory, loads ``logs.txt``, and plots
the validation loss recorded near each configured FLOP target.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.flops_utils import PtychoViTFlopsCalculator  # noqa: E402


ISO_FLOP_CONFIGS = {
    # Comment out a config or remove individual FLOP targets to exclude them.
    "configs/e192_d6.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
    "configs/e256_d8.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
    "configs/e384_d8.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
    "configs/e512_d8.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
    "configs/e512_d12.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
    "configs/e640_d12.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
    "configs/e768_d12.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
    "configs/e768_d16.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
    "configs/e1024_d12.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
    "configs/e1024_d16.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
    "configs/e1024_d24.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
    "configs/e1536_d16.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
    "configs/e1536_d24.yaml": [6.0e17, 1.0e18, 3.0e18, 6.0e18, 1.0e19],
}

FIGSIZE = (6.5, 2.5)
PNG_DPI = 400
POINT_SIZE = 14

plt.style.use("default")
plt.rcParams.update(
    {
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 6,
        "legend.title_fontsize": 6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "mathtext.default": "rm",
    }
)


def display_path(path: Path) -> str:
    """Return a repository-relative path without exposing absolute locations."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def deep_merge_config(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: Path, seen: set[Path] | None = None) -> dict:
    config_path = config_path.resolve()
    seen = set() if seen is None else seen
    if config_path in seen:
        raise ValueError(f"Circular config extends detected at {config_path}")
    seen.add(config_path)

    with config_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg

    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    return deep_merge_config(load_config(parent_path, seen), cfg)


def read_logs(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    with log_path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def find_row_for_target(
    rows: list[dict],
    target_tflops: float,
    tolerance: float,
    loss_metric: str,
) -> dict | None:
    candidates = []
    for row in rows:
        loss = to_float(row, loss_metric)
        tflops = to_float(row, "tflops_consumed")
        if loss is None or tflops is None:
            continue
        rel_err = abs(tflops - target_tflops) / target_tflops
        candidates.append((rel_err, row))
    if not candidates:
        return None

    rel_err, row = min(candidates, key=lambda item: item[0])
    return row if rel_err <= tolerance else None


def model_params_millions(config: dict) -> float:
    probe_modes = int(config.get("data", {}).get("max_probe_modes", 10))
    model_cfg = config["model"]
    img_size = int(model_cfg.get("encoder", {}).get("img_size", 256))
    calc = PtychoViTFlopsCalculator(
        model_cfg,
        batch_size=1,
        spatial=img_size,
        probe_modes=probe_modes,
    )
    return calc.param_count()


def label_for_config(config_path: Path, config: dict) -> str:
    run_name = config.get("wandb", {}).get("run_name")
    return str(run_name) if run_name else config_path.stem


def collect_points(
    config_flops: dict[str, list[float]],
    tolerance: float,
    loss_metric: str,
) -> list[dict]:
    points = []
    for config_name, target_flops_list in config_flops.items():
        config_path = REPO_ROOT / config_name
        if not config_path.exists():
            print(f"[warn] Missing config: {config_name}")
            continue
        if not target_flops_list:
            print(f"[warn] No FLOP targets configured for {config_name}; skipping")
            continue

        config = load_config(config_path)
        params_m = model_params_millions(config)
        run_num = config["trainer"]["run_num"]
        run_dir = Path(config["paths"]["model_save_path"]) / f"run{run_num}"
        if not run_dir.is_dir():
            print(f"[warn] Run does not exist for {config_name} (run {run_num})")
            continue

        rows = read_logs(run_dir / "logs.txt")
        if not rows:
            print(f"[warn] No logs found for {config_name} (run {run_num})")
            continue

        label = label_for_config(config_path, config)
        for target_flops in target_flops_list:
            target_flops = float(target_flops)
            row = find_row_for_target(
                rows,
                target_flops / 1e12,
                tolerance,
                loss_metric,
            )
            if row is None:
                print(
                    f"[warn] No {loss_metric} log row near "
                    f"{target_flops:.2e} FLOPs for {label}"
                )
                continue

            points.append(
                {
                    "config": config_name,
                    "label": label,
                    "run_num": run_num,
                    "target_flops": target_flops,
                    "params_m": params_m,
                    "train_loss": to_float(row, "train_loss"),
                    "val_loss": to_float(row, "val_loss"),
                    "iter": int(float(row["iter"])),
                    "tflops_consumed": float(row["tflops_consumed"]),
                }
            )
    return points


def fit_quadratic(xs: np.ndarray, ys: np.ndarray, fit_space: str) -> tuple[np.ndarray, np.ndarray] | None:
    if len(xs) < 3:
        return None
    positive = ys > 0
    xs = xs[positive]
    ys = ys[positive]
    if len(xs) < 3:
        return None

    fit_xs = np.log10(xs) if fit_space in {"log_params", "loglog"} else xs
    fit_ys = np.log10(ys) if fit_space == "loglog" else ys
    coeffs = np.polyfit(fit_xs, fit_ys, deg=2)
    grid = np.linspace(fit_xs.min(), fit_xs.max(), 200)
    fitted = np.polyval(coeffs, grid)
    plot_grid = 10**grid if fit_space in {"log_params", "loglog"} else grid
    if fit_space == "loglog":
        fitted = 10**fitted
    return plot_grid, fitted


def plot_points(
    points: list[dict],
    output: Path,
    fit_space: str,
    loss_metric: str,
) -> None:
    by_target = defaultdict(list)
    for point in points:
        by_target[point["target_flops"]].append(point)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    targets = sorted(by_target)
    cmap = plt.get_cmap("magma_r")
    colors = [cmap(i / max(len(targets) - 1, 1)) for i in range(len(targets))]

    for panel_idx, (ax, x_key, x_label) in enumerate(
        zip(axes, ("params_m", "iter"), ("Model size (M)", "Training iterations"))
    ):
        for target, color in zip(targets, colors):
            group = sorted(by_target[target], key=lambda point: point[x_key])
            xs = np.array([point[x_key] for point in group], dtype=float)
            ys = np.array([point[loss_metric] for point in group], dtype=float)
            label = f"{target:.0e}".replace("e+", "E")
            legend_label = label if panel_idx == 0 else None

            ax.scatter(
                xs,
                ys,
                color=color,
                label=legend_label if len(group) == 1 else None,
                s=POINT_SIZE,
                zorder=3,
            )
            fit = fit_quadratic(xs, ys, fit_space)
            if fit is not None:
                fit_xs, fit_ys = fit
                ax.plot(
                    fit_xs,
                    fit_ys,
                    color=color,
                    linewidth=1.1,
                    label=legend_label,
                )
            elif len(group) == 2:
                order = np.argsort(xs)
                ax.plot(
                    xs[order],
                    ys[order],
                    color=color,
                    linewidth=1.1,
                    label=legend_label,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _position: (
                    np.format_float_positional(float(value), unique=True, trim="-")
                    if np.isfinite(value) and value > 0
                    else ""
                )
            )
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", bottom=True)

    loss_label = "Training loss" if loss_metric == "train_loss" else "Validation loss"
    axes[0].set_ylabel(loss_label)
    axes[0].legend(title="FLOPs", loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {display_path(output)}")


def write_summary(points: list[dict], path: Path) -> None:
    if not points:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "label",
        "run_num",
        "target_flops",
        "params_m",
        "train_loss",
        "val_loss",
        "iter",
        "tflops_consumed",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)
    print(f"Wrote summary: {display_path(path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="isoflop.png")
    parser.add_argument("--summary-csv", default="isoflop_points.csv")
    parser.add_argument(
        "--fit-space",
        choices=["loglog", "log_params", "params"],
        default="loglog",
        help="Fit the quadratic in log-log space, log10(parameter count), or raw parameter count.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Maximum relative difference between logged TFLOPs and the target.",
    )
    parser.add_argument(
        "--loss-metric",
        choices=["val_loss", "train_loss"],
        default="val_loss",
        help="Loss to plot on the y axis (default: val_loss).",
    )
    args = parser.parse_args()

    points = collect_points(
        ISO_FLOP_CONFIGS,
        tolerance=args.tolerance,
        loss_metric=args.loss_metric,
    )
    if not points:
        raise RuntimeError("No isoFLOP points found. Have the runs produced logs.txt files?")

    write_summary(points, REPO_ROOT / args.summary_csv)
    plot_points(
        points,
        REPO_ROOT / args.output,
        args.fit_space,
        args.loss_metric,
    )


if __name__ == "__main__":
    main()
