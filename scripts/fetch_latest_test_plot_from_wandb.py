#!/usr/bin/env python3
"""Fetch the latest test_plot image logged to Weights & Biases.

Defaults to entity/project in config.yaml unless overridden.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import wandb  # type: ignore
except Exception as exc:
    print(f"ERROR: wandb is required to run this script: {exc}", file=sys.stderr)
    sys.exit(1)


def _load_config(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    if yaml is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _extract_path(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        path = val.get("path")
        return path if isinstance(path, str) else None
    if isinstance(val, list):
        for item in reversed(val):
            path = _extract_path(item)
            if path:
                return path
    return None


def _pick_latest_run(
    api: "wandb.Api",
    entity: str,
    project: str,
    run_id: str | None,
    run_name: str | None,
):
    if run_id:
        return api.run(f"{entity}/{project}/{run_id}")

    runs = list(api.runs(f"{entity}/{project}"))
    if not runs:
        raise RuntimeError(f"No runs found for {entity}/{project}")

    if run_name:
        for r in runs:
            if r.name == run_name:
                return r
        raise RuntimeError(f"Run name not found: {run_name}")

    def _run_sort_key(r):
        return r.updated_at or r.created_at or ""

    runs.sort(key=_run_sort_key, reverse=True)
    return runs[0]


def _find_latest_test_plot_path(run) -> str | None:
    latest_val = None
    latest_step = None

    try:
        for row in run.scan_history(keys=["test_plot"]):
            val = row.get("test_plot")
            if val:
                latest_val = val
                latest_step = row.get("_step")
    except Exception:
        # History may be large or restricted; fall back to file scan.
        latest_val = None

    path = _extract_path(latest_val)
    if path:
        return path

    # Fallback: scan files for latest test_plot image.
    files = [
        f for f in run.files()
        if f.name.startswith("media/images/") and "test_plot" in os.path.basename(f.name)
    ]
    if not files:
        return None

    def _file_sort_key(f):
        return f.updated_at or f.created_at or f.name

    files.sort(key=_file_sort_key, reverse=True)
    return files[0].name


def _resolve_output_path(output: str, remote_path: str) -> str:
    if os.path.isdir(output) or output.endswith(os.sep):
        os.makedirs(output, exist_ok=True)
        return os.path.join(output, os.path.basename(remote_path))

    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return output


def fetch_latest_test_plot(
    api_key: str | None = None,
    *,
    entity: str | None = None,
    project: str | None = None,
    run_id: str | None = None,
    run_name: str | None = None,
    config_path: str = "config.yaml",
    output: str = "workspace/latest_test_plot.png",
) -> str:
    """Download latest test_plot image and return local file path.

    Args:
        api_key: W&B API key. If provided, sets WANDB_API_KEY for this call.
        entity: W&B entity (overrides config).
        project: W&B project (overrides config).
        run_id: Specific W&B run ID.
        run_name: Specific W&B run name (aka experiment name).
        config_path: Config file used for default entity/project.
        output: Output file path or directory.
    """
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key

    cfg = _load_config(config_path)
    wandb_cfg = cfg.get("wandb", {}) if isinstance(cfg, dict) else {}

    resolved_entity = entity or wandb_cfg.get("entity")
    resolved_project = project or wandb_cfg.get("project")

    if not resolved_entity or not resolved_project:
        raise RuntimeError("W&B entity/project not provided. Use parameters or config.yaml.")

    api = wandb.Api()
    run = _pick_latest_run(api, resolved_entity, resolved_project, run_id, run_name)
    remote_path = _find_latest_test_plot_path(run)
    if not remote_path:
        raise RuntimeError("No test_plot image found in the selected run.")

    output_path = _resolve_output_path(output, remote_path)
    wfile = run.file(remote_path)
    downloaded = wfile.download(root=os.path.dirname(output_path) or ".", replace=True)

    # If download path differs from requested filename, rename.
    if os.path.abspath(downloaded.name) != os.path.abspath(output_path):
        os.replace(downloaded.name, output_path)

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch latest test_plot image from W&B.")
    parser.add_argument("--api-key", default=None, help="W&B API key (optional)")
    parser.add_argument("--entity", default="SYNAPS-I", help="W&B entity (overrides config)")
    parser.add_argument("--project", default="PtychoVit", help="W&B project (overrides config)")
    parser.add_argument("--run-id", default=None, help="Specific W&B run ID")
    parser.add_argument("--run-name", default=None, help="Specific W&B run name (experiment name)")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml for default entity/project",
    )
    parser.add_argument(
        "--output",
        default="latest_test_plot.png",
        help="Output file path or directory",
    )
    args = parser.parse_args()

    try:
        output_path = fetch_latest_test_plot(
            api_key=args.api_key,
            entity=args.entity,
            project=args.project,
            run_id=args.run_id,
            run_name=args.run_name,
            config_path=args.config,
            output=args.output,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Saved latest test_plot to {output_path}")
    print(f"Fetched at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
