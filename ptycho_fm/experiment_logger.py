"""Backend-neutral experiment tracking for PtychoFM training."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from ptycho_fm.mlflow_logger import MLflowLogger
from ptycho_fm.utils.config import resolve_run_name

_TRACKING_DEFAULTS = {
    "log_parameters": True,
    "log_metrics": True,
    "log_artifacts": True,
    "log_system_metrics": True,
    "log_every_n_batches": 50,
}


def resolve_tracking_config(config: dict) -> dict[str, Any]:
    """Return validated shared settings and the selected tracking backend."""
    settings = {**_TRACKING_DEFAULTS, **(config.get("tracking", {}) or {})}
    for key in ("log_parameters", "log_metrics", "log_artifacts", "log_system_metrics"):
        if not isinstance(settings[key], bool):
            raise TypeError(f"tracking.{key} must be true or false")

    interval = settings["log_every_n_batches"]
    if interval is not None and (
        not isinstance(interval, int) or isinstance(interval, bool) or interval < 1
    ):
        raise ValueError(
            "tracking.log_every_n_batches must be a positive integer or null"
        )

    enabled = [
        name
        for name in ("wandb", "mlflow")
        if bool((config.get(name, {}) or {}).get("enabled", False))
    ]
    if len(enabled) > 1:
        raise ValueError(
            "Enable only one experiment tracker: comment out either the "
            "wandb or mlflow configuration block."
        )
    settings["backend"] = enabled[0] if enabled else None
    return settings


class ExperimentLogger:
    """Expose the same logging surface for W&B and MLflow."""

    def __init__(
        self,
        config: dict,
        is_main_process: bool,
        *,
        params: dict[str, Any] | None = None,
        config_path: str | None = None,
        resume: bool = False,
    ):
        self.settings = resolve_tracking_config(config)
        self.backend = self.settings["backend"]
        self.enabled = bool(self.backend) and is_main_process
        self.log_every_n_batches = self.settings["log_every_n_batches"]
        self.run = None
        self.run_id = None
        self._wandb = None
        self._mlflow = None

        if not self.enabled:
            return

        run_name = resolve_run_name(config, generate=False)

        if self.backend == "wandb":

            self._start_wandb(config["wandb"], params, run_name, resume)
        else:
            self._mlflow = MLflowLogger(
                config,
                is_main_process=True,
                log_system_metrics=self.settings["log_system_metrics"],
                run_name=run_name,
                resume=resume,
            )
            self.run = self._mlflow.run
            self.run_id = self.run.info.run_id
            if (
                params is not None
                and self.settings["log_parameters"]
                and not resume
            ):
                self._mlflow.log_params(params)

        if config_path is not None:
            self.log_artifact(
                config_path, artifact_path="config", name="training-config"
            )

    def _start_wandb(
        self,
        backend_config: dict,
        params: dict[str, Any] | None,
        run_name: str,
        resume: bool,
    ) -> None:
        import wandb

        wandb.login()
        init_kwargs = {
            "entity": backend_config.get("entity"),
            "project": backend_config.get("project", "ptycho-fm"),
            "name": run_name,
            "config": (
                params if self.settings["log_parameters"] and not resume else None
            ),
            "settings": wandb.Settings(
                x_disable_stats=not self.settings["log_system_metrics"]
            ),
        }
        if resume:
            run_id = self._find_wandb_run_id(wandb, backend_config, run_name)
            init_kwargs.update(id=run_id, resume="must")
        self.run = wandb.init(**init_kwargs)
        self.run_id = self.run.id
        self._wandb = wandb

    @staticmethod
    def _find_wandb_run_id(wandb, backend_config: dict, run_name: str) -> str:
        entity = backend_config.get("entity")
        project = backend_config.get("project", "ptycho-fm")
        if not entity:
            raise ValueError("wandb.entity is required for name-based resume")
        runs = wandb.Api().runs(
            f"{entity}/{project}", filters={"display_name": run_name}
        )
        matches = [run for run in runs if run.name == run_name]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one W&B run named {run_name!r} in "
                f"{entity}/{project}, found {len(matches)}"
            )
        return matches[0].id


    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or not self.settings["log_metrics"]:
            return
        clean = {
            key: float(value)
            for key, value in metrics.items()
            if value is not None and _is_finite(value)
        }
        if not clean:
            return

        if self.backend == "wandb":
            self._wandb.log(clean, step=step)
        else:
            self._mlflow.log_metrics(clean, step=step)

    def log_artifact(
        self,
        path: str,
        *,
        artifact_path: str | None = None,
        name: str | None = None,
    ) -> None:
        if not self.enabled or not self.settings["log_artifacts"]:
            return
        if not os.path.exists(path):
            print(f"Experiment tracking: artifact path is missing: {path}", flush=True)
            return

        if self.backend == "wandb":
            artifact = self._wandb.Artifact(
                name=name or Path(path).stem,
                type=artifact_path or "file",
            )
            artifact.add_file(path)
            self.run.log_artifact(artifact)
        else:
            self._mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_image(
        self,
        key: str,
        path: str,
        *,
        caption: str,
        step: int | None,
        artifact_path: str,
    ) -> None:
        if not self.enabled or not self.settings["log_artifacts"]:
            return

        if self.backend == "wandb":
            self._wandb.log(
                {key: self._wandb.Image(path, caption=caption)},
                step=step,
            )
        else:
            self._mlflow.log_artifact(path, artifact_path=artifact_path)

    def finish(self, status: str = "FINISHED") -> None:
        if not self.enabled:
            return

        if self.backend == "wandb":
            self._wandb.finish()
        else:
            self._mlflow.finish(status=status)
        self.enabled = False


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
