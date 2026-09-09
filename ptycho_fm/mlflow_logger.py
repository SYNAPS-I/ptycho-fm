"""MLflow integration for ptycho-fm training.

Wraps the MLflow client so main.py / training.py can call log_params,
log_metrics, log_artifact, register_best_model, and finish without sprinkling
`if mlflow_enabled` checks throughout the training loop. All methods are
no-ops when MLflow is disabled or when invoked from a non-rank-0 process.

Reference: gwbischof/mlflow-examples/mlflow_tool.py.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any


class MLflowLogger:
    def __init__(self, config: dict, is_main_process: bool):
        self.is_main_process = is_main_process
        mlflow_cfg = config.get('mlflow', {}) or {}
        self.enabled = bool(mlflow_cfg.get('enabled', False)) and is_main_process
        self.config = mlflow_cfg
        self.run = None
        self._mlflow = None
        self._client = None

        if not self.enabled:
            return

        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI') or mlflow_cfg.get('tracking_uri')
        if not tracking_uri:
            raise ValueError(
                "mlflow.enabled is true but no tracking URI is configured. "
                "Set MLFLOW_TRACKING_URI or mlflow.tracking_uri in config.yaml."
            )
        mlflow.set_tracking_uri(tracking_uri)

        self._apply_azureml_compat_shim()

        experiment_name = mlflow_cfg.get('experiment_name', 'ptycho-vit')
        mlflow.set_experiment(experiment_name)

        run_name = mlflow_cfg.get('run_name')
        if not run_name:
            run_num = config.get('trainer', {}).get('run_num', 'run')
            run_name = f"{run_num}-{datetime.now(tz=UTC).astimezone().strftime('%Y%m%d-%H%M%S')}"

        tags = mlflow_cfg.get('tags') or {}
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        self._mlflow = mlflow
        self._client = MlflowClient()
        print(f"MLflow run started: {self.run.info.run_id} (experiment: {experiment_name})", flush=True)

    @staticmethod
    def _apply_azureml_compat_shim() -> None:
        # Newer mlflow injects a `tag.mlflow.prompt.is_prompt != 'true'` filter on
        # search-model-versions calls that Azure ML's MLflow endpoint rejects.
        # Disabling the capability probe makes mlflow skip the filter.
        try:
            import mlflow.tracking._model_registry.client as _rc
            _rc.is_prompt_supported_registry = lambda *_a, **_k: False
        except (ImportError, AttributeError):
            pass

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        flat = _flatten_params(params)
        # MLflow caps individual param values at 6000 chars and rejects None.
        clean = {k: _stringify(v) for k, v in flat.items() if v is not None}
        if clean:
            self._mlflow.log_params(clean)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled:
            return
        clean = {k: float(v) for k, v in metrics.items() if v is not None and _is_finite(v)}
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return
        if not os.path.exists(path):
            print(f"MLflow: skipping log_artifact, path missing: {path}", flush=True)
            return
        self._mlflow.log_artifact(path, artifact_path=artifact_path)

    def register_best_model(self, model, model_name: str) -> None:
        """Log a PyTorch nn.Module via mlflow.pytorch and register a new version.

        Uses ``mlflow.pytorch.log_model`` so the registry entry includes the
        MLmodel descriptor (flavor, signature, conda env). The model should be
        on CPU, in eval mode, with the best weights already loaded; the caller
        is responsible for unwrapping DDP and loading ``best_model.pth``.

        Registration is done manually with the resolved artifact URI rather
        than the ``runs:/`` shorthand because Azure ML's model registry does
        not accept the latter (matches the workaround in mlflow_tool.py).
        """
        if not self.enabled:
            return
        if model is None:
            print("MLflow: no model provided; skipping registration", flush=True)
            return

        self._mlflow.pytorch.log_model(pytorch_model=model, artifact_path='model')
        artifact_uri = f"{self.run.info.artifact_uri}/model"

        from mlflow.exceptions import MlflowException
        try:
            self._client.get_registered_model(model_name)
        except MlflowException:
            self._client.create_registered_model(
                model_name,
                description="ptycho-vit fine-tuned model",
            )

        result = self._client.create_model_version(
            name=model_name,
            source=artifact_uri,
            run_id=self.run.info.run_id,
            description="Logged via mlflow.pytorch.log_model()",
        )
        print(f"MLflow: registered model '{result.name}' version {result.version}", flush=True)

    def finish(self, status: str = 'FINISHED') -> None:
        if not self.enabled:
            return
        self._mlflow.end_run(status=status)
        self.enabled = False


def _flatten_params(d: dict[str, Any], prefix: str = '', out: dict[str, Any] | None = None) -> dict[str, Any]:
    if out is None:
        out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            _flatten_params(v, key, out)
        else:
            out[key] = v
    return out


def _stringify(v: Any) -> str:
    s = str(v)
    return s if len(s) <= 6000 else s[:5997] + '...'


def _is_finite(v: Any) -> bool:
    try:
        import math
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False
