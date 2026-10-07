from types import SimpleNamespace

import pytest

from ptycho_fm import experiment_logger as tracking


def test_tracking_config_selects_at_most_one_backend():
    assert tracking.resolve_tracking_config({})["backend"] is None
    assert (
        tracking.resolve_tracking_config({"trainer": {"run_name": "shared-run"}, "wandb": {"enabled": True}})["backend"]
        == "wandb"
    )
    assert (
        tracking.resolve_tracking_config({"trainer": {"run_name": "shared-run"}, "mlflow": {"enabled": True}})["backend"]
        == "mlflow"
    )

    with pytest.raises(ValueError, match="Enable only one"):
        tracking.resolve_tracking_config(
            {
                "wandb": {"enabled": True},
                "mlflow": {"enabled": True},
            }
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_tracking_config_rejects_invalid_batch_interval(value):
    with pytest.raises(ValueError, match="positive integer or null"):
        tracking.resolve_tracking_config({"tracking": {"log_every_n_batches": value}})


def test_tracking_config_allows_disabling_batch_metrics():
    settings = tracking.resolve_tracking_config(
        {"tracking": {"log_every_n_batches": None}}
    )
    assert settings["log_every_n_batches"] is None


def test_mlflow_and_wandb_receive_the_same_clean_metrics(monkeypatch):
    mlflow_calls = []

    class FakeMLflowLogger:
        def __init__(self, config, is_main_process, *, log_system_metrics, run_name, resume):
            assert is_main_process
            assert log_system_metrics
            assert run_name == "shared-run"
            assert resume is False
            self.run = SimpleNamespace(info=SimpleNamespace(run_id="mlflow-run"))

        def log_params(self, params):
            pass

        def log_metrics(self, metrics, step=None):
            mlflow_calls.append((metrics, step))

        def log_artifact(self, path, artifact_path=None):
            pass

        def finish(self, status="FINISHED"):
            pass

    monkeypatch.setattr(tracking, "MLflowLogger", FakeMLflowLogger)
    mlflow_logger = tracking.ExperimentLogger(
        {"trainer": {"run_name": "shared-run"}, "mlflow": {"enabled": True}},
        True,
    )

    wandb_calls = []

    class FakeWandb:
        @staticmethod
        def login():
            pass

        @staticmethod
        def Settings(**kwargs):
            return kwargs

        @staticmethod
        def init(**kwargs):
            return SimpleNamespace(id="wandb-run", log_artifact=lambda artifact: None)

        @staticmethod
        def log(metrics, step=None):
            wandb_calls.append((metrics, step))

        @staticmethod
        def finish():
            pass

    monkeypatch.setitem(__import__("sys").modules, "wandb", FakeWandb)
    wandb_logger = tracking.ExperimentLogger(
        {"trainer": {"run_name": "shared-run"}, "wandb": {"enabled": True}},
        True,
    )

    payload = {"loss": 1, "missing": None, "invalid": float("nan")}
    mlflow_logger.log_metrics(payload, step=7)
    wandb_logger.log_metrics(payload, step=7)

    assert mlflow_calls == [({"loss": 1.0}, 7)]
    assert wandb_calls == mlflow_calls


def test_wandb_resume_looks_up_the_run_id_by_canonical_name(monkeypatch):
    init_kwargs = {}

    class ExistingRun:
        name = "20260923-143512"
        id = "resolved-id"

    class FakeApi:
        def runs(self, path, filters):
            assert path == "entity/project"
            assert filters == {"display_name": "20260923-143512"}
            return [ExistingRun()]

    class FakeWandb:
        @staticmethod
        def login():
            pass

        @staticmethod
        def Settings(**kwargs):
            return kwargs

        @staticmethod
        def Api():
            return FakeApi()

        @staticmethod
        def init(**kwargs):
            init_kwargs.update(kwargs)
            return SimpleNamespace(id=kwargs["id"], log_artifact=lambda artifact: None)

    monkeypatch.setitem(__import__("sys").modules, "wandb", FakeWandb)
    logger = tracking.ExperimentLogger(
        {
            "trainer": {"run_name": "20260923-143512"},
            "wandb": {"enabled": True, "entity": "entity", "project": "project"},
        },
        True,
        resume=True,
    )
    assert logger.run_id == "resolved-id"
    assert init_kwargs["name"] == "20260923-143512"
    assert init_kwargs["id"] == "resolved-id"
    assert init_kwargs["resume"] == "must"



def test_non_main_process_does_not_initialize_backend(monkeypatch):
    class UnexpectedMLflow:
        def __init__(self, *args, **kwargs):
            raise AssertionError("backend should not initialize off rank zero")

    monkeypatch.setattr(tracking, "MLflowLogger", UnexpectedMLflow)
    logger = tracking.ExperimentLogger(
        {"trainer": {"run_name": "shared-run"}, "mlflow": {"enabled": True}},
        False,
    )
    assert not logger.enabled
    assert logger.run_id is None


def test_mlflow_logger_writes_a_local_run(tmp_path, monkeypatch):
    from ptycho_fm.mlflow_logger import MLflowLogger

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    azureml_calls = []
    monkeypatch.setattr(
        MLflowLogger,
        "_apply_azureml_compat_shim",
        staticmethod(lambda: azureml_calls.append(True)),
    )
    artifact = tmp_path / "config.yaml"
    artifact.write_text("training: {}\n")
    logger = MLflowLogger(
        {
            "mlflow": {
                "enabled": True,
                "tracking_uri": (tmp_path / "mlruns").as_uri(),
                "experiment_name": "test-experiment",
            }
        },
        True,
        log_system_metrics=False,
        run_name="test-run",
    )
    run_id = logger.run.info.run_id
    logger.log_params({"nested": {"value": 3}})
    logger.log_metrics({"loss": 1.25}, step=4)
    logger.log_artifact(str(artifact), artifact_path="config")
    logger.finish()

    run = logger._client.get_run(run_id)
    assert run.data.params["nested.value"] == "3"
    assert run.data.metrics["loss"] == pytest.approx(1.25)
    assert run.info.status == "FINISHED"
    assert [item.path for item in logger._client.list_artifacts(run_id, "config")] == [
        "config/config.yaml"
    ]
    assert azureml_calls == []

    resumed = MLflowLogger(
        {
            "mlflow": {
                "enabled": True,
                "tracking_uri": (tmp_path / "mlruns").as_uri(),
                "experiment_name": "test-experiment",
            }
        },
        True,
        log_system_metrics=False,
        run_name="test-run",
        resume=True,
    )
    assert resumed.run.info.run_id == run_id
    resumed.finish()

    azure_logger = MLflowLogger(
        {
            "mlflow": {
                "enabled": True,
                "tracking_uri": (tmp_path / "mlruns").as_uri(),
                "experiment_name": "test-experiment",
                "azureml_compat": True,
            }
        },
        True,
        log_system_metrics=False,
        run_name="azure-test-run",
    )
    azure_logger.finish()
    assert azureml_calls == [True]
