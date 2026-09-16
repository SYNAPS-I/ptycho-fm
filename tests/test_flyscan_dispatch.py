"""Check file ownership, CLI compatibility, and child cleanup without GPUs."""

import argparse
from pathlib import Path

import pytest

from ptycho_fm.utils.dispatch import local_ranks
from scripts.s26 import flyscan_inference_dispatch as dispatch
from scripts.s26.run_flyscan_inference import create_parser


def options(**overrides):
    values = {
        "config": "config.yaml",
        "checkpoint": "model.pth",
        "output_dir": "output",
        "batch_size": 16,
        "crop_size": 12,
        "crop_size_step": 2,
        "crop_size_range": (10, 14),
        "step_size": 4,
        "ny": 2,
        "nx": 3,
        "data_format": "raw-h5",
        "apply_noise": False,
        "scan_pattern": "raster",
        "flip_patch_y": False,
        "binning": 2,
        "normalization_file": "norm.pkl",
        "raw_dataset": "/entry/data/data",
        "raw_crop": (0, 63, 0, 63),
        "gpus": [0, 1],
        "dry_run": False,
        "poll_interval": 0,
        "terminate_timeout": 1,
        "omp_num_threads": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_worker_command_parses_with_current_cli():
    command = dispatch.build_inference_command(options(), Path("shard"))
    args = create_parser().parse_args(command[2:])
    assert args.input_path == "shard"
    assert args.device == "cuda:0"
    assert args.binning == 2
    assert args.scan_pattern == "raster"
    assert args.flip_patch_y is False
    assert args.apply_noise is False
    assert args.normalization_file == "norm.pkl"
    assert args.crop_size_range == [10, 14]
    assert args.output_dir == "output"


def test_plan_transfers_to_another_node_and_detects_changed_files(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for name in ("a.h5", "b.h5", "c.h5"):
        (first / name).write_bytes(b"data")
        (second / name).write_bytes(b"data")
    files = dispatch.find_raw_h5_files(first)
    shards = dispatch.verify_global_sharding(files, 4)
    plan = tmp_path / "plan.json"
    fingerprint = dispatch.write_shard_plan(plan, "raw-h5", shards)
    loaded, loaded_fingerprint = dispatch.load_shard_plan(plan, second, "raw-h5", 4)
    assert fingerprint == loaded_fingerprint
    assert [[p.dp_path.name for p in s] for s in loaded] == [
        ["a.h5"],
        ["b.h5"],
        ["c.h5"],
        [],
    ]
    assert all(p.dp_path.parent == second for s in loaded for p in s)
    (second / "a.h5").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="wrong size"):
        dispatch.load_shard_plan(plan, second, "raw-h5", 4)


def test_duplicate_gpus_rejected():
    with pytest.raises(ValueError, match="unique"):
        local_ranks(0, 2, [0, 0])
    assert local_ranks(1, 2, [0, 1]) == [2, 3]


def test_launch_failure_reaps_already_started_child(monkeypatch, tmp_path):
    class Process:
        stopped = False
        waited = False

        def poll(self):
            return None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            self.waited = True

    process = Process()
    pair = dispatch.FilePair("a", tmp_path / "a.h5", None)
    jobs = [dispatch.RankJob(i, i, i, tmp_path, [pair]) for i in range(2)]

    def launch(job, args):
        if job.gpu_id == 1:
            raise OSError("launch failed")
        return process

    monkeypatch.setattr(dispatch, "launch_job", launch)
    monkeypatch.setattr(dispatch, "get_available_gpus", lambda *args: [0, 1])
    with pytest.raises(OSError, match="launch failed"):
        dispatch.run_jobs(jobs, options())
    assert process.stopped and process.waited


def test_busy_gpu_wait_and_failure_status(monkeypatch, tmp_path):
    class Process:
        def poll(self):
            return 7

    pair = dispatch.FilePair("a", tmp_path / "a.h5", None)
    job = dispatch.RankJob(0, 0, 0, tmp_path, [pair])
    availability = iter([[], [0]])
    launches = []
    monkeypatch.setattr(
        dispatch, "get_available_gpus", lambda *args: next(availability)
    )
    monkeypatch.setattr(dispatch.time, "sleep", lambda *_: None)

    def launch(job, args):
        launches.append(job)
        return Process()

    monkeypatch.setattr(dispatch, "launch_job", launch)
    assert dispatch.run_jobs([job], options()) == 1
    assert launches == [job]
