import importlib
import os
import shlex
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).parents[1] / "scripts" / "iterative_reconstruction"
sys.path.insert(0, str(SCRIPT_DIR))
dispatch = importlib.import_module("recon_dispatch")


def reconstruction_options(**overrides):
    values = {
        "crop": (0, 255, 0, 255),
        "pixel_size": 6.65e-9,
        "engine": "lsqml",
        "opr": False,
        "num_opr_modes": 3,
        "total_variation": False,
        "total_variation_weight": 1e-2,
        "total_variation_start": 20,
        "total_variation_stride": 5,
        "continue_recon": False,
        "flip_x_positions": False,
        "filter_indices": None,
        "frame_stride": 1,
        "frame_sum": 1,
        "preprocess": False,
        "xray_energy_kev": None,
        "zp_diameter": None,
        "zp_outer_zone_width": None,
        "sample_detector_distance": None,
        "defocus_distance": None,
        "detector_pixel_size": None,
        "bin_factor": None,
        "max_position_jitter_m": None,
        "bin_pattern": "image_*.bin",
        "bin_frame_shape": None,
    }
    values.update(overrides)
    return dispatch.ReconstructionOptions(**values)


def recon_job(scan: int) -> object:
    return dispatch.ReconJob(
        scan=scan,
        data_path=f"/data/scan_{scan}.h5",
        data_dir="/data",
        data_format="h5",
        scan_glob=f"scan_{scan}.h5",
        bin_dir_template=".",
        probe_path="/data/probe.npy",
        positions_path="/data/positions.csv",
        output_dir=f"/output/{scan}",
        object_name=f"scan_{scan}",
        num_probe_modes=2,
    )


def test_build_reconstruction_command_includes_shared_options():
    options = reconstruction_options(
        engine="rpie",
        opr=True,
        num_opr_modes=4,
        total_variation=True,
        total_variation_weight=2e-2,
        total_variation_start=10,
        total_variation_stride=2,
        frame_stride=2,
        frame_sum=3,
        bin_factor=2,
    )

    command = dispatch.build_reconstruction_command(recon_job(7), options)

    assert command[0] == sys.executable
    assert command[1] == dispatch.PTYCHI_RECON
    assert command[command.index("--engine") + 1] == "rpie"
    assert command[command.index("--num-opr-modes") + 1] == "4"
    assert "--total-variation" in command
    assert command[command.index("--total-variation-weight") + 1] == "0.02"
    assert command[command.index("--total-variation-start") + 1] == "10"
    assert command[command.index("--total-variation-stride") + 1] == "2"
    assert command[command.index("--frame-stride") + 1] == "2"
    assert command[command.index("--frame-sum") + 1] == "3"
    assert command[command.index("--bin") + 1] == "2"


def test_launch_reconstruction_saves_an_executable_shell_script(tmp_path, monkeypatch):
    job = recon_job(7)
    job = dispatch.ReconJob(
        **{
            **job.__dict__,
            "output_dir": str(tmp_path),
        }
    )
    reconstruction = reconstruction_options()
    scheduler_options = dispatch.SchedulerOptions(
        allowed_gpus=None,
        excluded_gpus=set(),
        omp_num_threads=6,
        poll_interval=0,
        dry_run=False,
    )
    process = object()
    launched = {}

    def fake_popen(command, env):
        launched["command"] = command
        launched["env"] = env
        return process

    monkeypatch.setattr(dispatch.subprocess, "Popen", fake_popen)

    result = dispatch.launch_reconstruction(
        job, reconstruction, scheduler_options, gpu_id=3
    )

    script_path = tmp_path / "scan_7_launch.sh"
    script = script_path.read_text(encoding="utf-8")
    assert result is process
    assert os.access(script_path, os.X_OK)
    assert "export OMP_NUM_THREADS=6" in script
    assert "export CUDA_VISIBLE_DEVICES=3" in script
    assert f"cd -- {shlex.quote(os.getcwd())}" in script
    assert f"exec {shlex.join(launched['command'])}" in script
    assert launched["env"]["OMP_NUM_THREADS"] == "6"
    assert launched["env"]["CUDA_VISIBLE_DEVICES"] == "3"


def test_scheduler_reserves_a_different_gpu_for_each_active_job(monkeypatch):
    launched = []

    class RunningProcess:
        def poll(self):
            return None

    def fake_launch(job, reconstruction, scheduler, gpu_id):
        launched.append((job.scan, gpu_id))
        return RunningProcess()

    monkeypatch.setattr(dispatch, "get_available_gpus", lambda *_args: [0, 1])
    monkeypatch.setattr(dispatch, "launch_reconstruction", fake_launch)
    scheduler = dispatch.Scheduler(
        reconstruction=reconstruction_options(),
        options=dispatch.SchedulerOptions(
            allowed_gpus=None,
            excluded_gpus=set(),
            omp_num_threads=1,
            poll_interval=0,
            dry_run=False,
        ),
        pending=[recon_job(1), recon_job(2)],
    )

    scheduler.dispatch_pending()

    assert launched == [(1, 0), (2, 1)]
    assert [gpu_id for _process, _job, gpu_id in scheduler.active] == [0, 1]


def test_watch_mode_queues_previous_file_when_a_new_file_arrives(tmp_path):
    first = tmp_path / "scan_1_data.h5"
    second = tmp_path / "scan_2_data.h5"
    third = tmp_path / "scan_3_data.h5"
    for timestamp, path in enumerate((first, second, third), start=1):
        path.touch()
        os.utime(path, (timestamp, timestamp))

    state = dispatch.WatchState(
        known_files={str(first)},
        previous_file=str(first),
    )

    assert dispatch.observe_new_files(state, {str(first), str(second)}) == [str(first)]
    assert dispatch.observe_new_files(state, {str(first), str(second), str(third)}) == [
        str(second)
    ]
    assert state.previous_file == str(third)
    assert state.queue_previous_on_shutdown


def test_combined_parser_accepts_range_and_watch_arguments(tmp_path):
    probe_path = tmp_path / "probe.npy"
    positions_path = tmp_path / "positions.csv"
    probe_path.touch()
    positions_path.touch()
    shared = ["--crop", "0", "3", "0", "3"]

    range_args = dispatch.parse_args(
        [
            "range",
            "--data-root",
            str(tmp_path),
            "--scan-start",
            "1",
            "--scan-end",
            "2",
            "--probe-template",
            "probe.npy",
            "--positions-template",
            "positions.csv",
            *shared,
        ]
    )
    watch_args = dispatch.parse_args(
        [
            "watch",
            "--watch-dir",
            str(tmp_path),
            "--probe-path",
            str(probe_path),
            "--positions-path",
            str(positions_path),
            *shared,
        ]
    )

    assert range_args.mode == "range"
    assert watch_args.mode == "watch"


def test_total_variation_accepts_raar_engine():
    args = dispatch.parse_args(
        [
            "range",
            "--data-root",
            "/data",
            "--scan-start",
            "1",
            "--scan-end",
            "1",
            "--probe-template",
            "probe.npy",
            "--positions-template",
            "positions.csv",
            "--crop",
            "0",
            "3",
            "0",
            "3",
            "--engine",
            "raar",
            "--total-variation",
        ]
    )

    assert args.engine == "raar"
    assert args.total_variation


def test_yaml_config_can_supply_mode_and_all_required_range_arguments(tmp_path):
    config_path = tmp_path / "recon_config.yaml"
    config_path.write_text(
        """
mode: range
data_root: /data
scan_start: 4
scan_end: 8
probe_template: probe.npy
positions_template: positions.csv
crop: [0, 255, 0, 255]
engine: epie
total_variation: true
total_variation_weight: 0.03
total_variation_start: 15
total_variation_stride: 3
recursive: true
gpus: [0, 2]
""".strip(),
        encoding="utf-8",
    )

    args = dispatch.parse_args(
        ["--config", str(config_path), "--engine", "rpie", "--no-recursive"]
    )

    assert args.mode == "range"
    assert args.scan_start == 4
    assert args.crop == [0, 255, 0, 255]
    assert args.gpus == [0, 2]
    assert args.engine == "rpie"
    assert args.total_variation
    assert args.total_variation_weight == 0.03
    assert args.total_variation_start == 15
    assert args.total_variation_stride == 3
    assert not args.recursive
    assert args.config == str(config_path)


def test_yaml_config_supports_watch_mode(tmp_path):
    probe_path = tmp_path / "probe.npy"
    positions_path = tmp_path / "positions.csv"
    probe_path.touch()
    positions_path.touch()
    config_path = tmp_path / "recon_config.yaml"
    config_path.write_text(
        f"""
mode: watch
watch_dir: {tmp_path}
probe_path: {probe_path}
positions_path: {positions_path}
crop: [0, 3, 0, 3]
process_existing: true
""".strip(),
        encoding="utf-8",
    )

    args = dispatch.parse_args(["--config", str(config_path)])

    assert args.mode == "watch"
    assert args.process_existing
    assert args.watch_dir == str(tmp_path)


@pytest.mark.parametrize(
    "config_text, cli",
    [
        (
            """
mode: range
data_root: /data
scan_start: 1
scan_end: 1
probe_template: probe.npy
positions_template: positions.csv
crop: [0, 3, 0, 3]
unknown_option: value
""",
            [],
        ),
        (
            """
mode: watch
watch_dir: /data
probe_path: probe.npy
positions_path: positions.csv
crop: [0, 3, 0, 3]
""",
            ["range"],
        ),
    ],
)
def test_yaml_config_rejects_unknown_options_and_mode_conflicts(
    tmp_path, config_text, cli
):
    config_path = tmp_path / "recon_config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(SystemExit):
        dispatch.parse_args([*cli, "--config", str(config_path)])
