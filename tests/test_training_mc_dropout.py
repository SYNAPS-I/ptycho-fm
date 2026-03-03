import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from training import (
    _mc_mean_std_from_samples,
    _mc_stochastic_eval_context,
    _normalize_test_plot_mc_config,
    _save_test_plot_mc_figure,
    _select_mc_indices,
    _stitch_mc_uncertainty,
    Trainer,
)


def test_mc_config_defaults():
    cfg = _normalize_test_plot_mc_config({})
    assert cfg["enabled"] is False
    assert cfg["ensemble_size"] == 30
    assert cfg["frame_stride"] == 8
    assert cfg["frame_offset"] == 0
    assert cfg["phase_std_vmax"] is None


@pytest.mark.parametrize(
    "raw,msg",
    [
        ({"test_plot_mc_ensemble_size": 1}, "ensemble_size must be >= 2"),
        ({"test_plot_mc_frame_stride": 0}, "frame_stride must be > 0"),
        ({"test_plot_mc_frame_stride": 4, "test_plot_mc_frame_offset": -1}, "0 <= frame_offset < frame_stride"),
        ({"test_plot_mc_frame_stride": 4, "test_plot_mc_frame_offset": 4}, "0 <= frame_offset < frame_stride"),
        ({"test_plot_mc_phase_std_vmax": 0}, "phase_std_vmax must be > 0 when provided"),
    ],
)
def test_mc_config_invalid(raw, msg):
    with pytest.raises(ValueError, match=msg):
        _normalize_test_plot_mc_config(raw)


def test_select_mc_indices_deterministic_modulo():
    idx = _select_mc_indices(global_start=10, batch_size=8, frame_stride=4, frame_offset=2)
    assert idx == [0, 4]


def test_mc_config_phase_std_vmax_override():
    cfg = _normalize_test_plot_mc_config({"test_plot_mc_phase_std_vmax": 0.7})
    assert cfg["phase_std_vmax"] == pytest.approx(0.7)


def test_mc_stochastic_eval_context_keeps_bn_eval_enables_dropout():
    model = nn.Sequential(nn.BatchNorm2d(4), nn.Dropout2d(0.1))
    model.eval()
    with _mc_stochastic_eval_context(model):
        assert model[0].training is False
        assert model[1].training is True
    assert model.training is False


def test_mc_stochastic_eval_context_restores_eval_on_exception():
    model = nn.Sequential(nn.BatchNorm2d(4), nn.Dropout(0.1))
    model.eval()
    try:
        with _mc_stochastic_eval_context(model):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert model.training is False


def test_mc_mean_std_shape_and_values():
    samples = torch.stack([torch.ones(2, 1, 4, 4), 3 * torch.ones(2, 1, 4, 4)], dim=0)
    mean, std = _mc_mean_std_from_samples(samples)
    assert mean.shape == (2, 1, 4, 4)
    assert std.shape == (2, 1, 4, 4)
    assert torch.allclose(mean, 2 * torch.ones_like(mean))


def test_save_test_plot_mc_figure_writes_png(tmp_path):
    amp_mean = torch.rand(64, 64)
    ph_mean = torch.rand(64, 64)
    amp_std = torch.rand(64, 64)
    ph_std = torch.rand(64, 64)
    out_path = tmp_path / "mc.png"
    _save_test_plot_mc_figure(amp_mean, ph_mean, amp_std, ph_std, str(out_path), crop=8)
    assert out_path.exists()


def test_save_test_plot_mc_figure_writes_png_with_phase_vmax(tmp_path):
    amp_mean = torch.rand(64, 64)
    ph_mean = torch.rand(64, 64)
    amp_std = torch.rand(64, 64)
    ph_std = torch.rand(64, 64)
    out_path = tmp_path / "mc_vmax.png"
    _save_test_plot_mc_figure(
        amp_mean,
        ph_mean,
        amp_std,
        ph_std,
        str(out_path),
        crop=8,
        phase_std_vmax=0.7,
    )
    assert out_path.exists()


def test_stitch_mc_uncertainty_uses_selected_only_normalization():
    object_size = (8, 8)
    positions = torch.tensor([[4.0, 4.0], [4.0, 4.0]])
    patches = torch.zeros(2, 4, 4)
    patches[0] = 2.0
    selected_idx = torch.tensor([0], dtype=torch.long)

    stitched, mc_buffer = _stitch_mc_uncertainty(
        object_size=object_size,
        positions=positions,
        patches=patches,
        selected_idx=selected_idx,
        pad=0,
    )

    assert torch.allclose(stitched[2:6, 2:6], 2.0 * torch.ones(4, 4))
    assert torch.all(mc_buffer[2:6, 2:6] == 1)


class _DummyTestDataset(Dataset):
    def __init__(self, num_samples: int = 2, patch_size: int = 8):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.pattern_shape = (patch_size, patch_size)
        self.object_shape = (500, 500)
        self._cached_probe_positions = torch.tensor(
            [[250.0, 250.0] for _ in range(num_samples)], dtype=torch.float32
        )

    def _cache_positions(self):
        return None

    def _cache_object_data(self):
        return None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        diff_amp = torch.ones(1, self.patch_size, self.patch_size, dtype=torch.float32)
        amp_patch = torch.ones(1, self.patch_size, self.patch_size, dtype=torch.float32)
        ph_patch = torch.zeros(1, self.patch_size, self.patch_size, dtype=torch.float32)
        probe = torch.ones(1, 1, self.patch_size, self.patch_size, dtype=torch.complex64)
        probe_pos = torch.tensor([0.0, 0.0], dtype=torch.float32)
        norm = torch.tensor(1.0, dtype=torch.float32)
        scale = torch.tensor(1.0, dtype=torch.float32)
        return diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale


class _DummyModel(nn.Module):
    def forward(self, input_diff, input_probe, input_norm, input_scale):
        batch_size, _, height, width = input_diff.shape
        amp = torch.ones((batch_size, 1, height, width), device=input_diff.device)
        ph = torch.zeros((batch_size, 1, height, width), device=input_diff.device)
        return input_diff, amp, ph


def _make_dummy_loader():
    dataset = _DummyTestDataset(num_samples=2, patch_size=8)
    return DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)


def test_generate_test_plot_zero_frames_selected_skips_mc(tmp_path):
    trainer = Trainer(
        model=_DummyModel(),
        mode="supervised",
        run_num="unit_zero",
        device=torch.device("cpu"),
        model_save_path=str(tmp_path),
        is_main_process=True,
        use_ddp=False,
        wandb_enabled=False,
    )

    trainer.generate_test_plot(
        _make_dummy_loader(),
        epoch=0,
        filename="test_epoch0.png",
        central_crop=1,
        ph_crop=1,
        mc_cfg={
            "enabled": True,
            "ensemble_size": 2,
            "frame_stride": 100,
            "frame_offset": 99,
        },
    )

    run_path = tmp_path / "rununit_zero"
    assert (run_path / "test_epoch0.png").exists()
    assert not (run_path / "test_mc_epoch0.png").exists()


def test_generate_test_plot_baseline_unchanged_when_mc_disabled(tmp_path):
    trainer = Trainer(
        model=_DummyModel(),
        mode="supervised",
        run_num="unit_disabled",
        device=torch.device("cpu"),
        model_save_path=str(tmp_path),
        is_main_process=True,
        use_ddp=False,
        wandb_enabled=False,
    )

    trainer.generate_test_plot(
        _make_dummy_loader(),
        epoch=0,
        filename="test_epoch0.png",
        central_crop=1,
        ph_crop=1,
        mc_cfg={
            "enabled": False,
            "ensemble_size": 2,
            "frame_stride": 8,
            "frame_offset": 0,
        },
    )

    run_path = tmp_path / "rununit_disabled"
    assert (run_path / "test_epoch0.png").exists()
    assert not (run_path / "test_mc_epoch0.png").exists()
