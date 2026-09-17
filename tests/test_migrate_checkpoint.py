import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from ptycho_fm.model.decoders import Decoder256
from scripts.migrate_checkpoint import convert_decoder_output_to_linear


def test_converted_weights_load_strictly_and_preserve_decoder_output():
    model = nn.Module()
    for name in ("amp_decoder", "ph_decoder"):
        setattr(model, name, Decoder256(
            latent_dim=4, base_channels=2, out_channels=1,
            use_batchnorm=False, num_stages=1,
        ))
    state = model.state_dict()
    inputs = torch.randn(2, 4, 3, 3)
    expected = {}
    for name in ("amp_decoder", "ph_decoder"):
        decoder = getattr(model, name)
        features = decoder.stages[0](inputs)
        weight = state[f"{name}.output.weight"]
        bias = state[f"{name}.output.bias"]
        expected[name] = torch.nn.functional.conv2d(features, weight[..., None, None], bias)
        state[f"{name}.output.weight"] = weight[..., None, None]

    assert len(convert_decoder_output_to_linear(state)) == 2
    model.load_state_dict(state, strict=True)
    for name, output in expected.items():
        torch.testing.assert_close(getattr(model, name)(inputs), output)
    assert convert_decoder_output_to_linear(state) == []


def test_conversion_preserves_prefix_dtype_bias_and_unrelated_weights():
    weight = torch.randn(1, 3, 1, 1, dtype=torch.float64)
    bias = torch.randn(1)
    state = {
        "module.amp_decoder.output.weight": weight,
        "module.amp_decoder.output.bias": bias,
        "coupled_decoder.output.weight": weight,
    }
    convert_decoder_output_to_linear(state)
    torch.testing.assert_close(state["module.amp_decoder.output.weight"], weight[:, :, 0, 0])
    assert state["module.amp_decoder.output.bias"] is bias
    assert state["coupled_decoder.output.weight"] is weight


@pytest.mark.parametrize("shape", [(1, 3, 3, 3), (1, 3, 1)])
def test_conversion_rejects_unsupported_kernel(shape):
    with pytest.raises(ValueError, match="singleton kernel dimensions"):
        convert_decoder_output_to_linear({"amp_decoder.output.weight": torch.randn(*shape)})


@pytest.mark.parametrize("wrapper", [None, "state_dict", "model_state_dict", "model", "net", "weights"])
def test_cli_migrates_checkpoint_and_preserves_metadata(tmp_path, wrapper):
    state = {
        "log_scale_amp": torch.tensor(-2.0),
        "amp_decoder.output.weight": torch.randn(1, 3, 1, 1),
        "amp_decoder.output.bias": torch.randn(1),
    }
    checkpoint = state if wrapper is None else {wrapper: state, "epoch": 12}
    src, dst = tmp_path / "old.pth", tmp_path / "new.pth"
    torch.save(checkpoint, src)
    script = Path(__file__).resolve().parents[1] / "scripts" / "migrate_checkpoint.py"
    subprocess.run([
        sys.executable, str(script), str(src), str(dst),
        "--direction", "legacy-to-current", "--convert-decoder-output-to-linear",
        "--add-missing-current",
    ], check=True, capture_output=True, text=True)
    result = torch.load(dst, weights_only=True)
    if wrapper is not None:
        assert result["epoch"] == 12
        result = result[wrapper]
    assert result["amp_decoder.output.weight"].shape == (1, 3)
    assert "log_scale_amp" not in result
    torch.testing.assert_close(result["amp_scale"], state["log_scale_amp"])
    assert "ph_scale" in result and "amp_offset" in result
    original = torch.load(src, weights_only=True)
    assert (original if wrapper is None else original[wrapper])["amp_decoder.output.weight"].ndim == 4
