"""Export a trained PtychoViT checkpoint to an ONNX reconstruction model.

This exports the edge-inference script used by edge-ptycho-vit:

    diffraction amplitude [B, 1, H, W] -> amplitude, phase, or both

It intentionally bypasses PtychoViT.forward because the training forward also
requires probe, normalization, and scale tensors to synthesize diffraction.
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExportConfig:
    config_path: Path
    checkpoint_path: Path
    output_path: Path
    batch_size: int
    opset: int
    dynamic_batch: bool
    output_kind: str
    verify: bool


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(config).__name__}.")
    return config


def resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_dir():
        for filename in ("best_model.pth", "checkpoint_model.pth", "model.pth"):
            candidate = checkpoint_path / filename
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Checkpoint path is a directory, but no known weight file was found. "
            f"Tried best_model.pth, checkpoint_model.pth, and model.pth in {checkpoint_path}."
        )
    return checkpoint_path


def load_state_dict(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    try:
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            inner = obj.get(key)
            if isinstance(inner, dict):
                obj = inner
                break

    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()

    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected checkpoint to resolve to a state_dict mapping, got {type(obj).__name__}."
        )

    keys = list(obj.keys())
    for prefix in ("module.", "model."):
        if keys and all(isinstance(key, str) and key.startswith(prefix) for key in keys):
            obj = {key[len(prefix) :]: value for key, value in obj.items()}
            keys = list(obj.keys())

    return obj


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numel") and int(value.numel()) == 1 and hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return None
    return None


def extract_export_overrides(model_cfg: dict[str, Any]) -> dict[str, float | None]:
    scaling_cfg = model_cfg.get("scaling")
    if not isinstance(scaling_cfg, dict):
        scaling_cfg = {}

    amp_offset = optional_float(
        model_cfg.get(
            "amp_offset",
            model_cfg.get(
                "offset_amp",
                scaling_cfg.get("amp_offset", scaling_cfg.get("offset_amp")),
            ),
        )
    )

    amp_scale = optional_float(model_cfg.get("amp_scale", scaling_cfg.get("amp_scale")))
    if amp_scale is None:
        log_scale_amp = optional_float(scaling_cfg.get("log_scale_amp"))
        if log_scale_amp is not None:
            amp_scale = math.exp(log_scale_amp)

    ph_scale = optional_float(model_cfg.get("ph_scale", scaling_cfg.get("ph_scale")))
    if ph_scale is None:
        log_scale_ph = optional_float(scaling_cfg.get("log_scale_ph"))
        if log_scale_ph is not None:
            ph_scale = math.exp(log_scale_ph)

    return {
        "amp_offset": amp_offset,
        "amp_scale": amp_scale,
        "ph_scale": ph_scale,
    }


def resolve_model_config(
    raw_cfg: dict[str, Any],
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, float | None]]:
    """Normalize supported training config schemas into PtychoViT(config=...)."""

    model_cfg = raw_cfg.get("model", raw_cfg)
    if not isinstance(model_cfg, dict):
        raise TypeError(
            f"Expected config/model section to be a mapping, got {type(model_cfg).__name__}."
        )

    export_overrides = extract_export_overrides(model_cfg)

    if "encoder" in model_cfg and "decoder" in model_cfg:
        normalized = copy.deepcopy(model_cfg)
    else:
        model_type = model_cfg.get("model_type")
        if not model_type:
            raise ValueError(
                "Unsupported config format. Expected either model.encoder + model.decoder "
                "or model.model_type + model[model_type].encoder/decoder."
            )

        variant = model_cfg.get(str(model_type))
        if not isinstance(variant, dict):
            candidates = sorted(
                key
                for key, value in model_cfg.items()
                if isinstance(value, dict) and "encoder" in value and "decoder" in value
            )
            raise KeyError(
                "Expected model[model_type] to contain encoder/decoder settings, "
                f"but model_type={model_type!r} was not found. Available variants: {candidates}"
            )

        encoder = variant.get("encoder")
        decoder = variant.get("decoder")
        if not isinstance(encoder, dict) or not isinstance(decoder, dict):
            raise KeyError(
                f"Expected model variant {model_type!r} to contain mapping keys "
                "'encoder' and 'decoder'."
            )

        normalized = {
            "encoder_type": model_cfg.get("encoder_type", variant.get("encoder_type", "custom")),
            "encoder": copy.deepcopy(encoder),
            "decoder": copy.deepcopy(decoder),
        }
        for key in (
            "init",
            "amp_offset",
            "offset_amp",
            "amp_scale",
            "ph_scale",
            "subtract_probe_intensity",
        ):
            if key in variant:
                normalized[key] = copy.deepcopy(variant[key])
            elif key in model_cfg:
                normalized[key] = copy.deepcopy(model_cfg[key])

    resolve_relative_encoder_checkpoint(normalized, config_path)
    return normalized, export_overrides


def resolve_relative_encoder_checkpoint(model_cfg: dict[str, Any], config_path: Path) -> None:
    encoder_cfg = model_cfg.get("encoder")
    if not isinstance(encoder_cfg, dict):
        return

    checkpoint = encoder_cfg.get("checkpoint_path")
    if not checkpoint:
        return

    checkpoint_path = Path(str(checkpoint))
    if checkpoint_path.is_absolute() or checkpoint_path.exists():
        return

    for base in (config_path.parent, PROJECT_ROOT):
        candidate = base / checkpoint_path
        if candidate.exists():
            encoder_cfg["checkpoint_path"] = str(candidate)
            return


def sanitize_state_dict_for_model(
    model: Any,
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    filtered = {key: value for key, value in state_dict.items() if key in model_keys}
    extras = {key: value for key, value in state_dict.items() if key not in model_keys}

    missing = [key for key in model_state.keys() if key not in filtered]
    if missing:
        preview = ", ".join(missing[:20])
        if len(missing) > 20:
            preview += ", ..."
        raise RuntimeError(
            "Checkpoint is missing required model parameters. "
            f"Missing {len(missing)} key(s): {preview}"
        )

    return filtered, extras


def build_reconstruction_wrapper(
    model: Any,
    output_kind: str,
    *,
    amp_offset: float | None = None,
    amp_scale: float | None = None,
    ph_scale: float | None = None,
) -> Any:
    import torch
    import torch.nn as nn

    class PtychoViTReconstruction(nn.Module):
        def __init__(self, base_model: nn.Module) -> None:
            super().__init__()
            self.base = base_model
            self.output_kind = output_kind

            if amp_scale is not None:
                self.register_buffer(
                    "_amp_scale",
                    torch.tensor(float(amp_scale), dtype=torch.float32),
                )
            else:
                self._amp_scale = None

            if ph_scale is not None:
                self.register_buffer(
                    "_ph_scale",
                    torch.tensor(float(ph_scale), dtype=torch.float32),
                )
            else:
                self._ph_scale = None

            resolved_amp_offset = amp_offset
            if resolved_amp_offset is None:
                resolved_amp_offset = optional_float(getattr(base_model, "amp_offset", None))
            if resolved_amp_offset is None:
                resolved_amp_offset = 0.975
            self.register_buffer(
                "_amp_offset",
                torch.tensor(float(resolved_amp_offset), dtype=torch.float32),
            )

        def forward(self, diffraction: Any) -> Any:
            diffraction = 2 * torch.log10(diffraction + 1.0e-1)
            latent = self.base.encoder(diffraction)
            constrained_amp = self.base.amp_decoder(latent).squeeze(1)
            constrained_ph = self.base.ph_decoder(latent).squeeze(1)

            amp_scale_t = (
                self._amp_scale
                if self._amp_scale is not None
                else torch.exp(self.base.log_scale_amp)
            )
            ph_scale_t = (
                self._ph_scale
                if self._ph_scale is not None
                else torch.exp(self.base.log_scale_ph)
            )

            amp = (constrained_amp * amp_scale_t) + self._amp_offset
            ph = constrained_ph * ph_scale_t

            if self.output_kind == "amplitude":
                return amp.unsqueeze(1)
            if self.output_kind == "phase":
                return ph.unsqueeze(1)
            if self.output_kind == "amp_phase":
                return torch.stack([amp, ph], dim=1)
            raise ValueError(f"Unsupported output kind: {self.output_kind}")

    return PtychoViTReconstruction(model)


def export_onnx(config: ExportConfig) -> None:
    import torch

    from ptycho_vit.model.model import PtychoViT

    raw_cfg = load_yaml(config.config_path)
    model_cfg, export_overrides = resolve_model_config(raw_cfg, config.config_path)

    if bool(model_cfg.get("subtract_probe_intensity", False)):
        raise ValueError(
            "This edge-style ONNX export accepts only diffraction input, but the config has "
            "subtract_probe_intensity enabled and would need probe input for equivalent results."
        )

    model = PtychoViT(config=model_cfg)
    checkpoint_path = resolve_checkpoint_path(config.checkpoint_path)
    state_dict = load_state_dict(checkpoint_path)
    state_dict, extras = sanitize_state_dict_for_model(model, state_dict)

    if export_overrides.get("amp_offset") is None:
        for key in ("offset_amp", "amp_offset"):
            offset = optional_float(extras.get(key))
            if offset is not None:
                export_overrides["amp_offset"] = offset
                break

    if extras:
        preview = ", ".join(list(extras.keys())[:10])
        suffix = ", ..." if len(extras) > 10 else ""
        print(
            f"[convert_pt_to_onnx] Ignoring {len(extras)} checkpoint key(s) not present "
            f"in PtychoViT: {preview}{suffix}"
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    wrapper = build_reconstruction_wrapper(
        model,
        config.output_kind,
        amp_offset=export_overrides.get("amp_offset"),
        amp_scale=export_overrides.get("amp_scale"),
        ph_scale=export_overrides.get("ph_scale"),
    )
    wrapper.eval()

    img_size = int(model_cfg["encoder"]["img_size"])
    dummy = torch.rand(config.batch_size, 1, img_size, img_size, dtype=torch.float32)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if config.dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(config.output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=config.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    if config.verify:
        import onnx

        onnx_model = onnx.load(str(config.output_path))
        onnx.checker.check_model(onnx_model)

    print(f"Exported ONNX model to {config.output_path}")


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(
        description="Convert a trained PtychoViT .pth/.pt checkpoint to an ONNX model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the config.yaml used to instantiate the model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained weights or a run directory containing best_model.pth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ONNX path.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Static batch size for export.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with a dynamic batch axis.",
    )
    parser.add_argument(
        "--output-kind",
        choices=("phase", "amplitude", "amp_phase"),
        default="phase",
        help="Output tensor to export. amp_phase exports [B, 2, H, W].",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX checker validation after export.",
    )
    args = parser.parse_args()

    return ExportConfig(
        config_path=args.config.expanduser().resolve(),
        checkpoint_path=args.checkpoint.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        batch_size=args.batch_size,
        opset=args.opset,
        dynamic_batch=bool(args.dynamic_batch),
        output_kind=args.output_kind,
        verify=not bool(args.no_verify),
    )


def main() -> None:
    export_onnx(parse_args())


if __name__ == "__main__":
    main()
