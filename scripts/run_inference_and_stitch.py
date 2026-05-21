import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
import tifffile
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import PtychographyDataset  # noqa: E402
from model.model import PtychoViT  # noqa: E402
from development_logs.model512 import PtychoViT as PtychoViT512  # noqa: E402
from development_logs.model_cnn import PtychoCNN, PtychoCNN256  # noqa: E402
from utils.ptychi_utils import (  # noqa: E402
    make_hann_window,
    place_patches_fourier_shift,
    _place_patches_no_shift,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_model_and_size(config: dict):
    model_type = config.get("model", {}).get("model_type", "vit256")
    model_cfg = config.get("model", {})

    if model_type == "vit":
        model = PtychoViT512(config=model_cfg["vit"])
        img_size = model_cfg["vit"]["encoder"]["img_size"]
    elif model_type == "vit256":
        model = PtychoViT(config=model_cfg["vit256"])
        img_size = model_cfg["vit256"]["encoder"]["img_size"]
    elif model_type == "cnn":
        model = PtychoCNN(config=model_cfg["cnn"])
        img_size = 512
    elif model_type == "cnn256":
        model = PtychoCNN256(config=model_cfg["cnn256"])
        img_size = 256
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose 'vit', 'vit256', 'cnn', or 'cnn256'"
        )

    return model, img_size


def compute_dp_max(dp_dataset, chunk_size: int = 256) -> float:
    total = dp_dataset.shape[0]
    max_val = 0.0
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = dp_dataset[start:end]
        chunk_max = float(np.max(chunk))
        if chunk_max > max_val:
            max_val = chunk_max
    return max_val


def load_checkpoint(model, checkpoint_path: str, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)


def build_dataloader(data_path: str, config: dict, normalization_value: float, batch_size: int):
    data_cfg = config.get("data", {})
    model_type = config.get("model", {}).get("model_type", "vit256")
    target_size = data_cfg.get("target_size")
    if target_size is None:
        if model_type == "vit":
            target_size = config["model"]["vit"]["encoder"]["img_size"]
        elif model_type == "vit256":
            target_size = config["model"]["vit256"]["encoder"]["img_size"]
        elif model_type == "cnn256":
            target_size = 256
        else:
            target_size = 512

    dataset = PtychographyDataset(
        data_path,
        scale=data_cfg.get("scale", 100000.0),
        normalization_dict_path=None,
        default_normalization=1.0,
        apply_noise=data_cfg.get("apply_noise", False),
        cache_object=data_cfg.get("cache_object", True),
        max_probe_modes=data_cfg.get("max_probe_modes", 8),
        target_size=target_size,
        # [MODIFIED] Fix #9: forward the transpose flag from config so the dataset
        # extracts patches in the correct frame for HXN data.
        transpose_object_patches=data_cfg.get("transpose_object_patches", False),
    )
    dataset.normalization = normalization_value

    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": False,
    }
    return dataset, DataLoader(dataset, shuffle=False, **dataloader_kwargs)


def run_inference_and_stitch(
    model,
    dataloader,
    positions: torch.Tensor,
    object_shape,
    central_crop: int,
    pad: int,
    device: torch.device,
):
    # Canvases on `device` so FFT-based Fourier shifts run on GPU when available.
    # Phase uses circular mean: accumulate weighted cos/sin to avoid ±π wrap artefacts.
    pred_amp_object = torch.zeros(object_shape, device=device)
    pred_ph_cos     = torch.zeros(object_shape, device=device)  # Σ w·cos(φ)
    pred_ph_sin     = torch.zeros(object_shape, device=device)  # Σ w·sin(φ)
    # Weight accumulator stays on CPU — integer scatter-add, no FFT needed.
    weight_buffer   = torch.zeros(object_shape)  # CPU

    model.to(device)
    model.eval()

    hann_window = None  # built lazily from first batch's patch size
    hann_window_cpu = None

    positions = positions.to(device)

    scan_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", unit="batch"):
            diff_amp, _amp_patch, _ph_patch, probe, _probe_pos, norm, scale = batch
            b = diff_amp.size(0)

            input_diff  = diff_amp.to(device)
            input_probe = torch.view_as_real(probe.clone().detach()).to(device)
            input_norm  = norm.to(device)
            input_scale = scale.to(device)

            _output_diff, output_amp, output_ph = model(
                input_diff, input_probe, input_norm, input_scale
            )

            # Keep predictions on device — avoids CPU↔GPU round-trips.
            output_amp = output_amp.squeeze(1).detach()
            output_ph  = output_ph.squeeze(1).detach()

            # [MODIFIED] Fix #9: un-transpose HXN patches before placing into canvas.
            if getattr(dataloader.dataset, 'transpose_object_patches', False):
                output_amp = output_amp.transpose(-2, -1).contiguous()
                output_ph  = output_ph.transpose(-2, -1).contiguous()

            amp_patches = output_amp[:, central_crop:-central_crop, central_crop:-central_crop]
            ph_patches  = output_ph[:, central_crop:-central_crop, central_crop:-central_crop]

            # Build Hann window once from the first batch's patch dimensions.
            if hann_window is None:
                h_p, w_p = amp_patches.shape[-2:]
                hann_window     = make_hann_window(h_p, w_p, device=device)
                hann_window_cpu = hann_window.cpu()

            w_exp = hann_window.unsqueeze(0).expand(b, -1, -1)  # (B, H_p, W_p)

            batch_positions = positions[scan_idx : scan_idx + b]

            # Amplitude: Hann-weighted accumulation
            pred_amp_object = place_patches_fourier_shift(
                pred_amp_object, batch_positions, amp_patches * w_exp,
                op="add", adjoint_mode=False, pad=pad,
            )

            # Phase: circular mean — accumulate weighted cos and sin separately
            pred_ph_cos = place_patches_fourier_shift(
                pred_ph_cos, batch_positions, torch.cos(ph_patches) * w_exp,
                op="add", adjoint_mode=False, pad=pad,
            )
            pred_ph_sin = place_patches_fourier_shift(
                pred_ph_sin, batch_positions, torch.sin(ph_patches) * w_exp,
                op="add", adjoint_mode=False, pad=pad,
            )

            # Weight: integer scatter-add on CPU (no FFT needed for denominator)
            w_exp_cpu = hann_window_cpu.unsqueeze(0).expand(b, -1, -1)
            weight_buffer = _place_patches_no_shift(
                weight_buffer, batch_positions.cpu(), w_exp_cpu,
            )

            scan_idx += b

    # Finalize amplitude: divide by accumulated Hann weights
    safe_weight = torch.clamp(weight_buffer.to(device), min=1e-8)
    pred_amp_object = pred_amp_object / safe_weight

    # Finalize phase: circular mean = atan2(Σ w·sin φ, Σ w·cos φ)
    pred_ph_object = torch.atan2(pred_ph_sin, pred_ph_cos)

    return pred_amp_object.cpu(), pred_ph_object.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Run inference and stitch predicted patches into a whole-object image."
    )
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--data", required=True, help="Path to *_dp.hdf5 data file.")
    parser.add_argument("--output", required=True, help="Path to output TIFF.")
    parser.add_argument(
        "--output-kind",
        default="phase",
        choices=["phase", "amplitude"],
        help="Which stitched output to write.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (default: training batch size or 256).",
    )
    parser.add_argument(
        "--central-crop",
        type=int,
        default=None,
        help="Pixels to crop from each border before stitching (default: img_size // 4).",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=4,
        help="Padding for Fourier-shift placement (default: 4).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. 'cuda' or 'cpu'.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model, img_size = resolve_model_and_size(config)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = config.get("training", {}).get("batch_size", 256)

    if args.central_crop is None:
        central_crop = config.get("training", {}).get("test_plot_central_crop", img_size // 4)
    else:
        central_crop = args.central_crop

    with h5py.File(args.data, "r") as f:
        if "dp" not in f:
            raise KeyError(f"Missing 'dp' dataset in {args.data}")
        normalization_value = compute_dp_max(f["dp"])

    dataset, dataloader = build_dataloader(
        args.data, config, normalization_value, batch_size
    )

    if dataset._cached_probe_positions is None:
        dataset._cache_positions()

    object_shape = dataset.object_shape
    positions = dataset._cached_probe_positions

    load_checkpoint(model, args.checkpoint, device)

    pred_amp_object, pred_ph_object = run_inference_and_stitch(
        model,
        dataloader,
        positions,
        object_shape,
        central_crop,
        args.pad,
        device,
    )

    stitched = pred_ph_object if args.output_kind == "phase" else pred_amp_object
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tifffile.imwrite(args.output, stitched.numpy().astype(np.float32))


if __name__ == "__main__":
    main()
