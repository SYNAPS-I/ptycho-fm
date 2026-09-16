import argparse
from pathlib import Path

import torch
from ptychi.image_proc import place_patches_fourier_shift
from torch.utils.data import DataLoader
from tqdm import tqdm

from ptycho_fm.data import CombinedDataset
from ptycho_fm.utils.inference import (
    build_ptychography_dataloader,
    crop_patch_borders,
    iter_object_patch_predictions,
    load_checkpoint,
    load_config,
    load_normalization_map,
    object_name_from_data_path,
    resolve_inference_model,
    resolve_normalization,
    save_stitched_outputs,
)

build_dataloader = build_ptychography_dataloader
resolve_model_and_size = resolve_inference_model


def run_inference_and_stitch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    positions: torch.Tensor,
    object_shape,
    central_crop: int,
    pad: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_amp_object = torch.zeros(object_shape, device="cpu")
    pred_ph_object = torch.zeros(object_shape, device="cpu")
    buffer = torch.zeros(object_shape, device="cpu")

    scan_idx = 0
    batches = tqdm(dataloader, desc="Inference", unit="batch")
    for output_amp, output_ph in iter_object_patch_predictions(model, batches, device):
        batch_size = output_amp.size(0)
        amp_patches = crop_patch_borders(output_amp, central_crop)
        ph_patches = crop_patch_borders(output_ph, central_crop)
        batch_positions = positions[scan_idx : scan_idx + batch_size]

        pred_amp_object = place_patches_fourier_shift(
            pred_amp_object,
            batch_positions,
            amp_patches,
            op="add",
            adjoint_mode=False,
            pad=pad,
        )
        pred_ph_object = place_patches_fourier_shift(
            pred_ph_object,
            batch_positions,
            ph_patches,
            op="add",
            adjoint_mode=False,
            pad=pad,
        )
        buffer = place_patches_fourier_shift(
            buffer,
            batch_positions,
            torch.ones_like(ph_patches),
            op="add",
            adjoint_mode=False,
            pad=pad,
        )
        scan_idx += batch_size

    if scan_idx != len(positions):
        raise ValueError(
            f"Stitched {scan_idx} predictions but received {len(positions)} positions"
        )
    pred_amp_object = pred_amp_object / torch.clip(buffer, min=1)
    pred_ph_object = pred_ph_object / torch.clip(buffer, min=1)
    return pred_amp_object, pred_ph_object


def discover_data_files(input_path: str | Path, num_files: int | None) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(CombinedDataset.find_paired_files(input_path))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    for path in files:
        object_name_from_data_path(path)
    if num_files is not None:
        if num_files < 1:
            raise ValueError(f"--num-files must be >= 1, got {num_files}")
        files = files[:num_files]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference on one *_dp.hdf5 file or a directory of paired files, "
            "then save stitched amplitude and phase objects."
        )
    )
    parser.add_argument("input_path", help="Input *_dp.hdf5 file or data directory")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for output .npy files"
    )
    normalization_group = parser.add_mutually_exclusive_group()
    normalization_group.add_argument(
        "--normalization-file",
        help="Pickle mapping from object names to normalization values",
    )
    normalization_group.add_argument(
        "--normalization-value",
        type=float,
        help="Single-file normalization value",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size (default: training batch size or 256)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Process only the first N files from a directory",
    )
    parser.add_argument(
        "--central-crop",
        type=int,
        default=None,
        help="Pixels removed from each predicted patch border before stitching",
    )
    parser.add_argument(
        "--object-crop",
        type=int,
        default=100,
        help="Pixels removed from each stitched object border before saving (default: 100)",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=32,
        help="Padding for Fourier-shift placement (default: 32)",
    )
    parser.add_argument(
        "--apply-noise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override data.test_apply_noise from the config",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, for example 'cuda' or 'cpu'",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model, img_size = resolve_model_and_size(config)
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    load_checkpoint(model, args.checkpoint, device)

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = config.get("training", {}).get("batch_size", 256)
    if batch_size < 1:
        parser.error("--batch-size must be >= 1")

    central_crop = args.central_crop
    if central_crop is None:
        central_crop = config.get("training", {}).get(
            "test_plot_central_crop", img_size // 4
        )
    if central_crop < 0:
        parser.error("--central-crop must be >= 0")
    if args.object_crop < 0:
        parser.error("--object-crop must be >= 0")
    if args.pad < 0:
        parser.error("--pad must be >= 0")

    data_files = discover_data_files(args.input_path, args.num_files)
    if args.normalization_value is not None and not Path(args.input_path).is_file():
        parser.error("--normalization-value is only valid for single-file inference")
    normalization_map = (
        load_normalization_map(args.normalization_file)
        if args.normalization_file is not None
        else None
    )

    output_names = [object_name_from_data_path(path) for path in data_files]
    if len(output_names) != len(set(output_names)):
        parser.error(
            "input files contain duplicate object names, which would overwrite outputs"
        )

    for index, (data_path, object_name) in enumerate(
        zip(data_files, output_names, strict=True), start=1
    ):
        normalization_value = resolve_normalization(
            data_path,
            normalization_map=normalization_map,
            normalization_value=args.normalization_value,
        )
        print(
            f"[{index}/{len(data_files)}] Processing {object_name} "
            f"with normalization {normalization_value:g}"
        )
        dataset, dataloader = build_dataloader(
            data_path,
            config,
            normalization_value,
            batch_size,
            apply_noise=args.apply_noise,
        )
        try:
            pred_amp_object, pred_ph_object = run_inference_and_stitch(
                model,
                dataloader,
                dataset.get_probe_positions(),
                dataset.object_shape,
                central_crop,
                args.pad,
                device,
            )
            amp_path, ph_path = save_stitched_outputs(
                args.output_dir,
                object_name,
                pred_amp_object,
                pred_ph_object,
                args.object_crop,
            )
            print(f"Saved {amp_path}")
            print(f"Saved {ph_path}")
        finally:
            dataset.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
