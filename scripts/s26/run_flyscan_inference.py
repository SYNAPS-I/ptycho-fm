"""Run object-patch inference and integer-grid stitching for S26 fly scans."""

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ptycho_fm.data import CombinedDataset
from ptycho_fm.utils.inference import (
    build_ptychography_dataloader,
    crop_patch_borders,
    iter_object_patch_predictions,
    load_checkpoint,
    load_config,
    load_normalization_map,
    make_inference_dataloader,
    object_name_from_data_path,
    resolve_inference_model,
    resolve_normalization,
    save_stitched_outputs,
    validate_normalization,
)


class RawH5FlyscanDataset(Dataset):
    """Read cropped detector frames from an S26 raw HDF5 fly scan."""

    def __init__(
        self,
        file_path: str | Path,
        crop: tuple[int, int, int, int] | list[int],
        normalization: float,
        *,
        dataset_path: str = "/entry/data/data",
        scale: float = 10000.0,
        binning: int = 1,
        apply_noise: bool = False,
    ):
        if crop is None:
            raise ValueError("--raw-crop is required for raw HDF5 input")
        if binning < 1:
            raise ValueError(f"--bin must be >= 1, got {binning}")

        self.file_path = Path(file_path)
        self.dataset_path = dataset_path
        self.xmin, self.xmax, self.ymin, self.ymax = crop
        self.normalization = validate_normalization(
            normalization, "normalization value"
        )
        self.scale = float(scale)
        self.binning = binning
        self.apply_noise = apply_noise
        self._handle = None

        if self.xmax < self.xmin or self.ymax < self.ymin:
            raise ValueError(f"Invalid --raw-crop bounds: {crop}")

        with h5py.File(self.file_path, "r") as file:
            if self.dataset_path not in file:
                raise KeyError(
                    f"Raw HDF5 file {self.file_path} does not contain "
                    f"dataset {self.dataset_path!r}"
                )
            data_shape = file[self.dataset_path].shape

        if len(data_shape) != 3:
            raise ValueError(
                f"Expected raw dataset {self.dataset_path!r} to have shape "
                f"(N, H, W), got {data_shape}"
            )
        if (
            self.xmin < 0
            or self.ymin < 0
            or self.xmax >= data_shape[2]
            or self.ymax >= data_shape[1]
        ):
            raise ValueError(
                f"--raw-crop {crop} is outside raw data shape {data_shape}"
            )

        self.num_patterns = data_shape[0]
        crop_height = self.ymax - self.ymin + 1
        crop_width = self.xmax - self.xmin + 1
        if crop_height % binning or crop_width % binning:
            raise ValueError(
                f"Raw crop shape {(crop_height, crop_width)} is not divisible "
                f"by --bin {binning}"
            )
        self.pattern_shape = (crop_height // binning, crop_width // binning)

    def __len__(self) -> int:
        return self.num_patterns

    def _get_handle(self):
        if self._handle is None:
            self._handle = h5py.File(self.file_path, "r")
        return self._handle

    def __getitem__(self, index: int):
        index = int(index)
        if index < 0 or index >= self.num_patterns:
            raise IndexError(index)

        frame = self._get_handle()[self.dataset_path][
            index, self.ymin : self.ymax + 1, self.xmin : self.xmax + 1
        ]
        frame = np.asarray(frame, dtype=np.float32)
        np.maximum(frame, 0, out=frame)
        if self.binning > 1:
            height, width = frame.shape
            frame = frame.reshape(
                height // self.binning,
                self.binning,
                width // self.binning,
                self.binning,
            ).sum(axis=(1, 3))

        intensity = (frame / self.normalization) * self.scale
        if self.apply_noise:
            intensity = np.random.default_rng().poisson(intensity).astype(np.float32)
        diffraction = torch.from_numpy(np.sqrt(intensity)).unsqueeze(0)
        amplitude = torch.zeros_like(diffraction)
        phase = torch.zeros_like(diffraction)
        probe = torch.zeros((1, 1, *self.pattern_shape), dtype=torch.complex64)
        position = torch.zeros(2, dtype=torch.float32)
        normalization = torch.tensor(self.normalization, dtype=torch.float32)
        scale = torch.tensor(self.scale, dtype=torch.float32)
        return diffraction, amplitude, phase, probe, position, normalization, scale

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def compute_raw_data_max(
    file_path: str | Path,
    crop: tuple[int, int, int, int] | list[int],
    *,
    dataset_path: str,
    binning: int,
    chunk_size: int = 256,
) -> float:
    """Compute the maximum after applying the S26 detector crop and binning."""
    if crop is None:
        raise ValueError("--raw-crop is required for raw HDF5 input")
    if binning < 1:
        raise ValueError(f"--bin must be >= 1, got {binning}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    xmin, xmax, ymin, ymax = crop
    if xmax < xmin or ymax < ymin:
        raise ValueError(f"Invalid --raw-crop bounds: {crop}")
    crop_height = ymax - ymin + 1
    crop_width = xmax - xmin + 1
    if crop_height % binning or crop_width % binning:
        raise ValueError(
            f"Raw crop shape {(crop_height, crop_width)} is not divisible "
            f"by --bin {binning}"
        )

    maximum = -np.inf
    with h5py.File(file_path, "r") as file:
        if dataset_path not in file:
            raise KeyError(
                f"Raw HDF5 file {file_path} does not contain dataset {dataset_path!r}"
            )
        dataset = file[dataset_path]
        if dataset.ndim != 3:
            raise ValueError(
                f"Expected raw dataset {dataset_path!r} to have shape (N, H, W), "
                f"got {dataset.shape}"
            )
        if xmin < 0 or ymin < 0 or xmax >= dataset.shape[2] or ymax >= dataset.shape[1]:
            raise ValueError(
                f"--raw-crop {crop} is outside raw data shape {dataset.shape}"
            )
        for start in range(0, dataset.shape[0], chunk_size):
            frames = np.asarray(
                dataset[
                    start : start + chunk_size,
                    ymin : ymax + 1,
                    xmin : xmax + 1,
                ],
                dtype=np.float32,
            )
            np.maximum(frames, 0, out=frames)
            if binning > 1:
                frames = frames.reshape(
                    frames.shape[0],
                    crop_height // binning,
                    binning,
                    crop_width // binning,
                    binning,
                ).sum(axis=(2, 4))
            maximum = max(maximum, float(frames.max()))
    return validate_normalization(maximum, "maximum diffraction intensity")


class GridAccumulator:
    """Accumulate amplitude and phase patches on an integer fly-scan grid."""

    def __init__(
        self,
        *,
        ny: int,
        nx: int,
        step_size: int,
        patch_shape: tuple[int, int],
        scan_pattern: str,
        flip_patch_y: bool,
    ):
        validate_grid_geometry(ny, nx, step_size, scan_pattern)
        patch_height, patch_width = patch_shape
        object_shape = (
            (ny - 1) * step_size + patch_height,
            (nx - 1) * step_size + patch_width,
        )
        self.ny = ny
        self.nx = nx
        self.step_size = step_size
        self.patch_shape = patch_shape
        self.scan_pattern = scan_pattern
        self.flip_patch_y = flip_patch_y
        self.amplitude = torch.zeros(object_shape, dtype=torch.float32)
        self.phase = torch.zeros(object_shape, dtype=torch.float32)
        self.count = torch.zeros(object_shape, dtype=torch.float32)
        self.samples = 0

    def add(
        self,
        amplitude_patches: torch.Tensor,
        phase_patches: torch.Tensor,
    ) -> None:
        if amplitude_patches.shape != phase_patches.shape:
            raise ValueError(
                "Amplitude and phase patch batches must have the same shape, got "
                f"{tuple(amplitude_patches.shape)} and {tuple(phase_patches.shape)}"
            )
        if tuple(amplitude_patches.shape[-2:]) != self.patch_shape:
            raise ValueError(
                f"Expected patch shape {self.patch_shape}, got "
                f"{tuple(amplitude_patches.shape[-2:])}"
            )

        patch_height, patch_width = self.patch_shape
        for amplitude, phase in zip(amplitude_patches, phase_patches, strict=True):
            if self.samples >= self.ny * self.nx:
                raise ValueError(
                    f"Received more than the expected {self.ny * self.nx} patches"
                )
            row, column = grid_position(self.samples, self.nx, self.scan_pattern)
            if self.flip_patch_y:
                amplitude = torch.flip(amplitude, dims=(0,))
                phase = torch.flip(phase, dims=(0,))
            y_start = row * self.step_size
            x_start = column * self.step_size
            selection = (
                slice(y_start, y_start + patch_height),
                slice(x_start, x_start + patch_width),
            )
            self.amplitude[selection] += amplitude
            self.phase[selection] += phase
            self.count[selection] += 1
            self.samples += 1

    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.ny * self.nx
        if self.samples != expected:
            raise ValueError(f"Expected {expected} patches, received {self.samples}")
        denominator = torch.clamp(self.count, min=1)
        return self.amplitude / denominator, self.phase / denominator


def validate_grid_geometry(ny: int, nx: int, step_size: int, scan_pattern: str) -> None:
    if ny < 1 or nx < 1:
        raise ValueError(f"ny and nx must be positive, got ny={ny}, nx={nx}")
    if step_size < 1:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if scan_pattern not in {"raster", "zigzag"}:
        raise ValueError(
            f"scan_pattern must be 'raster' or 'zigzag', got {scan_pattern!r}"
        )


def grid_position(index: int, nx: int, scan_pattern: str) -> tuple[int, int]:
    """Map acquisition order to a physical row and column."""
    row, acquisition_column = divmod(index, nx)
    if scan_pattern == "zigzag" and row % 2:
        return row, nx - 1 - acquisition_column
    return row, acquisition_column


def stitch_grid_patches(
    amplitude_patches: torch.Tensor,
    phase_patches: torch.Tensor,
    *,
    ny: int,
    nx: int,
    step_size: int,
    scan_pattern: str = "zigzag",
    flip_patch_y: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stitch complete patch tensors; useful for testing and small scans."""
    if amplitude_patches.ndim != 3:
        raise ValueError(
            "Expected amplitude patches with shape (N, H, W), got "
            f"{tuple(amplitude_patches.shape)}"
        )
    accumulator = GridAccumulator(
        ny=ny,
        nx=nx,
        step_size=step_size,
        patch_shape=tuple(amplitude_patches.shape[-2:]),
        scan_pattern=scan_pattern,
        flip_patch_y=flip_patch_y,
    )
    accumulator.add(amplitude_patches, phase_patches)
    return accumulator.finish()


def get_crop_sizes(
    crop_size: int,
    crop_size_range: tuple[int, int] | list[int] | None,
    crop_size_step: int,
) -> list[int]:
    """Resolve a single patch crop or an inclusive crop sweep."""
    if crop_size_range is None:
        crop_sizes = [crop_size]
    else:
        start, end = crop_size_range
        if crop_size_step < 1:
            raise ValueError(f"crop_size_step must be positive, got {crop_size_step}")
        if end < start:
            raise ValueError(
                f"crop_size_range END must be >= START, got {start}, {end}"
            )
        crop_sizes = list(range(start, end + 1, crop_size_step))
    if any(crop < 0 for crop in crop_sizes):
        raise ValueError(f"crop sizes must be non-negative, got {crop_sizes}")
    return crop_sizes


def run_grid_inference_and_stitch(
    model: torch.nn.Module,
    dataloader,
    *,
    crop_sizes: list[int],
    step_size: int,
    ny: int,
    nx: int,
    scan_pattern: str,
    flip_patch_y: bool,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Stream model predictions into one or more grid-stitch accumulators."""
    expected_samples = ny * nx
    if len(dataloader.dataset) != expected_samples:
        raise ValueError(
            f"Fly-scan grid ny*nx ({expected_samples}) does not match dataset "
            f"samples ({len(dataloader.dataset)})"
        )
    validate_grid_geometry(ny, nx, step_size, scan_pattern)

    accumulators = None
    batches = tqdm(dataloader, desc="Inference", unit="batch")
    for amplitude, phase in iter_object_patch_predictions(model, batches, device):
        if accumulators is None:
            patch_shape = tuple(amplitude.shape[-2:])
            for crop in crop_sizes:
                if 2 * crop >= patch_shape[0] or 2 * crop >= patch_shape[1]:
                    raise ValueError(
                        f"crop size {crop} removes all pixels from patch shape "
                        f"{patch_shape}"
                    )
            accumulators = {
                crop: GridAccumulator(
                    ny=ny,
                    nx=nx,
                    step_size=step_size,
                    patch_shape=(
                        patch_shape[0] - 2 * crop,
                        patch_shape[1] - 2 * crop,
                    ),
                    scan_pattern=scan_pattern,
                    flip_patch_y=flip_patch_y,
                )
                for crop in crop_sizes
            }
        for crop, accumulator in accumulators.items():
            accumulator.add(
                crop_patch_borders(amplitude, crop),
                crop_patch_borders(phase, crop),
            )

    if accumulators is None:
        raise ValueError("Cannot stitch an empty dataset")
    return {crop: accumulator.finish() for crop, accumulator in accumulators.items()}


def discover_input_files(
    input_path: str | Path,
    data_format: str,
    num_files: int | None,
) -> tuple[str, list[Path]]:
    """Discover either paired Ptychodus or raw S26 HDF5 inputs."""
    input_path = Path(input_path)
    if num_files is not None and num_files < 1:
        raise ValueError(f"--num-files must be >= 1, got {num_files}")

    if input_path.is_file():
        files = [input_path]
        if data_format == "auto":
            if input_path.name.endswith("_dp.hdf5"):
                data_format = "ptychodus"
            elif input_path.suffix.lower() in {".h5", ".hdf5"}:
                data_format = "raw-h5"
            else:
                raise ValueError(f"Cannot infer data format from {input_path}")
    elif input_path.is_dir():
        if data_format in {"auto", "ptychodus"}:
            try:
                paired_files = list(CombinedDataset.find_paired_files(input_path))
            except ValueError:
                paired_files = []
        else:
            paired_files = []
        raw_files = sorted(
            path
            for pattern in ("*.h5", "*.hdf5")
            for path in input_path.rglob(pattern)
            if not path.stem.endswith(("_dp", "_para"))
        )
        if data_format == "auto":
            if paired_files and raw_files:
                raise ValueError(
                    "Directory contains both paired Ptychodus and raw HDF5 files; "
                    "select --data-format explicitly"
                )
            if paired_files:
                data_format, files = "ptychodus", paired_files
            else:
                data_format, files = "raw-h5", raw_files
        elif data_format == "ptychodus":
            files = paired_files
        else:
            files = raw_files
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not files:
        raise FileNotFoundError(f"No {data_format} input files found in {input_path}")
    if num_files is not None:
        files = files[:num_files]
    return data_format, files


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run S26 fly-scan inference and integer-grid stitching."
    )
    parser.add_argument("input_path", help="Raw HDF5 file or paired-data directory")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for stitched .npy outputs"
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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-files", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=104)
    parser.add_argument(
        "--crop-size-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Inclusive patch-crop range, overriding --crop-size",
    )
    parser.add_argument("--crop-size-step", type=int, default=1)
    parser.add_argument("--step-size", type=int, default=14)
    parser.add_argument("--ny", type=int, default=201)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument(
        "--scan-pattern", choices=("raster", "zigzag"), default="zigzag"
    )
    parser.add_argument(
        "--flip-patch-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip each predicted patch vertically before placement (default: enabled)",
    )
    parser.add_argument(
        "--data-format",
        choices=("auto", "ptychodus", "raw-h5"),
        default="auto",
    )
    parser.add_argument("--raw-dataset", default="/entry/data/data")
    parser.add_argument(
        "--raw-crop",
        type=int,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    parser.add_argument("--bin", type=int, default=1, dest="binning")
    parser.add_argument(
        "--apply-noise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override data.test_apply_noise for paired input; raw input defaults off",
    )
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    try:
        crop_sizes = get_crop_sizes(
            args.crop_size, args.crop_size_range, args.crop_size_step
        )
        data_format, data_files = discover_input_files(
            args.input_path, args.data_format, args.num_files
        )
        validate_grid_geometry(args.ny, args.nx, args.step_size, args.scan_pattern)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if args.normalization_value is not None and not Path(args.input_path).is_file():
        parser.error("--normalization-value is only valid for single-file inference")
    if args.binning < 1:
        parser.error("--bin must be >= 1")
    if data_format == "ptychodus" and args.binning != 1:
        parser.error("--bin is only valid for raw HDF5 input")
    if data_format == "raw-h5" and args.raw_crop is None:
        parser.error("--raw-crop is required for raw HDF5 input")

    config = load_config(args.config)
    model, image_size = resolve_inference_model(config)
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

    normalization_map = (
        load_normalization_map(args.normalization_file)
        if args.normalization_file is not None
        else None
    )
    object_names = [
        object_name_from_data_path(path) if data_format == "ptychodus" else path.stem
        for path in data_files
    ]
    if len(object_names) != len(set(object_names)):
        parser.error("input files contain duplicate object names")

    data_config = config.get("data", {})
    for index, (data_path, object_name) in enumerate(
        zip(data_files, object_names, strict=True), start=1
    ):
        if (
            data_format == "raw-h5"
            and normalization_map is None
            and args.normalization_value is None
        ):
            normalization = compute_raw_data_max(
                data_path,
                args.raw_crop,
                dataset_path=args.raw_dataset,
                binning=args.binning,
            )
        else:
            dataset_path = "dp" if data_format == "ptychodus" else args.raw_dataset
            normalization = resolve_normalization(
                data_path,
                normalization_map=normalization_map,
                normalization_value=args.normalization_value,
                object_name=object_name,
                dataset_path=dataset_path,
            )
        print(
            f"[{index}/{len(data_files)}] Processing {object_name} "
            f"with normalization {normalization:g}"
        )

        if data_format == "ptychodus":
            dataset, dataloader = build_ptychography_dataloader(
                data_path,
                config,
                normalization,
                batch_size,
                apply_noise=args.apply_noise,
            )
        else:
            apply_noise = False if args.apply_noise is None else args.apply_noise
            dataset = RawH5FlyscanDataset(
                data_path,
                args.raw_crop,
                normalization,
                dataset_path=args.raw_dataset,
                scale=data_config.get("scale", 10000.0),
                binning=args.binning,
                apply_noise=apply_noise,
            )
            dataloader = make_inference_dataloader(dataset, batch_size)

        try:
            if tuple(dataset.pattern_shape) != (image_size, image_size):
                raise ValueError(
                    f"Input pattern shape {tuple(dataset.pattern_shape)} does not "
                    f"match model image size {(image_size, image_size)}"
                )
            stitched = run_grid_inference_and_stitch(
                model,
                dataloader,
                crop_sizes=crop_sizes,
                step_size=args.step_size,
                ny=args.ny,
                nx=args.nx,
                scan_pattern=args.scan_pattern,
                flip_patch_y=args.flip_patch_y,
                device=device,
            )
            for crop, (amplitude, phase) in stitched.items():
                output_dir = (
                    Path(args.output_dir)
                    if len(crop_sizes) == 1
                    else Path(args.output_dir) / f"crop_{crop}"
                )
                amp_path, ph_path = save_stitched_outputs(
                    output_dir, object_name, amplitude, phase
                )
                print(f"Saved {amp_path}")
                print(f"Saved {ph_path}")
        finally:
            dataset.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
