import argparse
import glob
import os
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from ptycho_fm.utils.cli import parse_boolean

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_template_path(template: str, base_dir: str, scan_num: int) -> str:
    path = template.format(scan=scan_num, scan_padded=f"{scan_num:04d}")
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.abspath(path)


def get_scan_path(data_dir: str, scan_num: int, scan_glob: str) -> str:
    matches = glob.glob(resolve_template_path(scan_glob, data_dir, scan_num))
    if len(matches) == 0:
        raise FileNotFoundError(f"No scan file found for scan {scan_num} in {data_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple scan files found for scan {scan_num} in {data_dir}: {matches}")
    return matches[0]


def get_dp_path(data_dir: str, scan_num: int) -> str:
    if os.path.isfile(data_dir):
        if data_dir.endswith("_dp.hdf5"):
            return os.path.abspath(data_dir)
        raise ValueError(f"Expected --data-dir file to end with '_dp.hdf5': {data_dir}")

    patterns = [
        os.path.join(data_dir, f"*{scan_num}*_dp.hdf5"),
        os.path.join(data_dir, "*_dp.hdf5"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches) == 1:
            return os.path.abspath(matches[0])
        if len(matches) > 1:
            raise ValueError(
                f"Multiple saved diffraction pattern files found for scan {scan_num} "
                f"with pattern {pattern}: {matches}"
            )
    raise FileNotFoundError(f"No saved _dp.hdf5 file found for scan {scan_num} in {data_dir}")


def infer_object_name_from_dp_path(dp_path: str) -> str:
    stem = os.path.basename(dp_path)
    suffix = "_dp.hdf5"
    if not stem.endswith(suffix):
        raise ValueError(f"Cannot infer object name from non-_dp.hdf5 file: {dp_path}")
    return stem[: -len(suffix)]


def validate_frame_reduction(frame_stride: int, frame_sum: int) -> None:
    if frame_stride < 1:
        raise ValueError(f"--frame-stride must be >= 1, got {frame_stride}")
    if frame_sum < 1:
        raise ValueError(f"--frame-sum must be >= 1, got {frame_sum}")


def mask_negative_pixels(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint32:
        data = data.astype(np.int32)
    elif data.dtype == np.uint16:
        data = data.astype(np.int16)
    data[data < 0] = 0
    return data


def prepare_frames_for_summing(data: np.ndarray) -> np.ndarray:
    data = mask_negative_pixels(data)
    if np.issubdtype(data.dtype, np.integer) and data.dtype != np.int32:
        data = data.astype(np.int32)
    return data


def sum_frame_groups(data: np.ndarray, frame_sum: int, label: str) -> np.ndarray:
    data = prepare_frames_for_summing(data)
    if frame_sum == 1:
        return data

    group_count = data.shape[0] // frame_sum
    if group_count == 0:
        raise ValueError(
            f"Cannot sum {label}: only {data.shape[0]} frame(s), fewer than --frame-sum {frame_sum}"
        )

    kept_frames = group_count * frame_sum
    dropped_frames = data.shape[0] - kept_frames
    if dropped_frames:
        print(f"  Dropping {dropped_frames} trailing {label} frame(s) not divisible by --frame-sum {frame_sum}")

    grouped = data[:kept_frames].reshape(group_count, frame_sum, *data.shape[1:])
    sum_dtype = np.int32 if np.issubdtype(grouped.dtype, np.integer) else np.float32
    return grouped.sum(axis=1, dtype=sum_dtype)


def bin_diffraction_patterns(data: np.ndarray, bin_factor: int) -> np.ndarray:
    """Sum non-overlapping bin_factor x bin_factor detector-pixel blocks."""
    if bin_factor < 1:
        raise ValueError(f"--bin must be >= 1, got {bin_factor}")
    if data.ndim != 3:
        raise ValueError(f"Expected diffraction data with shape (N, H, W); got {data.shape}")
    if bin_factor == 1:
        return data

    frame_count, height, width = data.shape
    if height % bin_factor != 0 or width % bin_factor != 0:
        raise ValueError(
            f"Diffraction shape {data.shape} is not divisible by --bin {bin_factor}"
        )

    return data.reshape(
        frame_count,
        height // bin_factor,
        bin_factor,
        width // bin_factor,
        bin_factor,
    ).sum(axis=(2, 4))


def load_all_images(
    data_path: str,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    stride: int = 1,
    frame_sum: int = 1,
):
    """Load HDF5 data BEFORE torch/ptychi is imported."""
    validate_frame_reduction(stride, frame_sum)
    with h5py.File(data_path, "r") as f:
        # Cropping follows dude convention
        data = f["/entry/data/data"][::stride, ymin : ymax + 1, xmin : xmax + 1][()]
    return sum_frame_groups(data, frame_sum, "HDF5")


def load_dp_file(dp_path: str, stride: int = 1, frame_sum: int = 1):
    """Load diffraction patterns from saved Ptychodus-style _dp.hdf5 output."""
    validate_frame_reduction(stride, frame_sum)
    with h5py.File(dp_path, "r") as f:
        if "dp" not in f:
            raise KeyError(f"Diffraction pattern file {dp_path} does not contain dataset 'dp'")
        data = f["dp"][::stride]
    return sum_frame_groups(data, frame_sum, "saved diffraction")


def load_positions(positions_path: str):
    """Load positions in meters from CSV or saved Ptychodus-style _para.hdf5 output."""
    if positions_path.endswith((".hdf5", ".h5")):
        with h5py.File(positions_path, "r") as f:
            missing = [
                key
                for key in ("probe_position_y_m", "probe_position_x_m")
                if key not in f
            ]
            if missing:
                raise KeyError(
                    f"Positions file {positions_path} is missing required dataset(s): "
                    f"{', '.join(missing)}"
                )
            positions = np.stack(
                [f["probe_position_y_m"][()], f["probe_position_x_m"][()]], axis=1
            )
    else:
        positions = np.genfromtxt(positions_path, delimiter=",")

    positions = np.asarray(positions)
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"Expected positions in {positions_path} to have shape (N, 2); "
            f"got {positions.shape}"
        )
    return positions


def load_prepared_positions(
    positions_path: str,
    remove_indices=None,
) -> np.ndarray:
    positions = load_positions(positions_path)
    if remove_indices is not None:
        positions = np.delete(positions, remove_indices, axis=0)
    return positions


def add_position_jitter(
    positions: np.ndarray,
    max_jitter_m: float,
    rng=None,
) -> np.ndarray:
    """Return a copy of positions with independent uniform jitter in meters."""
    if max_jitter_m <= 0:
        raise ValueError(
            f"--max-position-jitter-m must be > 0, got {max_jitter_m}"
        )

    rng = np.random.default_rng() if rng is None else rng
    positions = np.asarray(positions)
    jitter = rng.uniform(-max_jitter_m, max_jitter_m, size=positions.shape)
    return positions + jitter


def calculate_stage_m_per_detector_px(
    energy_kev: float,
    zp_diameter_m: float,
    zp_outer_zone_width_m: float,
    sample_detector_distance_m: float,
    defocus_distance_m: float,
    detector_pixel_size_m: float,
    bin_factor: int,
) -> float:
    """Return zone-plate-stage meters per pixel in the binned detector image."""
    positive_inputs = {
        "--xray-energy-kev": energy_kev,
        "--zp-diameter": zp_diameter_m,
        "--zp-outer-zone-width": zp_outer_zone_width_m,
        "--sample-detector-distance": sample_detector_distance_m,
        "--detector-pixel-size": detector_pixel_size_m,
    }
    for name, value in positive_inputs.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    if not np.isfinite(defocus_distance_m):
        raise ValueError(
            f"--defocus-distance must be finite, got {defocus_distance_m}"
        )
    if not isinstance(bin_factor, (int, np.integer)) or bin_factor < 1:
        raise ValueError(f"--bin must be a positive integer, got {bin_factor}")

    wavelength_m = 1.239841984e-9 / energy_kev
    focal_length_m = zp_diameter_m * zp_outer_zone_width_m / wavelength_m
    zp_detector_distance_m = (
        focal_length_m + defocus_distance_m + sample_detector_distance_m
    )
    if zp_detector_distance_m <= 0:
        raise ValueError(
            "Zone-plate-to-detector distance must be > 0; check "
            "--sample-detector-distance and --defocus-distance"
        )

    detector_shift_gain = zp_detector_distance_m / focal_length_m
    binned_detector_pixel_size_m = detector_pixel_size_m * bin_factor
    return binned_detector_pixel_size_m / detector_shift_gain


def preprocess_weakly_scattering_data(
    data: np.ndarray,
    positions: np.ndarray,
    stage_m_per_detector_px: float,
) -> np.ndarray:
    """Apply weakly scattering sample flatfield correction."""
    if not np.isfinite(stage_m_per_detector_px) or stage_m_per_detector_px <= 0:
        raise ValueError("stage_m_per_detector_px must be finite and > 0")

    from numpy.fft import fft2, ifft2
    from scipy.ndimage import fourier_shift

    positions = np.asarray(positions).T / stage_m_per_detector_px
    if positions.shape[1] != data.shape[0]:
        raise ValueError(
            f"Positions count ({positions.shape[1]}) does not match data frame count ({data.shape[0]}) "
            "for preprocessing. Check --positions-path, --frame-stride, and --frame-sum."
        )

    diff_amp = np.sqrt(data).astype(np.float32)
    diff_mean = np.mean(diff_amp, axis=0)
    diff_shifted = np.zeros_like(diff_amp)
    diff_mean_shifted = np.zeros_like(diff_amp)

    for i in range(diff_amp.shape[0]):
        shift = [-positions[0, i], -positions[1, i]]
        diff_shifted[i] = np.abs(ifft2(fourier_shift(fft2(diff_amp[i]), shift)))
        diff_mean_shifted[i] = np.abs(ifft2(fourier_shift(fft2(diff_mean), shift)))

    flatfield = np.sum(diff_shifted, 0) / np.sum(diff_mean_shifted, 0)

    diff_corrected = np.zeros_like(diff_amp)
    for i in range(diff_amp.shape[0]):
        diff_corrected[i] = diff_amp[i] / np.abs(ifft2(fourier_shift(fft2(flatfield), [positions[0, i], positions[1, i]])))

    return (diff_corrected ** 2).astype(np.float32, copy=False)


def get_bin_scan_dir(data_dir: str, scan_num: int, bin_dir_template: str) -> str:
    path = resolve_template_path(bin_dir_template, data_dir, scan_num)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No binary scan directory found for scan {scan_num}: {path}")
    return path


def infer_frame_shape(first_data: np.ndarray) -> tuple:
    side = int(np.sqrt(first_data.size))
    if side * side == first_data.size:
        return (side, side)
    return first_data.shape


def load_bin_files(
    directory: str,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    shape=None,
    pattern: str = "image_*.bin",
    stride: int = 1,
    frame_sum: int = 1,
):
    validate_frame_reduction(stride, frame_sum)
    file_list = sorted(glob.glob(os.path.join(directory, pattern)))[::stride]
    if len(file_list) == 0:
        raise ValueError(f"No files matching {pattern!r} found in {directory}")

    first_data = np.fromfile(file_list[0], dtype=np.float32)
    frame_shape = tuple(shape) if shape is not None else infer_frame_shape(first_data)
    first_data = first_data.reshape(frame_shape)

    all_data = np.empty((len(file_list),) + first_data.shape, dtype=np.float32)
    all_data[0] = first_data

    for i, filepath in enumerate(file_list[1:], start=1):
        data = np.fromfile(filepath, dtype=np.float32).reshape(frame_shape)
        all_data[i] = data

    all_data = sum_frame_groups(all_data, frame_sum, "binary")

    print(f"Loaded {len(file_list)} binary files from {directory}")
    print(f"Final array shape: {all_data.shape}")
    return all_data


def detect_scan_format(
    data_dir: str,
    scan_num: int,
    data_format: str,
    scan_glob: str,
    bin_dir_template: str,
    bin_pattern: str,
) -> str:
    if data_format != "auto":
        return data_format

    try:
        get_dp_path(data_dir, scan_num)
        return "hdf5"
    except FileNotFoundError:
        pass

    h5_matches = glob.glob(resolve_template_path(scan_glob, data_dir, scan_num))
    bin_dir = resolve_template_path(bin_dir_template, data_dir, scan_num)
    bin_matches = glob.glob(os.path.join(bin_dir, bin_pattern)) if os.path.isdir(bin_dir) else []
    if h5_matches:
        return "h5"
    if bin_matches:
        return "bin"
    raise FileNotFoundError(
        f"No saved _dp.hdf5, raw HDF5 scan, or binary files found for scan {scan_num} under {data_dir}"
    )


def load_scans(
    data_dir: str,
    scan_start: int,
    scan_end: int,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    data_format: str = "auto",
    scan_glob: str = "scan_{scan}_*.h5",
    bin_dir_template: str = "{scan}/raw",
    bin_pattern: str = "image_*.bin",
    bin_frame_shape=None,
    frame_stride: int = 1,
    frame_sum: int = 1,
):
    """Load one or more scans and concatenate along the first axis."""
    if scan_end < scan_start:
        raise ValueError(f"scan-end ({scan_end}) must be >= scan-start ({scan_start})")
    arrays = []
    dp_paths = []
    for scan_num in range(scan_start, scan_end + 1):
        scan_format = detect_scan_format(
            data_dir, scan_num, data_format, scan_glob, bin_dir_template, bin_pattern
        )
        if scan_format == "hdf5":
            path = get_dp_path(data_dir, scan_num)
            print(f"  Loading saved diffraction patterns for scan {scan_num}: {path}")
            arrays.append(load_dp_file(path, frame_stride, frame_sum))
            dp_paths.append(path)
        elif scan_format == "h5":
            if None in (xmin, xmax, ymin, ymax):
                raise ValueError("--crop is required when loading raw HDF5 scan files")
            path = get_scan_path(data_dir, scan_num, scan_glob)
            print(f"  Loading HDF5 scan {scan_num}: {path}")
            arrays.append(load_all_images(path, xmin, xmax, ymin, ymax, frame_stride, frame_sum))
        elif scan_format == "bin":
            if None in (xmin, xmax, ymin, ymax):
                raise ValueError("--crop is required when loading binary scan files")
            path = get_bin_scan_dir(data_dir, scan_num, bin_dir_template)
            print(f"  Loading binary scan {scan_num}: {path}")
            arrays.append(load_bin_files(path, xmin, xmax, ymin, ymax, bin_frame_shape, bin_pattern, frame_stride, frame_sum))
        else:
            raise ValueError(f"Unknown data format: {scan_format}")
    data = np.concatenate(arrays, axis=0)
    if len(dp_paths) == 1:
        return data, infer_object_name_from_dp_path(dp_paths[0])
    return data, None


def save_dp(output_dir, object_name, data):
    """Save diffraction patterns to HDF5 (written once; data is not modified during reconstruction)."""
    os.makedirs(output_dir, exist_ok=True)
    dp_path = os.path.join(output_dir, f"{object_name}_dp.hdf5")
    with h5py.File(dp_path, "w") as f:
        f.create_dataset("dp", data=data)
    print(f"Saved diffraction patterns to {dp_path}")


def save_para(output_dir, object_name, pixel_size, data, recon_object, recon_probe, corrected_pos, epoch):
    """Save reconstruction parameters to HDF5 with epoch number in the filename."""
    os.makedirs(output_dir, exist_ok=True)
    para_path = os.path.join(output_dir, f"{object_name}_epoch{epoch}_para.hdf5")
    with h5py.File(para_path, "w") as f:
        f.create_dataset("object", data=recon_object)
        f["object"].attrs["center_x_m"] = corrected_pos[0, 1] * pixel_size
        f["object"].attrs["center_y_m"] = corrected_pos[0, 0] * pixel_size
        f["object"].attrs["pixel_height_m"] = pixel_size
        f["object"].attrs["pixel_width_m"] = pixel_size

        f.create_dataset("probe", data=recon_probe)
        f.create_dataset(
            "probe_position_indexes", data=np.arange(data.shape[0], dtype=np.int64)
        )
        f.create_dataset("probe_position_x_m", data=corrected_pos[:, 1] * pixel_size)
        f.create_dataset("probe_position_y_m", data=corrected_pos[:, 0] * pixel_size)
        f.create_dataset("normalization", data=np.float32(np.max(data)))
    print(f"Saved reconstruction parameters to {para_path}")


def run_ptychi_reconstruction(
    data,
    probe_path: str,
    positions_path: str,
    pixel_size_m: float,
    output_dir: str,
    object_name: str,
    num_probe_modes: int = 8,
    engine: str = "lsqml",
    continue_recon: bool = False,
    flip_x_positions: bool = False,
    frame_stride: int = 1,
    frame_sum: int = 1,
    remove_indices=None,
    reconstruction_positions=None,
    opr: bool = False,
    num_opr_modes: int = 3,
    total_variation: bool = False,
    total_variation_weight: float = 1e-2,
    total_variation_start: int = 20,
    total_variation_stride: int = 5,
):
    """Run ptychi reconstruction with pre-loaded data."""
    # Import torch and ptychi AFTER data is loaded
    import torch
    from ptychi import api
    from ptychi.api.task import PtychographyTask
    from ptychi.utils import (
        add_additional_opr_probe_modes_to_probe,
        generate_initial_opr_mode_weights,
        get_default_complex_dtype,
        get_suggested_object_size,
        rescale_probe,
    )

    class _ReconstructionRunner:
        def prepare_data(
            self, data, probe_path, positions_path, pixel_size_m, num_probe_modes, flip_x_positions, frame_stride, frame_sum, remove_indices, reconstruction_positions
        ):
            torch.set_default_device("cuda")

            data = torch.from_numpy(data)
            if probe_path.endswith(".csv"):
                probe = np.genfromtxt(probe_path, delimiter=",", dtype=np.complex64)
                pixels_per_mode = probe.size // num_probe_modes
                probe_side = int(np.sqrt(pixels_per_mode))
                if probe.size % num_probe_modes != 0 or probe_side * probe_side != pixels_per_mode:
                    raise ValueError(
                        f"Cannot reshape CSV probe {probe_path} with shape {probe.shape} "
                        f"into {num_probe_modes} square modes"
                    )
                probe = np.reshape(probe, (num_probe_modes, probe_side, probe_side))
                if probe_side == 512:
                    probe = probe[:, 128:-128, 128:-128]
                elif probe_side != 256:
                    raise ValueError(
                        f"Unsupported CSV probe mode size {probe_side}x{probe_side}. "
                        "Expected 256x256 or 512x512."
                    )
                probe = torch.from_numpy(probe).unsqueeze(
                    0
                )  # Dummy OPR modes dimension
            elif probe_path.endswith(".hdf5"):
                with h5py.File(probe_path, "r") as f:
                    probe = f["probe"][:]
            elif probe_path.endswith(".npy"):
                probe = np.load(probe_path)
            else:
                raise ValueError(
                    f"Unsupported probe file format: {probe_path}. Expected .csv or .hdf5 or .npy"
                )

            if reconstruction_positions is None:
                positions = load_prepared_positions(positions_path, remove_indices)
            else:
                positions = np.array(reconstruction_positions, copy=True)
            if flip_x_positions:
                positions[:, 1] = -positions[:, 1]
            if positions.shape[0] != data.shape[0]:
                raise ValueError(
                    f"Positions count ({positions.shape[0]}) does not match data frame count ({data.shape[0]}). "
                    "Check --positions-path, --filter-indices, and frame reduction options."
                )
            positions_px = positions / pixel_size_m
            return data, probe, positions_px

        def ptycho(
            self,
            data,
            probe_path: str,
            positions_path: str,
            pixel_size_m: float,
            output_dir: str,
            object_name: str,
            num_probe_modes: int = 8,
            engine: str = "lsqml",
            continue_recon: bool = False,
            flip_x_positions: bool = False,
            frame_stride: int = 1,
            frame_sum: int = 1,
            remove_indices=None,
            reconstruction_positions=None,
            opr: bool = False,
            num_opr_modes: int = 3,
            total_variation: bool = False,
            total_variation_weight: float = 1e-2,
            total_variation_start: int = 20,
            total_variation_stride: int = 5,
        ):
            data, probe, positions_px = self.prepare_data(
                data, probe_path, positions_path, pixel_size_m, num_probe_modes, flip_x_positions, frame_stride, frame_sum, remove_indices, reconstruction_positions
            )

            # Select options class based on engine
            if engine == "lsqml":
                options = api.LSQMLOptions()
            elif engine == "epie":
                options = api.EPIEOptions()
            elif engine == "rpie":
                options = api.RPIEOptions()
            elif engine == "dm":
                options = api.DMOptions()
            elif engine == "raar":
                options = api.RAAROptions()
            else:
                raise ValueError(f"Unknown engine: {engine}")

            projection_engine = engine in {"dm", "raar"}
            projection_engine_name = (
                "Difference Map" if engine == "dm" else "RAAR"
            )

            if opr and projection_engine:
                raise ValueError(
                    "OPR modes are only supported by non-projection engines "
                    "(lsqml, epie, and rpie)"
                )

            if projection_engine:
                if probe.ndim != 4 or probe.shape[0] != 1:
                    raise ValueError(
                        f"{projection_engine_name} requires a probe with shape "
                        "(1, n_modes, height, width): exactly one OPR mode. "
                        f"Got {tuple(probe.shape)}."
                    )
                if probe.shape[1] > 1:
                    print(
                        f"{projection_engine_name} selected: using only incoherent probe mode 0 "
                        f"of {probe.shape[1]} supplied modes"
                    )
                    probe = probe[:, :1, ...]

                # Correct a possible exposure mismatch before the projection
                # engine initializes its persistent exit waves. Pty-Chi performs
                # the propagation internally, so Fourier normalization is handled.
                probe = rescale_probe(probe, data)
                print(
                    f"Rescaled the initial {projection_engine_name} probe "
                    "to the active diffraction data"
                )
            elif opr:
                if num_opr_modes < 1:
                    raise ValueError(
                        f"num_opr_modes must be >= 1 when OPR is enabled, got {num_opr_modes}"
                    )
                if not isinstance(probe, torch.Tensor):
                    probe = torch.from_numpy(probe)
                if probe.ndim == 2:
                    probe = probe.unsqueeze(0).unsqueeze(0)
                elif probe.ndim == 3:
                    probe = probe.unsqueeze(0)
                elif probe.ndim != 4:
                    raise ValueError(
                        "OPR requires a probe with 2, 3, or 4 dimensions; "
                        f"got shape {tuple(probe.shape)}"
                    )
                probe = add_additional_opr_probe_modes_to_probe(
                    probe.to("cuda"), n_opr_modes_to_add=num_opr_modes
                )
                options.opr_mode_weight_options.initial_weights = (
                    generate_initial_opr_mode_weights(
                        len(positions_px), probe.shape[0]
                    )
                )
                options.opr_mode_weight_options.optimizable = True
                print(f"Enabled OPR with {num_opr_modes} additional probe mode(s)")

            # --- Object ---
            if probe_path.endswith(".hdf5") and continue_recon:
                with h5py.File(probe_path, "r") as f:
                    object_data = f["object"][:]
            else:
                object_data = torch.ones(
                    [
                        1,
                        *get_suggested_object_size(
                            positions_px, probe.shape[-2:], extra=100
                        ),
                    ],
                    dtype=get_default_complex_dtype(),
                )
            if projection_engine and (
                object_data.ndim != 3
                or object_data.shape[0] != 1
            ):
                raise ValueError(
                    f"{projection_engine_name} requires a single-slice object with shape "
                    "(1, height, width). Got "
                    f"{tuple(object_data.shape)}."
                )
            options.object_options.pixel_size_m = pixel_size_m
            options.object_options.optimizable = True
            options.object_options.optimizer = api.Optimizers.SGD

            if engine == "lsqml":
                options.object_options.step_size = 1
                options.object_options.build_preconditioner_with_all_modes = False
            elif engine == "epie":
                options.object_options.step_size = 0.1
                options.object_options.alpha = 1
            elif engine == "rpie":
                options.object_options.step_size = 1
                options.object_options.alpha = 0.1
                options.object_options.build_preconditioner_with_all_modes = False
            elif projection_engine:
                # Dampen the object projection to suppress growing phase halos
                # and blind object-probe oscillations.
                options.object_options.inertia = 0.5
                options.object_options.amplitude_clamp_limit = 1000.0
                # Keep the object transmission near one and transfer the
                # compensating global scale to the probe every 10 epochs.
                ambiguity_removal = options.object_options.remove_object_probe_ambiguity
                ambiguity_removal.enabled = True
                ambiguity_removal.optimization_plan.stride = 10

            tv = options.object_options.total_variation
            tv.enabled = total_variation
            tv.weight = total_variation_weight
            tv.optimization_plan.start = total_variation_start
            tv.optimization_plan.stride = total_variation_stride
            if total_variation:
                print(
                    "Enabled total variation regularization "
                    f"(weight={total_variation_weight:g}, "
                    f"start={total_variation_start}, stride={total_variation_stride})"
                )

            # --- Probe ---
            options.probe_options.optimizer = api.Optimizers.SGD
            options.probe_options.optimizable = True

            if engine == "lsqml":
                options.probe_options.step_size = 1
            elif engine == "epie":
                options.probe_options.step_size = 0.1
                options.probe_options.alpha = 1
            elif engine == "rpie":
                options.probe_options.step_size = 1
                options.probe_options.alpha = 0.1
            elif projection_engine:
                # Preserve the supplied, exposure-rescaled probe while the
                # initially uniform object begins to form. Subsequent probe
                # updates are damped to reduce blind object-probe oscillations.
                options.probe_options.inertia = 0.5
                options.probe_options.optimization_plan.start = 3
                options.probe_options.power_constraint.enabled = False

            # --- Probe positions ---
            options.probe_position_options.optimizable = True
            if projection_engine:
                # Refine the slightly jittered positions only while the
                # supplied probe is fixed. Position refinement stops when
                # probe refinement begins, keeping the two blind
                # updates from competing with one another.
                options.probe_position_options.optimizable = True
                options.probe_position_options.optimization_plan.stop = 3
                # Both projection engines support position refinement only
                # through the gradient correction method.
                options.probe_position_options.correction_options.correction_type = (
                    api.PositionCorrectionTypes.GRADIENT
                )

            # --- Reconstructor ---
            options.reconstructor_options.num_epochs = 300
            options.reconstructor_options.allow_nondeterministic_algorithms = False

            if engine == "lsqml":
                options.reconstructor_options.batch_size = 1024
                options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
                options.reconstructor_options.batching_mode = api.BatchingModes.COMPACT
                options.reconstructor_options.momentum_acceleration_gain = 0.5
            elif projection_engine:
                # DM and RAAR use all scan points rather than mini-batches.
                # chunk_length controls temporary working memory, not the
                # persistent exit-wave array, which contains every scan point.
                options.reconstructor_options.chunk_length = 1024
                if engine == "dm":
                    # Dampen the global Difference Map projection cycle. A full
                    # relaxation produced growing phase halos and an oscillatory
                    # consistency loss on the experimental fly-scan data.
                    options.reconstructor_options.exit_wave_update_relaxation = 0.5
                else:
                    # Use a more conservative reflection weight than the
                    # upstream default (0.75) for the experimental fly scan.
                    options.reconstructor_options.beta = 0.55
                # Projection engines report their exit-wave update error and
                # ignore the selectable scattering-amplitude loss.
                options.reconstructor_options.displayed_loss_function = None
                bytes_per_complex = torch.empty(
                    (), dtype=get_default_complex_dtype()
                ).element_size()
                exit_wave_bytes = (
                    positions_px.shape[0]
                    * probe.shape[1]
                    * data.shape[-2]
                    * data.shape[-1]
                    * bytes_per_complex
                )
                print(
                    f"Estimated {projection_engine_name} persistent exit-wave storage: "
                    f"{exit_wave_bytes / 1024**3:.2f} GiB "
                    "(excluding diffraction data, model arrays, and temporary buffers)"
                )
                print(f"{projection_engine_name} chunk length: 1024")
            else:
                options.reconstructor_options.batch_size = 1024

            task = PtychographyTask(
                options,
                diffraction_data=data,
                object_data=object_data,
                probe_data=probe,
                probe_position_x_px=positions_px[:, 1],
                probe_position_y_px=positions_px[:, 0],
            )
            data_np = data.detach().cpu().numpy()
            save_dp(output_dir, object_name, data_np)
            save_interval = 10
            for epoch in range(0, options.reconstructor_options.num_epochs, save_interval):
                n_epochs = min(save_interval, options.reconstructor_options.num_epochs - epoch)
                task.run(n_epochs=n_epochs)
                recon = task.get_data_to_cpu("object", as_numpy=True)
                recon_probe = task.get_data_to_cpu("probe", as_numpy=True)
                corrected_pos = task.get_data_to_cpu("probe_positions", as_numpy=True)
                loss_df = task.reconstructor.loss_tracker.table
                save_para(
                    output_dir, object_name, pixel_size_m,
                    data_np, recon, recon_probe, corrected_pos, epoch + n_epochs,
                )

            loss_df.to_csv(os.path.join(output_dir, f"{object_name}_loss.csv"))
            if engine == "dm":
                print("Saved reconstruction loss (Difference Map exit-wave consistency error)")
            elif engine == "raar":
                print("Saved reconstruction loss (RAAR exit-wave update error)")
            else:
                print("Saved reconstruction loss (MSE of scattering amplitude)")
            return data_np, recon, recon_probe, corrected_pos, loss_df

    runner = _ReconstructionRunner()
    return runner.ptycho(
        data,
        probe_path,
        positions_path,
        pixel_size_m,
        output_dir,
        object_name,
        num_probe_modes,
        engine,
        continue_recon,
        flip_x_positions,
        frame_stride,
        frame_sum,
        remove_indices,
        reconstruction_positions,
        opr,
        num_opr_modes,
        total_variation,
        total_variation_weight,
        total_variation_start,
        total_variation_stride,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pty-Chi ptychographic reconstruction")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        required=True,
        help="Path to input data directory, or a saved _dp.hdf5 diffraction pattern file",
    )
    parser.add_argument(
        "--data-format",
        choices=["auto", "hdf5", "h5", "bin"],
        default="auto",
        help="Input data format. auto checks for saved _dp.hdf5 first, then raw HDF5, then binary files.",
    )
    parser.add_argument(
        "--scan-start", type=int, default=None, help="Starting scan number"
    )
    parser.add_argument(
        "--scan-end", type=int, default=None, help="Ending scan number (inclusive); must be >= scan-start"
    )
    parser.add_argument(
        "--scan-glob",
        default="scan_{scan}_*.h5",
        help="HDF5 glob template under data-dir; placeholders: {scan}, {scan_padded}",
    )
    parser.add_argument(
        "--bin-dir-template",
        default="{scan}/raw",
        help="Binary frame directory template under data-dir; placeholders: {scan}, {scan_padded}",
    )
    parser.add_argument(
        "--bin-pattern",
        default="image_*.bin",
        help="Binary frame filename glob inside each binary scan directory",
    )
    parser.add_argument(
        "--bin-frame-shape",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Shape of each binary frame before crop. If omitted, square frames are inferred when possible.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Crop bounds: xmin, xmax, ymin, ymax",
    )
    parser.add_argument(
        "--probe-path", type=str, required=True, help="Path to probe file"
    )
    parser.add_argument(
        "--positions-path",
        type=str,
        required=True,
        help="Path to positions CSV file or saved _para.hdf5 parameter file",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=6.65e-9, help="Pixel size in meters"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./recon_outputs",
        help="Output directory for dp and para files",
    )
    parser.add_argument(
        "--num-probe-modes", type=int, default=8, help="Number of probe modes"
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default=None,
        help="Output file prefix. Defaults to the _dp.hdf5 stem before _dp when available, otherwise 'scan'.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="lsqml",
        help="Reconstruction engine",
        choices=["lsqml", "epie", "rpie", "dm", "raar"],
    )
    parser.add_argument(
        "--opr",
        type=parse_boolean,
        nargs="?",
        const=True,
        default=False,
        metavar="{true,false}",
        help="Enable OPR modes for non-projection engines (default: false)",
    )
    parser.add_argument(
        "--num-opr-modes",
        type=int,
        default=3,
        help="Number of additional OPR probe modes to add when --opr is true (default: 3)",
    )
    parser.add_argument(
        "--total-variation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable object total variation regularization (default: disabled)",
    )
    parser.add_argument(
        "--total-variation-weight",
        type=float,
        default=1e-2,
        help="Total variation regularization weight (default: 1e-2)",
    )
    parser.add_argument(
        "--total-variation-start",
        type=int,
        default=20,
        help="First epoch at which to apply total variation (default: 20)",
    )
    parser.add_argument(
        "--total-variation-stride",
        type=int,
        default=5,
        help="Apply total variation every N epochs after it starts (default: 5)",
    )
    parser.add_argument(
        "--continue-recon",
        action="store_true",
        default=False,
        help="Continue reconstruction from a previous result loaded from the probe path hdf5 file",
    )
    parser.add_argument(
        "--filter-indices",
        type=str,
        default=None,
        help="Path to CSV file containing indices of data points to remove",
    )
    parser.add_argument(
        "--flip-x-positions",
        action="store_true",
        default=False,
        help="Flip probe x positions by applying positions[:, 1] = -positions[:, 1]",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Load every Nth frame (e.g. 20 loads frames 0, 20, 40, ...)",
    )
    parser.add_argument(
        "--frame-sum",
        type=int,
        default=1,
        help="Sum every N consecutive loaded frames into one frame (e.g. 10 sums 0-9, 10-19, ...)",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Apply weakly scattering sample flatfield preprocessing before reconstruction",
    )
    parser.add_argument(
        "--xray-energy-kev",
        type=float,
        default=None,
        help="X-ray photon energy in keV; required with --preprocess",
    )
    parser.add_argument(
        "--zp-diameter",
        type=float,
        default=None,
        help="Zone plate diameter in meters; required with --preprocess",
    )
    parser.add_argument(
        "--zp-outer-zone-width",
        type=float,
        default=None,
        help="Zone plate outermost-zone width in meters; required with --preprocess",
    )
    parser.add_argument(
        "--sample-detector-distance",
        type=float,
        default=None,
        help="Sample-to-detector distance in meters; required with --preprocess",
    )
    parser.add_argument(
        "--defocus-distance",
        type=float,
        default=None,
        help=(
            "Signed sample defocus in meters, positive downstream of focus; "
            "required with --preprocess"
        ),
    )
    parser.add_argument(
        "--detector-pixel-size",
        type=float,
        default=None,
        help="Raw detector pixel pitch in meters; required with --preprocess",
    )
    parser.add_argument(
        "--bin",
        type=int,
        default=None,
        help="Spatial binning factor; sums non-overlapping bin x bin blocks after loading",
    )
    parser.add_argument(
        "--max-position-jitter-m",
        type=float,
        default=None,
        help="Maximum absolute uniform position jitter in meters; required with --preprocess and invalid otherwise",
    )
    args = parser.parse_args()
    if args.opr and args.engine in {"dm", "raar"}:
        parser.error("--opr true is only supported with lsqml, epie, and rpie")
    if args.opr and args.num_opr_modes < 1:
        parser.error("--num-opr-modes must be >= 1 when --opr is true")
    if args.total_variation and (
        not np.isfinite(args.total_variation_weight)
        or args.total_variation_weight <= 0
    ):
        parser.error("--total-variation-weight must be finite and > 0")
    if args.total_variation_start < 0:
        parser.error("--total-variation-start must be >= 0")
    if args.total_variation_stride < 1:
        parser.error("--total-variation-stride must be >= 1")
    if args.bin is not None and args.bin < 1:
        parser.error("--bin must be >= 1")
    if args.preprocess:
        required_preprocess_args = {
            "--xray-energy-kev": args.xray_energy_kev,
            "--zp-diameter": args.zp_diameter,
            "--zp-outer-zone-width": args.zp_outer_zone_width,
            "--sample-detector-distance": args.sample_detector_distance,
            "--defocus-distance": args.defocus_distance,
            "--detector-pixel-size": args.detector_pixel_size,
            "--bin": args.bin,
            "--max-position-jitter-m": args.max_position_jitter_m,
        }
        missing_preprocess_args = [
            name for name, value in required_preprocess_args.items() if value is None
        ]
        if missing_preprocess_args:
            parser.error("--preprocess requires " + " and ".join(missing_preprocess_args))

        positive_preprocess_args = {
            "--xray-energy-kev": args.xray_energy_kev,
            "--zp-diameter": args.zp_diameter,
            "--zp-outer-zone-width": args.zp_outer_zone_width,
            "--sample-detector-distance": args.sample_detector_distance,
            "--detector-pixel-size": args.detector_pixel_size,
            "--max-position-jitter-m": args.max_position_jitter_m,
        }
        for name, value in positive_preprocess_args.items():
            if not np.isfinite(value) or value <= 0:
                parser.error(f"{name} must be finite and > 0")
        if not np.isfinite(args.defocus_distance):
            parser.error("--defocus-distance must be finite")
    elif args.max_position_jitter_m is not None:
        parser.error("--max-position-jitter-m is only valid with --preprocess")

    # Load diffraction data BEFORE importing ptychi
    print("Loading diffraction data...")
    if args.scan_start is None and args.scan_end is None:
        scan_start = 0
        scan_end = 0
    elif args.scan_start is not None and args.scan_end is not None:
        scan_start = args.scan_start
        scan_end = args.scan_end
    else:
        raise ValueError("--scan-start and --scan-end must be provided together")

    if args.crop is None:
        xmin = xmax = ymin = ymax = None
    else:
        xmin, xmax, ymin, ymax = args.crop

    raw_data, inferred_object_name = load_scans(
        args.data_dir,
        scan_start,
        scan_end,
        xmin,
        xmax,
        ymin,
        ymax,
        args.data_format,
        args.scan_glob,
        args.bin_dir_template,
        args.bin_pattern,
        args.bin_frame_shape,
        args.frame_stride,
        args.frame_sum,
    )
    object_name = args.object_name or inferred_object_name or "scan"
    print(f"Loaded data shape: {raw_data.shape}")
    if args.bin is not None:
        raw_data = bin_diffraction_patterns(raw_data, args.bin)
        print(f"Binned data shape: {raw_data.shape}")

    remove_indices = None
    if args.filter_indices is not None:
        remove_indices = np.genfromtxt(args.filter_indices, delimiter=",", dtype=int).flatten()
        raw_data = np.delete(raw_data, remove_indices, axis=0)
        print(f"Filtered {len(remove_indices)} data points; new shape: {raw_data.shape}")

    reconstruction_positions = None
    if args.preprocess:
        positions = load_prepared_positions(
            args.positions_path,
            remove_indices,
        )
        stage_m_per_detector_px = calculate_stage_m_per_detector_px(
            args.xray_energy_kev,
            args.zp_diameter,
            args.zp_outer_zone_width,
            args.sample_detector_distance,
            args.defocus_distance,
            args.detector_pixel_size,
            args.bin,
        )
        print(
            "Applying weakly scattering sample preprocessing with "
            f"{stage_m_per_detector_px:g} stage m per binned detector pixel..."
        )
        raw_data = preprocess_weakly_scattering_data(
            raw_data,
            positions,
            stage_m_per_detector_px,
        )
        print(f"Preprocessed data shape: {raw_data.shape}")
        reconstruction_positions = add_position_jitter(
            positions,
            args.max_position_jitter_m,
        )
        print(
            "Added uniform position jitter after preprocessing "
            f"(maximum absolute jitter: {args.max_position_jitter_m:g} m)"
        )

    # Now run reconstruction (ptychi imports happen inside)
    # save_dp and save_para are called inside run_ptychi_reconstruction
    print("Running ptychi reconstruction...")
    data, recon_object, recon_probe, corrected_pos, loss_df = run_ptychi_reconstruction(
        raw_data,
        args.probe_path,
        args.positions_path,
        args.pixel_size,
        args.output_dir,
        object_name,
        args.num_probe_modes,
        args.engine,
        args.continue_recon,
        args.flip_x_positions,
        args.frame_stride,
        args.frame_sum,
        remove_indices,
        reconstruction_positions,
        args.opr,
        args.num_opr_modes,
        args.total_variation,
        args.total_variation_weight,
        args.total_variation_start,
        args.total_variation_stride,
    )
    print(
        f"Data shape: {data.shape} | Object shape: {recon_object.shape} | Probe shape: {recon_probe.shape}"
    )
