import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import fft2 as scipy_fft2, next_fast_len
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.special import erf, erfinv


def _validate_spectral_image(image: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"{name} must be 2D; received shape {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{name} must contain only finite values")
    complex_dtype = np.result_type(image.dtype, np.complex64)
    return image.astype(complex_dtype, copy=False)


def _spectral_pixel_size(
    pixel_size: float | tuple[float, float],
) -> tuple[float, float]:
    if np.isscalar(pixel_size):
        dy = dx = float(pixel_size)
    else:
        if len(pixel_size) != 2:
            raise ValueError("pixel_size must be a scalar or (row, column) pair")
        dy, dx = map(float, pixel_size)
    if dy <= 0 or dx <= 0:
        raise ValueError("pixel sizes must be positive")
    return dy, dx


def _spectral_apodization(
    shape: tuple[int, int],
    window: str | np.ndarray | None,
) -> np.ndarray:
    if window is None:
        return np.ones(shape, dtype=float)
    if isinstance(window, str):
        if window == "none":
            return np.ones(shape, dtype=float)
        if window != "hann":
            raise ValueError("window must be None, 'none', 'hann', or a 2D array")
        row_window = np.hanning(shape[0])
        column_window = np.hanning(shape[1])
        return np.multiply.outer(row_window, column_window)

    window_array = np.asarray(window, dtype=float)
    if window_array.shape != shape:
        raise ValueError(
            f"window shape {window_array.shape} does not match image shape {shape}"
        )
    if not np.all(np.isfinite(window_array)):
        raise ValueError("window must contain only finite values")
    if not np.any(window_array):
        raise ValueError("window must contain at least one nonzero value")
    return window_array


def _spectral_fft_shape(
    image_shape: tuple[int, int],
    pad_to_fast_shape: bool,
) -> tuple[int, int]:
    if not pad_to_fast_shape:
        return image_shape
    return tuple(next_fast_len(size) for size in image_shape)



def _radial_frequency_geometry(
    shape: tuple[int, int],
    pixel_size: float | tuple[float, float],
    n_bins: int | None,
) -> dict[str, np.ndarray | float | int | tuple[float, float]]:
    dy, dx = _spectral_pixel_size(pixel_size)
    if n_bins is None:
        n_bins = min(shape) // 2
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    frequency_y = np.fft.fftfreq(shape[0], d=dy)
    frequency_x = np.fft.fftfreq(shape[1], d=dx)
    radial_frequency = np.hypot(frequency_y[:, None], frequency_x[None, :])
    nyquist_frequency = min(0.5 / dy, 0.5 / dx)
    bin_edges = np.linspace(0.0, nyquist_frequency, n_bins + 1)
    # Include samples exactly on the inscribed Nyquist circle in the final bin.
    bin_edges[-1] = np.nextafter(bin_edges[-1], np.inf)

    ring_index = np.full(radial_frequency.size, -1, dtype=np.int32)
    inside = radial_frequency.ravel() <= nyquist_frequency
    ring_index[inside] = np.searchsorted(
        bin_edges, radial_frequency.ravel()[inside], side="right"
    ) - 1
    counts = np.bincount(
        ring_index[inside], minlength=n_bins
    ).astype(int, copy=False)

    return {
        "frequency": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
        "bin_edges": bin_edges,
        "ring_index": ring_index,
        "counts": counts,
        "frequency_y": frequency_y,
        "frequency_x": frequency_x,
        "nyquist_frequency": nyquist_frequency,
        "pixel_size": (dy, dx),
        "n_bins": n_bins,
    }


def compute_fourier_ring_correlation(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    pixel_size: float | tuple[float, float],
    n_bins: int | None = None,
    subtract_mean: bool = True,
    window: str | np.ndarray | None = "hann",
    align_global_phase: bool = True,
    pad_to_fast_shape: bool = True,
    fft_workers: int | None = -1,
    eps: float = 1e-15,
) -> dict[str, Any]:
    """Compute ring-wise Fourier correlation for two complex 2D images.

    This implementation supports complex transmission functions directly. A shared
    apodization suppresses boundary leakage, and the prediction can be corrected for
    one arbitrary global complex phase before correlation. The returned curve is a
    spectral-agreement measure. A threshold crossing is an unbiased resolution
    estimate only when ``pred`` and ``target`` come from independent data subsets.

    Args:
        pred: Predicted complex image.
        target: Reference complex image with the same shape.
        pixel_size: Scalar pixel size or ``(row, column)`` pair in any length unit.
        n_bins: Number of rings up to the inscribed Nyquist frequency.
        subtract_mean: Remove each image's complex DC component before the FFT.
        window: Shared ``"hann"`` window, no window, or a custom 2D window.
        align_global_phase: Correct the prediction's arbitrary global complex phase.
        pad_to_fast_shape: Zero-pad each axis to a fast FFT length. This changes
            frequency sampling but adds no independent spatial information.
        fft_workers: Worker count passed to SciPy's FFT implementation.
        eps: Minimum spectral energy used to define a valid ring.

    Returns:
        Ring frequencies, FRC values, sample counts, Nyquist frequency, and the
        applied global phase correction. Frequencies use the inverse unit of
        ``pixel_size``.
    """
    pred_array = _validate_spectral_image(pred, "pred")
    target_array = _validate_spectral_image(target, "target")
    if pred_array.shape != target_array.shape:
        raise ValueError(
            f"pred and target must have the same shape, got "
            f"{pred_array.shape} and {target_array.shape}"
        )

    if subtract_mean:
        pred_array = pred_array - pred_array.mean()
        target_array = target_array - target_array.mean()

    common_dtype = np.result_type(pred_array.dtype, target_array.dtype)
    pred_array = pred_array.astype(common_dtype, copy=False)
    target_array = target_array.astype(common_dtype, copy=False)
    apodization = _spectral_apodization(pred_array.shape, window).astype(
        pred_array.real.dtype, copy=False
    )
    pred_windowed = pred_array * apodization
    target_windowed = target_array * apodization

    phase_correction = 1.0 + 0.0j
    if align_global_phase:
        overlap = np.vdot(pred_windowed.ravel(), target_windowed.ravel())
        if abs(overlap) > eps:
            phase_correction = overlap / abs(overlap)
            pred_windowed = pred_windowed * phase_correction

    fft_shape = _spectral_fft_shape(pred_array.shape, pad_to_fast_shape)
    pred_fft = scipy_fft2(pred_windowed, s=fft_shape, workers=fft_workers)
    target_fft = scipy_fft2(target_windowed, s=fft_shape, workers=fft_workers)
    geometry = _radial_frequency_geometry(fft_shape, pixel_size, n_bins)
    ring_index = geometry["ring_index"]
    valid_pixels = ring_index >= 0
    valid_ring_index = ring_index[valid_pixels]
    n_rings = int(geometry["n_bins"])

    cross_spectrum = (pred_fft * np.conj(target_fft)).ravel()[valid_pixels]
    pred_power = np.abs(pred_fft).ravel()[valid_pixels] ** 2
    target_power = np.abs(target_fft).ravel()[valid_pixels] ** 2
    numerator = np.bincount(
        valid_ring_index,
        weights=cross_spectrum.real,
        minlength=n_rings,
    )
    denominator = np.sqrt(
        np.bincount(valid_ring_index, weights=pred_power, minlength=n_rings)
        * np.bincount(valid_ring_index, weights=target_power, minlength=n_rings)
    )

    frc = np.full(n_rings, np.nan, dtype=float)
    valid_rings = denominator > eps
    frc[valid_rings] = numerator[valid_rings] / denominator[valid_rings]
    frc[valid_rings] = np.clip(frc[valid_rings], -1.0, 1.0)

    return {
        "frequency": geometry["frequency"],
        "frc": frc,
        "counts": geometry["counts"],
        "nyquist_frequency": geometry["nyquist_frequency"],
        "pixel_size": geometry["pixel_size"],
        "global_phase_correction_rad": float(np.angle(phase_correction)),
        "fft_shape": fft_shape,
        "subtract_mean": subtract_mean,
        "window": window,
    }


def compute_power_spectral_density(
    image: np.ndarray,
    *,
    pixel_size: float | tuple[float, float],
    n_bins: int | None = None,
    subtract_mean: bool = True,
    window: str | np.ndarray | None = "hann",
    return_2d: bool = False,
    pad_to_fast_shape: bool = True,
    fft_workers: int | None = -1,
) -> dict[str, Any]:
    """Compute the periodogram and radial PSD of a complex 2D image.

    The 2D normalization is chosen so integrating the unwindowed PSD over spatial
    frequency recovers the mean squared magnitude of the mean-subtracted image.
    Radial PSD is the arithmetic mean within each ring of the inscribed Nyquist
    circle. PSD reports spectral power, not whether that power is signal or noise.

    Args:
        image: Complex 2D image, such as ``amplitude * exp(1j * phase)``.
        pixel_size: Scalar pixel size or ``(row, column)`` pair in any length unit.
        n_bins: Number of radial frequency rings.
        subtract_mean: Remove the complex DC component before the FFT.
        window: ``"hann"``, no window, or a custom 2D apodization.
        return_2d: Include the shifted 2D PSD and frequency axes in the result.
        pad_to_fast_shape: Zero-pad each axis to a fast FFT length.
        fft_workers: Worker count passed to SciPy's FFT implementation.

    Returns:
        Radial frequencies, radial PSD, counts, Nyquist frequency, and optional
        shifted 2D PSD. Frequencies use the inverse unit of ``pixel_size``.
    """
    image_array = _validate_spectral_image(image, "image")
    if subtract_mean:
        image_array = image_array - image_array.mean()

    apodization = _spectral_apodization(image_array.shape, window).astype(
        image_array.real.dtype, copy=False
    )
    window_power = float(np.mean(apodization**2))
    if window_power <= 0:
        raise ValueError("window has zero mean-square power")

    windowed = image_array * apodization
    fft_shape = _spectral_fft_shape(image_array.shape, pad_to_fast_shape)
    spectrum = scipy_fft2(windowed, s=fft_shape, workers=fft_workers)
    dy, dx = _spectral_pixel_size(pixel_size)
    n_pixels = image_array.size
    psd_2d = (
        (dy * dx / n_pixels)
        * np.abs(spectrum) ** 2
        / window_power
    )

    geometry = _radial_frequency_geometry(fft_shape, pixel_size, n_bins)
    ring_index = geometry["ring_index"]
    valid_pixels = ring_index >= 0
    valid_ring_index = ring_index[valid_pixels]
    n_rings = int(geometry["n_bins"])
    radial_sum = np.bincount(
        valid_ring_index,
        weights=psd_2d.ravel()[valid_pixels],
        minlength=n_rings,
    )
    counts = geometry["counts"]
    radial_psd = np.full(n_rings, np.nan, dtype=float)
    nonempty = counts > 0
    radial_psd[nonempty] = radial_sum[nonempty] / counts[nonempty]

    result = {
        "frequency": geometry["frequency"],
        "psd": radial_psd,
        "counts": counts,
        "nyquist_frequency": geometry["nyquist_frequency"],
        "pixel_size": geometry["pixel_size"],
        "fft_shape": fft_shape,
        "subtract_mean": subtract_mean,
        "window": window,
    }
    if return_2d:
        result.update({
            "psd_2d": np.fft.fftshift(psd_2d),
            "frequency_y": np.fft.fftshift(geometry["frequency_y"]),
            "frequency_x": np.fft.fftshift(geometry["frequency_x"]),
        })
    return result


def analyze_complex_fourier_metrics(
    images: Mapping[str, np.ndarray],
    reference_name: str,
    *,
    pixel_size: float | tuple[float, float],
    n_bins: int | None = None,
    subtract_mean: bool = True,
    window: str | np.ndarray | None = "hann",
    align_global_phase: bool = True,
    pad_to_fast_shape: bool = True,
    fft_workers: int | None = -1,
    eps: float = 1e-15,
) -> dict[str, Any]:
    """Compute radial PSDs and reference-FRC curves with one FFT per image.

    This collection-level routine is equivalent to calling
    :func:`compute_power_spectral_density` for every image and
    :func:`compute_fourier_ring_correlation` against one reference, but it reuses
    the apodization, radial geometry, and reference FFT.
    """
    if not isinstance(images, Mapping) or not images:
        raise ValueError("images must be a non-empty mapping")
    if reference_name not in images:
        raise KeyError(f"reference image {reference_name!r} is missing")

    names = list(images)
    arrays = {
        name: _validate_spectral_image(image, f"images[{name!r}]")
        for name, image in images.items()
    }
    reference_shape = arrays[reference_name].shape
    mismatched = {
        name: array.shape
        for name, array in arrays.items()
        if array.shape != reference_shape
    }
    if mismatched:
        raise ValueError(
            f"all images must have shape {reference_shape}; mismatches: {mismatched}"
        )

    common_dtype = np.result_type(*(array.dtype for array in arrays.values()))
    arrays = {
        name: array.astype(common_dtype, copy=False)
        for name, array in arrays.items()
    }
    apodization = _spectral_apodization(reference_shape, window).astype(
        arrays[reference_name].real.dtype, copy=False
    )
    window_power = float(np.mean(apodization**2))
    if window_power <= 0:
        raise ValueError("window has zero mean-square power")

    def prepare_windowed(array: np.ndarray) -> np.ndarray:
        if subtract_mean:
            array = array - array.mean()
        return array * apodization

    fft_shape = _spectral_fft_shape(reference_shape, pad_to_fast_shape)
    geometry = _radial_frequency_geometry(fft_shape, pixel_size, n_bins)
    ring_index = geometry["ring_index"]
    valid_pixels = ring_index >= 0
    valid_ring_index = ring_index[valid_pixels]
    n_rings = int(geometry["n_bins"])
    counts = geometry["counts"]

    reference_windowed = prepare_windowed(arrays[reference_name])
    reference_fft = scipy_fft2(
        reference_windowed, s=fft_shape, workers=fft_workers
    )
    reference_power_flat = (
        np.abs(reference_fft).ravel()[valid_pixels] ** 2
    )
    reference_ring_power = np.bincount(
        valid_ring_index,
        weights=reference_power_flat,
        minlength=n_rings,
    )

    dy, dx = _spectral_pixel_size(pixel_size)
    psd_scale = dy * dx / (np.prod(reference_shape) * window_power)
    frc_results: dict[str, dict[str, Any]] = {}
    psd_results: dict[str, dict[str, Any]] = {}

    for name in names:
        if name == reference_name:
            windowed = reference_windowed
            spectrum = reference_fft
            phase_correction = 1.0 + 0.0j
        else:
            windowed = prepare_windowed(arrays[name])
            phase_correction = 1.0 + 0.0j
            if align_global_phase:
                overlap = np.vdot(windowed.ravel(), reference_windowed.ravel())
                if abs(overlap) > eps:
                    phase_correction = overlap / abs(overlap)
            spectrum = scipy_fft2(
                windowed, s=fft_shape, workers=fft_workers
            )

        spectrum_power_flat = np.abs(spectrum).ravel()[valid_pixels] ** 2
        radial_power_sum = np.bincount(
            valid_ring_index,
            weights=spectrum_power_flat * psd_scale,
            minlength=n_rings,
        )
        radial_psd = np.full(n_rings, np.nan, dtype=float)
        nonempty = counts > 0
        radial_psd[nonempty] = radial_power_sum[nonempty] / counts[nonempty]
        psd_results[name] = {
            "frequency": geometry["frequency"],
            "psd": radial_psd,
            "counts": counts,
            "nyquist_frequency": geometry["nyquist_frequency"],
            "pixel_size": geometry["pixel_size"],
            "fft_shape": fft_shape,
            "subtract_mean": subtract_mean,
            "window": window,
        }

        if name == reference_name:
            continue
        aligned_spectrum = spectrum * phase_correction
        cross_spectrum = (
            aligned_spectrum * np.conj(reference_fft)
        ).ravel()[valid_pixels]
        numerator = np.bincount(
            valid_ring_index,
            weights=cross_spectrum.real,
            minlength=n_rings,
        )
        denominator = np.sqrt(
            np.bincount(
                valid_ring_index,
                weights=spectrum_power_flat,
                minlength=n_rings,
            )
            * reference_ring_power
        )
        frc = np.full(n_rings, np.nan, dtype=float)
        valid_rings = denominator > eps
        frc[valid_rings] = numerator[valid_rings] / denominator[valid_rings]
        frc[valid_rings] = np.clip(frc[valid_rings], -1.0, 1.0)
        frc_results[name] = {
            "frequency": geometry["frequency"],
            "frc": frc,
            "counts": counts,
            "nyquist_frequency": geometry["nyquist_frequency"],
            "pixel_size": geometry["pixel_size"],
            "global_phase_correction_rad": float(np.angle(phase_correction)),
            "fft_shape": fft_shape,
            "subtract_mean": subtract_mean,
            "window": window,
        }

    return {
        "reference_name": reference_name,
        "names": names,
        "frc_results": frc_results,
        "psd_results": psd_results,
        "frequency": geometry["frequency"],
        "counts": counts,
        "nyquist_frequency": geometry["nyquist_frequency"],
        "fft_shape": fft_shape,
    }


def prepare_image_metric_inputs(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    align_mean: bool = False,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a prediction/reference pair for SSIM or PSNR.

    Both metrics use the finite value range of ``target`` as their signal range.
    Set ``align_mean=True`` only when a constant additive offset is a nuisance
    parameter, such as the arbitrary global offset of a phase reconstruction.
    The same aligned prediction can then be passed to both metrics.

    Args:
        pred: Predicted image as a NumPy array or torch tensor.
        target: Reference image with the same shape as ``pred``.
        align_mean: Shift ``pred`` so its finite-value mean equals that of
            ``target``. Leave disabled for quantities whose absolute level matters.
        eps: Lower bound for the returned reference data range.

    Returns:
        ``(pred_tensor, target_tensor, data_range)`` as float32 tensors.
    """
    if torch.is_tensor(target):
        target_tensor = target.to(dtype=torch.float32)
    else:
        target_tensor = torch.from_numpy(np.ascontiguousarray(target)).float()

    if torch.is_tensor(pred):
        pred_tensor = pred.to(device=target_tensor.device, dtype=torch.float32)
    else:
        pred_tensor = torch.from_numpy(np.ascontiguousarray(pred)).to(
            device=target_tensor.device, dtype=torch.float32
        )

    if pred_tensor.shape != target_tensor.shape:
        raise ValueError(
            f"Prediction and target must have the same shape, got "
            f"{tuple(pred_tensor.shape)} and {tuple(target_tensor.shape)}"
        )

    finite_pred = torch.isfinite(pred_tensor)
    finite_target = torch.isfinite(target_tensor)
    if not finite_pred.any():
        raise ValueError("Prediction contains no finite values")
    if not finite_target.any():
        raise ValueError("Target contains no finite values")

    if align_mean:
        mean_offset = (
            target_tensor[finite_target].mean() - pred_tensor[finite_pred].mean()
        )
        pred_tensor = pred_tensor + mean_offset

    finite_target_values = target_tensor[finite_target]
    data_range = (
        finite_target_values.max() - finite_target_values.min()
    ).clamp(min=eps)
    return pred_tensor, target_tensor, data_range


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float | torch.Tensor | None = None,
    eps: float = 1e-8,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between prediction and target.

    Args:
        pred: Predicted tensor of shape (B, C, H, W) or (B, H, W) or (H, W)
        target: Ground truth tensor of same shape
        window_size: Size of the Gaussian window
        data_range: The dynamic range of the data. If None, computed from target.

    Returns:
        SSIM value (0 to 1, higher is better)
    """
    # Ensure 4D tensors (B, C, H, W)
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    pred = pred.float()
    target = target.float()

    if data_range is None:
        data_range = target.amax(dim=(-2, -1), keepdim=True) - target.amin(dim=(-2, -1), keepdim=True)
    if not torch.is_tensor(data_range):
        data_range = torch.tensor(float(data_range), device=target.device)
    data_range = data_range.clamp(min=eps)

    # Constants for numerical stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create Gaussian window
    def gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = g.view(1, -1) * g.view(-1, 1)
        return kernel_2d

    _, channels, _, _ = pred.shape
    window_2d = gaussian_window(window_size, 1.5).to(pred.device)
    window = window_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)

    # Compute means
    mu_pred = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu_target = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    # Compute variances and covariance
    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu_pred_target

    # SSIM formula
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return ssim_map.mean().item()


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | torch.Tensor | None = None,
    eps: float = 1e-8,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between prediction and target.

    Args:
        pred: Predicted tensor
        target: Ground truth tensor
        data_range: The dynamic range of the data. If None, computed from target.

    Returns:
        PSNR value in dB
    """
    pred = pred.float()
    target = target.float()

    if data_range is None:
        data_range = target.max() - target.min()
    if not torch.is_tensor(data_range):
        data_range = torch.tensor(float(data_range), device=target.device)
    data_range = data_range.clamp(min=eps)

    mse = F.mse_loss(pred, target)
    if mse.item() <= eps:
        return float("inf")

    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return psnr.item()


def sample_parallel_line_profiles(
    image: np.ndarray,
    start_rc: tuple[float, float],
    end_rc: tuple[float, float],
    *,
    pixel_size_nm: float,
    n_samples: int,
    strip_width_px: float,
    n_parallel_cuts: int,
    distance_interval_nm: tuple[float, float] | None = None,
    interpolation_order: int = 1,
    interpolation_mode: str = "nearest",
) -> dict[str, Any]:
    """Sample and average parallel line cuts through a 2D image.

    Args:
        image: Two-dimensional image in the desired display/sampling orientation.
        start_rc: Full central-line start coordinate as ``(row, column)``.
        end_rc: Full central-line end coordinate as ``(row, column)``.
        pixel_size_nm: Native pixel size along either image axis.
        n_samples: Number of interpolated positions in the requested interval.
        strip_width_px: Total perpendicular width spanned by the parallel cuts.
        n_parallel_cuts: Number of evenly spaced cuts across the strip.
        distance_interval_nm: Optional distance interval along the full central line.
        interpolation_order: Spline order passed to ``map_coordinates``.
        interpolation_mode: Boundary mode passed to ``map_coordinates``.

    Returns:
        Geometry, individual cut profiles, their mean, and their SEM.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D; received shape {image.shape}")
    if pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be positive")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if n_parallel_cuts < 1:
        raise ValueError("n_parallel_cuts must be at least 1")
    if strip_width_px < 0:
        raise ValueError("strip_width_px must be nonnegative")

    y0, x0 = map(float, start_rc)
    y1, x1 = map(float, end_rc)
    line_length_px = float(np.hypot(y1 - y0, x1 - x0))
    if line_length_px == 0:
        raise ValueError("start_rc and end_rc must differ")
    total_distance_nm = line_length_px * pixel_size_nm

    if distance_interval_nm is None:
        interval_start_nm, interval_end_nm = 0.0, total_distance_nm
    else:
        interval_start_nm, interval_end_nm = map(float, distance_interval_nm)
        if not 0 <= interval_start_nm < interval_end_nm <= total_distance_nm:
            raise ValueError(
                "distance_interval_nm must lie inside the full line distance "
                f"[0, {total_distance_nm:.6g}] nm"
            )

    start_fraction = interval_start_nm / total_distance_nm
    end_fraction = interval_end_nm / total_distance_nm
    rows = np.linspace(
        y0 + start_fraction * (y1 - y0),
        y0 + end_fraction * (y1 - y0),
        n_samples,
    )
    cols = np.linspace(
        x0 + start_fraction * (x1 - x0),
        x0 + end_fraction * (x1 - x0),
        n_samples,
    )
    distance_nm = np.linspace(interval_start_nm, interval_end_nm, n_samples)

    normal_row = -(x1 - x0) / line_length_px
    normal_col = (y1 - y0) / line_length_px
    offsets_px = np.linspace(
        -strip_width_px / 2,
        strip_width_px / 2,
        n_parallel_cuts,
    )

    cut_profiles = []
    for offset_px in offsets_px:
        cut_profiles.append(
            map_coordinates(
                image,
                [
                    rows + offset_px * normal_row,
                    cols + offset_px * normal_col,
                ],
                order=interpolation_order,
                mode=interpolation_mode,
            )
        )
    cut_profiles = np.asarray(cut_profiles, dtype=np.float64)
    mean_profile = cut_profiles.mean(axis=0)
    sem_profile = (
        cut_profiles.std(axis=0, ddof=1) / np.sqrt(n_parallel_cuts)
        if n_parallel_cuts > 1
        else np.zeros(n_samples, dtype=np.float64)
    )

    return {
        "start_rc": (y0, x0),
        "end_rc": (y1, x1),
        "rows": rows,
        "cols": cols,
        "distance_nm": distance_nm,
        "interval_nm": (interval_start_nm, interval_end_nm),
        "total_distance_nm": total_distance_nm,
        "line_length_px": line_length_px,
        "normal_row": normal_row,
        "normal_col": normal_col,
        "offsets_px": offsets_px,
        "cut_profiles": cut_profiles,
        "mean_profile": mean_profile,
        "sem_profile": sem_profile,
    }


def _validate_image_mapping(
    images: Mapping[str, np.ndarray], reference_name: str
) -> tuple[list[str], tuple[int, int]]:
    if not images:
        raise ValueError("images must not be empty")
    if reference_name not in images:
        raise KeyError(f"reference_name {reference_name!r} is absent from images")
    names = list(images)
    reference_shape = np.asarray(images[reference_name]).shape
    if len(reference_shape) != 2:
        raise ValueError("reference image must be 2D")
    for name, image in images.items():
        shape = np.asarray(image).shape
        if shape != reference_shape:
            raise ValueError(
                f"{name} has shape {shape}, while reference {reference_name} "
                f"has shape {reference_shape}"
            )
    return names, reference_shape


def _edge_quality_statistics(
    observed: np.ndarray,
    fitted: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    residual = observed - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    residual_noise = float(
        1.4826 * np.median(np.abs(residual - np.median(residual)))
    )
    return residual, r_squared, residual_noise


def analyze_reference_template_broadening(
    images: Mapping[str, np.ndarray],
    reference_name: str,
    start_rc: tuple[float, float],
    end_rc: tuple[float, float],
    *,
    pixel_size_nm: float,
    n_samples: int = 600,
    strip_width_px: float = 10,
    n_parallel_cuts: int = 11,
    reference_edge_nm: float | None = None,
    fit_half_width_nm: float = 150,
    shift_tolerance_nm: float = 50,
    extra_sigma_min_px: float = 0.25,
    extra_sigma_max_nm: float = 100,
    contrast_scale_bounds: tuple[float, float] = (0, 5),
    background_slope_limit: float | None = None,
    minimum_r_squared: float = 0.8,
    minimum_template_cnr: float = 3.0,
    maximum_relative_sigma_uncertainty: float = 0.5,
) -> dict[str, Any]:
    """Estimate Gaussian broadening of many images relative to one reference.

    Each target profile is modeled as a shifted, contrast-scaled, Gaussian-broadened
    version of the measured reference profile plus an affine background. The function
    performs no plotting and returns all sampled profiles, fit curves, diagnostics, and
    line geometry needed by callers.
    """
    names, _ = _validate_image_mapping(images, reference_name)
    if fit_half_width_nm <= 0:
        raise ValueError("fit_half_width_nm must be positive")
    if shift_tolerance_nm <= 0:
        raise ValueError("shift_tolerance_nm must be positive")
    if extra_sigma_min_px <= 0:
        raise ValueError("extra_sigma_min_px must be positive")
    if extra_sigma_max_nm <= extra_sigma_min_px * pixel_size_nm:
        raise ValueError("extra_sigma_max_nm must exceed the positive lower bound")
    scale_lower, scale_upper = map(float, contrast_scale_bounds)
    if not 0 <= scale_lower < scale_upper:
        raise ValueError("contrast_scale_bounds must be increasing and nonnegative")

    sampled = {
        name: sample_parallel_line_profiles(
            image,
            start_rc,
            end_rc,
            pixel_size_nm=pixel_size_nm,
            n_samples=n_samples,
            strip_width_px=strip_width_px,
            n_parallel_cuts=n_parallel_cuts,
        )
        for name, image in images.items()
    }
    geometry = sampled[reference_name]
    distance_nm = geometry["distance_nm"]
    reference_profile = sampled[reference_name]["mean_profile"]
    samples_per_native_px = (n_samples - 1) / geometry["line_length_px"]

    detected_reference = gaussian_filter1d(
        reference_profile,
        sigma=max(samples_per_native_px, 1.0),
    )
    if reference_edge_nm is None:
        gradient = np.gradient(detected_reference, distance_nm)
        reference_edge_nm = float(distance_nm[np.argmax(np.abs(gradient))])
    else:
        reference_edge_nm = float(reference_edge_nm)

    fit_min_nm = max(float(distance_nm[0]), reference_edge_nm - fit_half_width_nm)
    fit_max_nm = min(float(distance_nm[-1]), reference_edge_nm + fit_half_width_nm)
    fit_mask = (distance_nm >= fit_min_nm) & (distance_nm <= fit_max_nm)
    all_fit_indices = np.flatnonzero(fit_mask)
    if all_fit_indices.size < 8:
        raise ValueError("fitting interval contains too few samples")

    fit_stride = max(1, int(round(samples_per_native_px)))
    fit_indices = all_fit_indices[::fit_stride]
    fit_t = distance_nm[fit_indices]
    fit_reference = reference_profile[fit_indices]
    if fit_t.size < 8:
        raise ValueError("too few native-scale samples remain in fitting interval")

    dt_nm = float(np.median(np.diff(distance_nm)))
    extra_sigma_min_nm = extra_sigma_min_px * pixel_size_nm

    def broaden_reference(sigma_extra_nm: float) -> np.ndarray:
        if sigma_extra_nm <= 0:
            return reference_profile.copy()
        return gaussian_filter1d(
            reference_profile,
            sigma=float(sigma_extra_nm) / dt_nm,
            mode="nearest",
        )

    def template_model(
        t_eval: np.ndarray,
        contrast_scale: float,
        shift_nm: float,
        sigma_extra_nm: float,
        offset: float,
        slope: float,
    ) -> np.ndarray:
        broadened = broaden_reference(sigma_extra_nm)
        shifted = np.interp(t_eval - shift_nm, distance_nm, broadened)
        return (
            contrast_scale * shifted
            + offset
            + slope * (t_eval - reference_edge_nm)
        )

    fit_plot_t = distance_nm[fit_mask]

    def fit_one(name: str) -> dict[str, Any]:
        if name == reference_name:
            return {
                "success": True,
                "reliable": True,
                "name": name,
                "popt": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                "perr": np.zeros(5),
                "sigma": 0.0,
                "rise_1090": 0.0,
                "shift": 0.0,
                "target_edge": reference_edge_nm,
                "scale": 1.0,
                "offset": 0.0,
                "slope": 0.0,
                "r_squared": 1.0,
                "cnr": np.inf,
                "warnings": [],
                "fit_t": fit_plot_t,
                "fit_curve": reference_profile[fit_mask].copy(),
            }

        fit_target = sampled[name]["mean_profile"][fit_indices]
        fit_span_nm = float(fit_t[-1] - fit_t[0])
        target_range = float(np.ptp(fit_target))
        slope_limit = (
            max(2 * target_range / fit_span_nm, np.finfo(float).eps)
            if background_slope_limit is None
            else float(background_slope_limit)
        )
        if slope_limit <= 0:
            return {"success": False, "name": name, "error": "invalid slope bound"}

        design = np.column_stack(
            [fit_reference, np.ones(fit_t.size), fit_t - reference_edge_nm]
        )
        scale_init, offset_init, slope_init = np.linalg.lstsq(
            design, fit_target, rcond=None
        )[0]
        margin = 1e-6
        scale_init = np.clip(scale_init, scale_lower + margin, scale_upper - margin)
        slope_init = np.clip(slope_init, -slope_limit + margin, slope_limit - margin)
        sigma_init = np.clip(
            10.0,
            extra_sigma_min_nm + margin,
            extra_sigma_max_nm - margin,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                popt, pcov = curve_fit(
                    template_model,
                    fit_t,
                    fit_target,
                    p0=[scale_init, 0.0, sigma_init, offset_init, slope_init],
                    bounds=(
                        [
                            scale_lower,
                            -shift_tolerance_nm,
                            extra_sigma_min_nm,
                            -np.inf,
                            -slope_limit,
                        ],
                        [
                            scale_upper,
                            shift_tolerance_nm,
                            extra_sigma_max_nm,
                            np.inf,
                            slope_limit,
                        ],
                    ),
                    maxfev=50000,
                    x_scale="jac",
                    diff_step=1e-4,
                )

            contrast_scale, shift_nm, sigma_extra_nm, offset, slope = popt
            perr = np.sqrt(np.diag(pcov))
            if not np.all(np.isfinite(perr)):
                raise ValueError("parameter covariance is not finite")

            fitted = template_model(fit_t, *popt)
            residual, r_squared, residual_noise = _edge_quality_statistics(
                fit_target, fitted
            )
            broadened = broaden_reference(sigma_extra_nm)
            shifted = np.interp(fit_t - shift_nm, distance_nm, broadened)
            component = contrast_scale * shifted
            template_contrast = float(
                np.percentile(component, 95) - np.percentile(component, 5)
            )
            cnr = template_contrast / residual_noise if residual_noise > 0 else np.inf
            relative_uncertainty = perr[2] / sigma_extra_nm
            rise_1090 = 2 * np.sqrt(2) * erfinv(0.8) * sigma_extra_nm

            native_tol_nm = max(0.1 * pixel_size_nm, 1e-6)
            scale_tol = max(1e-4 * (scale_upper - scale_lower), 1e-8)
            slope_tol = max(1e-4 * slope_limit, 1e-12)
            quality_warnings: list[str] = []
            if np.isclose(abs(shift_nm), shift_tolerance_nm, atol=native_tol_nm):
                quality_warnings.append("shift pinned to its allowed range")
            if np.isclose(sigma_extra_nm, extra_sigma_min_nm, atol=native_tol_nm):
                quality_warnings.append(
                    "extra blur at lower bound; only an upper limit is supported"
                )
            if np.isclose(sigma_extra_nm, extra_sigma_max_nm, atol=native_tol_nm):
                quality_warnings.append("extra blur pinned to upper bound")
            if (
                np.isclose(contrast_scale, scale_lower, atol=scale_tol)
                or np.isclose(contrast_scale, scale_upper, atol=scale_tol)
            ):
                quality_warnings.append("contrast scale pinned to a bound")
            if np.isclose(abs(slope), slope_limit, atol=slope_tol):
                quality_warnings.append("background slope pinned to its bound")
            if not np.isfinite(r_squared) or r_squared < minimum_r_squared:
                quality_warnings.append(
                    f"R²={r_squared:.3f} below {minimum_r_squared:.3f}"
                )
            if not np.isfinite(cnr) or cnr < minimum_template_cnr:
                quality_warnings.append(
                    f"template CNR={cnr:.2f} below {minimum_template_cnr:.2f}"
                )
            if (
                not np.isfinite(relative_uncertainty)
                or relative_uncertainty > maximum_relative_sigma_uncertainty
            ):
                quality_warnings.append(
                    "relative sigma uncertainty exceeds "
                    f"{maximum_relative_sigma_uncertainty:.0%}"
                )

            return {
                "success": True,
                "reliable": len(quality_warnings) == 0,
                "name": name,
                "popt": popt,
                "perr": perr,
                "sigma": float(sigma_extra_nm),
                "rise_1090": float(rise_1090),
                "shift": float(shift_nm),
                "target_edge": float(reference_edge_nm + shift_nm),
                "scale": float(contrast_scale),
                "offset": float(offset),
                "slope": float(slope),
                "r_squared": float(r_squared),
                "cnr": float(cnr),
                "warnings": quality_warnings,
                "fit_t": fit_plot_t,
                "fit_curve": template_model(fit_plot_t, *popt),
            }
        except (RuntimeError, ValueError, OptimizeWarning) as exc:
            return {"success": False, "name": name, "error": str(exc)}

    results = {name: fit_one(name) for name in names}
    return {
        "reference_name": reference_name,
        "names": names,
        "geometry": geometry,
        "sampled": sampled,
        "mean_profiles": {
            name: sampled[name]["mean_profile"] for name in names
        },
        "profile_sems": {
            name: sampled[name]["sem_profile"] for name in names
        },
        "distance_nm": distance_nm,
        "reference_edge_nm": reference_edge_nm,
        "fit_interval_nm": (fit_min_nm, fit_max_nm),
        "fit_mask": fit_mask,
        "fit_indices": fit_indices,
        "fit_stride": fit_stride,
        "results": results,
    }


def analyze_independent_erf_edges(
    images: Mapping[str, np.ndarray],
    reference_name: str,
    start_rc: tuple[float, float],
    end_rc: tuple[float, float],
    *,
    pixel_size_nm: float,
    fit_interval_nm: tuple[float, float],
    reference_edge_nm: float,
    shift_tolerance_nm: float = 50,
    n_steps: int = 500,
    strip_width_px: float = 10,
    n_parallel_cuts: int = 11,
    sigma_min_px: float = 0.25,
    minimum_r_squared: float = 0.8,
    minimum_edge_cnr: float = 3.0,
    maximum_relative_sigma_uncertainty: float = 0.5,
) -> dict[str, Any]:
    """Fit ideal error-function edges to many images on one shared line interval.

    The reference is fitted first. All other edge centers are constrained to
    ``shift_tolerance_nm`` around the fitted reference center. The returned covariance
    uncertainty is inflated for correlation among interpolated samples.
    """
    names, _ = _validate_image_mapping(images, reference_name)
    if n_steps < 8:
        raise ValueError("n_steps must be at least 8")
    if shift_tolerance_nm <= 0:
        raise ValueError("shift_tolerance_nm must be positive")
    if sigma_min_px <= 0:
        raise ValueError("sigma_min_px must be positive")

    sampled = {
        name: sample_parallel_line_profiles(
            image,
            start_rc,
            end_rc,
            pixel_size_nm=pixel_size_nm,
            n_samples=n_steps,
            strip_width_px=strip_width_px,
            n_parallel_cuts=n_parallel_cuts,
            distance_interval_nm=fit_interval_nm,
        )
        for name, image in images.items()
    }
    distance_nm = sampled[reference_name]["distance_nm"]
    sigma_min_nm = sigma_min_px * pixel_size_nm
    sigma_max_nm = float((distance_nm[-1] - distance_nm[0]) / 2)
    effective_native_samples = max(
        (distance_nm[-1] - distance_nm[0]) / pixel_size_nm,
        1.0,
    )
    covariance_correction = float(np.sqrt(n_steps / effective_native_samples))
    samples_per_native_px = (n_steps - 1) / effective_native_samples

    def erf_model(
        t_eval: np.ndarray,
        amplitude: float,
        t0_nm: float,
        sigma_nm: float,
        offset: float,
    ) -> np.ndarray:
        return amplitude * erf(
            (t_eval - t0_nm) / (np.sqrt(2) * sigma_nm)
        ) + offset

    def fit_one(
        name: str,
        t0_bounds_nm: tuple[float, float],
        t0_initial_nm: float,
    ) -> dict[str, Any]:
        profile = sampled[name]["mean_profile"]
        t0_lower_nm = max(float(distance_nm[0]), float(t0_bounds_nm[0]))
        t0_upper_nm = min(float(distance_nm[-1]), float(t0_bounds_nm[1]))
        if t0_upper_nm <= t0_lower_nm:
            return {
                "success": False,
                "name": name,
                "error": "allowed t0 range does not overlap fitting interval",
            }

        n_level = max(3, distance_nm.size // 5)
        left_level = np.median(profile[:n_level])
        right_level = np.median(profile[-n_level:])
        amplitude_init = (right_level - left_level) / 2
        offset_init = (right_level + left_level) / 2
        t0_init = np.clip(
            t0_initial_nm,
            t0_lower_nm + 1e-6,
            t0_upper_nm - 1e-6,
        )
        sigma_init = np.clip(
            (distance_nm[-1] - distance_nm[0]) / 10,
            sigma_min_nm + 1e-6,
            sigma_max_nm - 1e-6,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                popt, pcov = curve_fit(
                    erf_model,
                    distance_nm,
                    profile,
                    p0=[amplitude_init, t0_init, sigma_init, offset_init],
                    bounds=(
                        [-np.inf, t0_lower_nm, sigma_min_nm, -np.inf],
                        [np.inf, t0_upper_nm, sigma_max_nm, np.inf],
                    ),
                    maxfev=20000,
                    x_scale="jac",
                    diff_step=1e-4,
                )

            amplitude, t0_nm, sigma_nm, offset = popt
            perr_raw = np.sqrt(np.diag(pcov))
            if not np.all(np.isfinite(perr_raw)):
                raise ValueError("parameter covariance is not finite")
            perr = perr_raw * covariance_correction

            fitted = erf_model(distance_nm, *popt)
            residual, r_squared, residual_noise = _edge_quality_statistics(
                profile, fitted
            )
            contrast = 2 * abs(amplitude)
            cnr = contrast / residual_noise if residual_noise > 0 else np.inf
            relative_uncertainty = perr[2] / sigma_nm
            rise_1090 = 2 * np.sqrt(2) * erfinv(0.8) * sigma_nm

            smooth_profile = gaussian_filter1d(
                profile,
                sigma=max(samples_per_native_px, 1.0),
            )
            gradient = np.gradient(smooth_profile, distance_nm)
            threshold = 0.1 * np.max(np.abs(gradient))
            significant = np.abs(gradient) >= threshold
            expected_direction = np.sign(amplitude)
            opposing_fraction = (
                np.mean(expected_direction * gradient[significant] < 0)
                if expected_direction != 0 and np.any(significant)
                else np.nan
            )

            bound_tol_nm = max(0.1 * pixel_size_nm, 1e-6)
            quality_warnings: list[str] = []
            if (
                np.isclose(t0_nm, t0_lower_nm, atol=bound_tol_nm)
                or np.isclose(t0_nm, t0_upper_nm, atol=bound_tol_nm)
            ):
                quality_warnings.append("t0 pinned to its allowed range")
            if (
                np.isclose(sigma_nm, sigma_min_nm, atol=bound_tol_nm)
                or np.isclose(sigma_nm, sigma_max_nm, atol=bound_tol_nm)
            ):
                quality_warnings.append("sigma pinned to a bound")
            if not np.isfinite(r_squared) or r_squared < minimum_r_squared:
                quality_warnings.append(
                    f"R²={r_squared:.3f} below {minimum_r_squared:.3f}"
                )
            if not np.isfinite(cnr) or cnr < minimum_edge_cnr:
                quality_warnings.append(
                    f"edge CNR={cnr:.2f} below {minimum_edge_cnr:.2f}"
                )
            if (
                not np.isfinite(relative_uncertainty)
                or relative_uncertainty > maximum_relative_sigma_uncertainty
            ):
                quality_warnings.append(
                    "relative sigma uncertainty exceeds "
                    f"{maximum_relative_sigma_uncertainty:.0%}"
                )
            if opposing_fraction > 0.15:
                quality_warnings.append(
                    f"{opposing_fraction:.1%} of significant slope opposes "
                    "a monotonic edge"
                )

            return {
                "success": True,
                "reliable": len(quality_warnings) == 0,
                "name": name,
                "popt": popt,
                "perr": perr,
                "perr_raw": perr_raw,
                "sigma": float(sigma_nm),
                "rise_1090": float(rise_1090),
                "t0": float(t0_nm),
                "amplitude": float(amplitude),
                "offset": float(offset),
                "r_squared": float(r_squared),
                "cnr": float(cnr),
                "opposing_slope_fraction": float(opposing_fraction),
                "warnings": quality_warnings,
                "fit_t": distance_nm,
                "fit_curve": fitted,
            }
        except (RuntimeError, ValueError, OptimizeWarning) as exc:
            return {"success": False, "name": name, "error": str(exc)}

    reference_result = fit_one(
        reference_name,
        (float(distance_nm[0]), float(distance_nm[-1])),
        reference_edge_nm,
    )
    if not reference_result["success"]:
        raise RuntimeError(
            f"reference erf fit failed: {reference_result['error']}"
        )
    reference_t0_nm = reference_result["t0"]

    results = {reference_name: reference_result}
    for name in names:
        if name == reference_name:
            continue
        results[name] = fit_one(
            name,
            (
                reference_t0_nm - shift_tolerance_nm,
                reference_t0_nm + shift_tolerance_nm,
            ),
            reference_t0_nm,
        )

    return {
        "reference_name": reference_name,
        "names": names,
        "sampled": sampled,
        "profiles": {name: sampled[name]["mean_profile"] for name in names},
        "profile_sems": {name: sampled[name]["sem_profile"] for name in names},
        "distance_nm": distance_nm,
        "fit_interval_nm": tuple(map(float, fit_interval_nm)),
        "reference_edge_initial_nm": float(reference_edge_nm),
        "reference_t0_nm": float(reference_t0_nm),
        "covariance_correction": covariance_correction,
        "results": results,
    }