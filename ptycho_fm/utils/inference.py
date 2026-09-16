"""Shared helpers for object-patch inference workflows."""

import pickle
from collections.abc import Iterable, Iterator
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from ptycho_fm.data import PtychographyDataset
from ptycho_fm.model.model import (
    PtychoFMCoupledAmpPh,
    PtychoFMInference,
    PtychoFMReIm,
)


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_inference_model(config: dict) -> tuple[torch.nn.Module, int]:
    """Construct the object-patch model selected by the training configuration."""
    model_config = config["model"]
    image_size = model_config["encoder"]["img_size"]
    coupled_decoder_mode = model_config.get("coupled_decoder_mode")

    if coupled_decoder_mode is None:
        model = PtychoFMInference(config=model_config)
    elif coupled_decoder_mode == "real_imag":
        model = PtychoFMReIm(config=model_config)
    elif coupled_decoder_mode == "amp_phase":
        model = PtychoFMCoupledAmpPh(config=model_config)
    else:
        raise ValueError(
            "Unknown model.coupled_decoder_mode: "
            f"{coupled_decoder_mode!r}. Expected null, 'real_imag', or 'amp_phase'."
        )
    return model, image_size


def load_checkpoint(
    model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device
) -> None:
    """Load a model state dictionary using the requested device for deserialization."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)


def make_inference_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """Create the single-process loader required by the HDF5-backed datasets."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def build_ptychography_dataloader(
    data_path: str | Path,
    config: dict,
    normalization_value: float,
    batch_size: int,
    *,
    apply_noise: bool | None = None,
) -> tuple[PtychographyDataset, DataLoader]:
    """Build the paired-Ptychodus dataset and its inference loader."""
    data_config = config.get("data", {})
    if apply_noise is None:
        apply_noise = data_config.get("test_apply_noise", False)
    dataset = PtychographyDataset(
        str(data_path),
        scale=data_config.get("scale", 10000.0),
        normalization_dict_path=None,
        default_normalization=normalization_value,
        apply_noise=apply_noise,
        cache_object=data_config.get("cache_object", True),
        max_probe_modes=data_config.get("max_probe_modes", 8),
        max_OPR_modes=data_config.get("max_OPR_modes", 1),
        cache_memory_budget_mb=data_config.get("cache_memory_budget_mb", 512),
    )
    dataset.normalization = normalization_value
    return dataset, make_inference_dataloader(dataset, batch_size)


def predict_object_patches(
    model: torch.nn.Module,
    batch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt current model output signatures to amplitude and phase patches."""
    diffraction, _amplitude, _phase, probe, _position, normalization, scale = batch
    input_diffraction = diffraction.to(device)

    if isinstance(model, PtychoFMInference):
        output_amplitude, output_phase = model(input_diffraction)
    elif isinstance(model, (PtychoFMReIm, PtychoFMCoupledAmpPh)):
        input_probe = torch.view_as_real(probe).to(device)
        _output_diffraction, output_amplitude, output_phase = model(
            input_diffraction,
            input_probe,
            normalization.to(device),
            scale.to(device),
        )
    else:
        raise TypeError(f"Unsupported inference model: {type(model).__name__}")

    return output_amplitude.squeeze(1).cpu(), output_phase.squeeze(1).cpu()


def iter_object_patch_predictions(
    model: torch.nn.Module,
    batches: Iterable,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield CPU amplitude and phase predictions one batch at a time."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in batches:
            yield predict_object_patches(model, batch, device)


def compute_dataset_max(dataset, chunk_size: int = 256) -> float:
    """Compute a positive maximum from an array-like HDF5 dataset in chunks."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    total = dataset.shape[0]
    maximum = -np.inf
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        maximum = max(maximum, float(np.max(dataset[start:end])))
    return validate_normalization(maximum, "maximum diffraction intensity")


def validate_normalization(value: float, source: str) -> float:
    """Return a finite, positive normalization value."""
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{source} must be finite and > 0, got {value}")
    return value


def load_normalization_map(path: str | Path) -> dict:
    """Load an object-name-to-normalization mapping."""
    with open(path, "rb") as file:
        normalization_map = pickle.load(file)
    if not isinstance(normalization_map, dict):
        raise TypeError("normalization file must contain a dict")
    return normalization_map


def object_name_from_data_path(data_path: str | Path) -> str:
    """Extract an object name from a paired ``*_dp.hdf5`` path."""
    path = Path(data_path)
    suffix = "_dp"
    if path.suffix.lower() != ".hdf5" or not path.stem.endswith(suffix):
        raise ValueError(f"Expected an *_dp.hdf5 file, got {path}")
    return path.stem[: -len(suffix)]


def resolve_normalization(
    data_path: str | Path,
    *,
    normalization_map: dict | None = None,
    normalization_value: float | None = None,
    object_name: str | None = None,
    dataset_path: str = "dp",
) -> float:
    """Resolve normalization from a scalar, mapping, or HDF5 dataset maximum."""
    if normalization_map is not None and normalization_value is not None:
        raise ValueError(
            "normalization_map and normalization_value are mutually exclusive"
        )
    if normalization_value is not None:
        return validate_normalization(normalization_value, "normalization value")
    if normalization_map is not None:
        if object_name is None:
            object_name = object_name_from_data_path(data_path)
        if object_name not in normalization_map:
            raise KeyError(
                f"Normalization file has no value for object {object_name!r}"
            )
        return validate_normalization(
            normalization_map[object_name],
            f"normalization value for {object_name!r}",
        )

    with h5py.File(data_path, "r") as file:
        if dataset_path not in file:
            raise KeyError(f"Missing {dataset_path!r} dataset in {data_path}")
        return compute_dataset_max(file[dataset_path])


def crop_patch_borders(patches: torch.Tensor, crop: int) -> torch.Tensor:
    """Remove an equal number of pixels from every patch edge."""
    if crop < 0:
        raise ValueError(f"patch crop must be >= 0, got {crop}")
    if crop == 0:
        return patches
    if 2 * crop >= patches.shape[-2] or 2 * crop >= patches.shape[-1]:
        raise ValueError(
            f"patch crop {crop} removes all pixels from patch shape "
            f"{tuple(patches.shape[-2:])}"
        )
    return patches[:, crop:-crop, crop:-crop]


def crop_object(image: torch.Tensor | np.ndarray, crop: int):
    """Remove an equal number of pixels from every stitched-object edge."""
    if crop < 0:
        raise ValueError(f"object crop must be >= 0, got {crop}")
    if crop == 0:
        return image
    if 2 * crop >= image.shape[-2] or 2 * crop >= image.shape[-1]:
        raise ValueError(
            f"object crop {crop} removes all pixels from object shape "
            f"{tuple(image.shape[-2:])}"
        )
    return image[crop:-crop, crop:-crop]


def _as_float32_numpy(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    return np.asarray(image).astype(np.float32, copy=False)


def save_stitched_outputs(
    output_dir: str | Path,
    object_name: str,
    pred_amp_object: torch.Tensor | np.ndarray,
    pred_ph_object: torch.Tensor | np.ndarray,
    object_crop: int = 0,
) -> tuple[Path, Path]:
    """Save stitched objects using the canonical lazy-inference filenames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    amp_path = output_dir / f"pred_amp_object_{object_name}.npy"
    ph_path = output_dir / f"pred_ph_object_{object_name}.npy"
    amplitude = _as_float32_numpy(crop_object(pred_amp_object, object_crop))
    phase = _as_float32_numpy(crop_object(pred_ph_object, object_crop))
    np.save(amp_path, amplitude)
    np.save(ph_path, phase)
    return amp_path, ph_path
