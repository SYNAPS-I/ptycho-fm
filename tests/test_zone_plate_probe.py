import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.special import j1, j0


SCRIPT = Path(__file__).parents[1] / "scripts/iterative_reconstruction/ptychi_recon.py"
spec = importlib.util.spec_from_file_location("ptychi_recon", SCRIPT)
recon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recon)


def make_probe(**overrides):
    parameters = dict(
        zp_diameter_m=100e-6,
        central_stop_diameter_m=30e-6,
        zp_outer_zone_width_m=40e-9,
        energy_kev=8.6,
        defocus_distance_m=0.0,
        probe_shape=(97, 129),
        pixel_size_m=10e-9,
        upsample_factor=8,
    )
    parameters.update(overrides)
    return recon.generate_zone_plate_probe(**parameters)


@pytest.mark.parametrize("stop_fraction", [0.0, 0.3])
def test_focal_profile_matches_analytic_annular_airy_pattern(stop_fraction):
    probe = make_probe(central_stop_diameter_m=stop_fraction * 100e-6)
    assert probe.shape == (97, 129)
    assert probe.dtype == np.complex64
    assert np.isfinite(probe).all()
    assert np.sum(np.abs(probe) ** 2) == pytest.approx(probe.size, rel=1e-6)
    assert np.all(probe[[0, -1], :] == 0)
    assert np.all(probe[:, [0, -1]] == 0)

    # Annular-aperture Fraunhofer diffraction has a known Bessel profile.
    offsets = np.arange(1, 25)
    argument = np.pi * offsets * 10e-9 / 40e-9
    amplitude = 2 * j1(argument) / argument
    if stop_fraction:
        amplitude -= 2 * stop_fraction * j1(stop_fraction * argument) / argument
    amplitude /= 1 - stop_fraction**2
    measured = np.abs(probe[48, 64 + offsets] / probe[48, 64]) ** 2
    np.testing.assert_allclose(measured, amplitude**2, atol=2e-3)


@pytest.mark.parametrize("defocus", [-20e-6, 0.0, 20e-6])
def test_complex_defocused_field_matches_radial_fresnel_quadrature(defocus):
    probe = make_probe(defocus_distance_m=defocus)
    wavelength = 1.239841984e-9 / 8.6
    focal_length = 100e-6 * 40e-9 / wavelength
    distance = focal_length + defocus
    offsets = np.arange(20)
    rho = offsets * 10e-9
    radius = np.linspace(15e-6, 50e-6, 8193)
    # Independent continuous radial Fresnel integral, without any FFT.
    pupil_phase = np.exp(
        -1j * np.pi * defocus * radius**2 / (wavelength * distance * focal_length)
    )
    integrand = radius * pupil_phase * j0(
        2 * np.pi * rho[:, None] * radius / (wavelength * distance)
    )
    reference = simpson(integrand, x=radius, axis=1)
    reference *= np.exp(1j * np.pi * rho**2 / (wavelength * distance))
    reference /= reference[0]
    measured = probe[48, 64 + offsets] / probe[48, 64]
    np.testing.assert_allclose(measured, reference, atol=4e-3)


@pytest.mark.parametrize("shape", [(96, 128), (97, 128), (96, 129)])
def test_focal_peak_is_centered_for_even_and_odd_grids(shape):
    probe = make_probe(probe_shape=shape, upsample_factor=3)
    assert np.unravel_index(np.abs(probe).argmax(), shape) == (
        shape[0] // 2, shape[1] // 2
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"zp_diameter_m": 0}, "zp_diameter_m"),
        ({"zp_outer_zone_width_m": -1}, "zp_outer_zone_width_m"),
        ({"energy_kev": np.nan}, "energy_kev"),
        ({"pixel_size_m": 0}, "pixel_size_m"),
        ({"central_stop_diameter_m": -1}, "central_stop_diameter_m"),
        ({"central_stop_diameter_m": 100e-6}, "central_stop_diameter_m"),
        ({"defocus_distance_m": np.inf}, "defocus_distance_m"),
        ({"defocus_distance_m": -1}, "focal length"),
        ({"probe_shape": (1, 128)}, "probe_shape"),
        ({"probe_shape": (128.0, 128)}, "probe_shape"),
        ({"upsample_factor": 0}, "upsample_factor"),
        ({"pixel_size_m": 1e-6}, "does not fit"),
    ],
)
def test_invalid_optics_and_sampling_raise_value_error(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_probe(**overrides)
