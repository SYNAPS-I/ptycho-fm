import os
import subprocess
import configparser
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import glob
from scipy.ndimage import zoom as scipy_zoom


def rescale_complex_array(arr, new_shape):
    """Rescale a complex 2D array to new_shape using bilinear interpolation on mag/phase."""
    mag = np.abs(arr)
    phase = np.angle(arr)
    zoom_factors = (new_shape[-2] / arr.shape[-2], new_shape[-1] / arr.shape[-1])
    mag_r = scipy_zoom(mag, zoom_factors, order=1)
    phase_r = scipy_zoom(phase, zoom_factors, order=1)
    return (mag_r * np.exp(1j * phase_r)).astype(np.complex64)


def write_hxn_h5(path, diffamp, points_um, ptycho_data):
    """Write an HDF5 file in the HXN format expected by run-ptycho-backend."""
    angle = ptycho_data['tomography_angle_deg']
    dr_x = ptycho_data.get('dr_x_um', 0.0)
    dr_y = ptycho_data.get('dr_y_um', 0.0)
    x_range = ptycho_data.get('x_range_um', float(np.max(points_um[0]) - np.min(points_um[0])))
    y_range = ptycho_data.get('y_range_um', float(np.max(points_um[1]) - np.min(points_um[1])))

    with h5.File(path, 'w') as f:
        f.create_dataset('diffamp', data=[np.fft.fftshift(x) for x in diffamp[:]])
        f.create_dataset('points', data=points_um)
        f.create_dataset('lambda_nm', data=ptycho_data['wavelength_m'] * 1e9)
        f.create_dataset('z_m', data=ptycho_data['detector_distance_m'])
        f.create_dataset('ccd_pixel_um', data=ptycho_data['ccd_pixel_size_m'] * 1e6)
        f.create_dataset('angle', data=angle)
        f.create_dataset('bragg_theta', data=angle)
        f.create_dataset('bragg_delta', data=np.int64(0))
        f.create_dataset('bragg_gamma', data=np.int64(0))
        f.create_dataset('dr_x', data=dr_x)
        f.create_dataset('dr_y', data=dr_y)
        f.create_dataset('x_range', data=x_range)
        f.create_dataset('y_range', data=y_range)


def write_ptycho_config(path, orig_config_path, overrides):
    """
    Write a ptycho config file based on an existing one with overrides applied.
    Falls back to a minimal config if no original exists.
    """
    config = configparser.ConfigParser(inline_comment_prefixes=('#',))
    if orig_config_path and os.path.exists(orig_config_path):
        config.read(orig_config_path)
    if not config.has_section('GUI'):
        config.add_section('GUI')
    for key, value in overrides.items():
        config.set('GUI', key, str(value))
    with open(path, 'w') as f:
        config.write(f)


def _run_backend_stage(scan_id, workdir, config_path, orig_sign, ptycho_backend_path, label):
    """Run run-ptycho-backend and return (probe, obj) in h5_conv conventions."""
    print(f"  Running run-ptycho-backend for {label}...")
    result = subprocess.run(
        ["srun", "-n", "4", "--gpu-bind", "none", ptycho_backend_path, config_path],
        #[ptycho_backend_path, config_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  WARNING: run-ptycho-backend exited with code {result.returncode}")
        print(f"  stdout: {result.stdout[-500:] if result.stdout else '(empty)'}")
        print(f"  stderr: {result.stderr[-500:] if result.stderr else '(empty)'}")
        raise RuntimeError(f"run-ptycho-backend failed for scan {scan_id} ({label})")
    print(f"  run-ptycho-backend completed successfully ({label}).")

    recon_output_dir = os.path.join(workdir, "recon_result", f"S{scan_id}", orig_sign, "recon_data")
    prb_path = os.path.join(recon_output_dir, f"recon_{scan_id}_{orig_sign}_probe.npy")
    obj_path = os.path.join(recon_output_dir, f"recon_{scan_id}_{orig_sign}_object.npy")

    prb_raw = np.load(prb_path)
    if prb_raw.ndim == 2:
        probe = prb_raw[np.newaxis, np.newaxis, :, :]
    elif prb_raw.ndim == 3:
        probe = prb_raw[:, np.newaxis, :, :]
    else:
        probe = prb_raw

    obj_raw = np.load(obj_path)
    if obj_raw.ndim == 2:
        obj_raw = obj_raw[np.newaxis, :, :]
    obj = np.transpose(obj_raw, (0, 2, 1))

    return probe, obj


def run_ptycho_refinement(scan_id, dp_data, ptycho_data, obj, probe, recons_dir,
                          out_dir, orig_config_path=None, n_iterations=20,
                          ptycho_backend_path="run-ptycho-backend", gpu_id=0,
                          multimode_refinement=False):
    """
    Run run-ptycho-backend with frozen object to refine the probe after DP cropping.

    If multimode_refinement=True, runs a 5-mode reconstruction from scratch
    (no intermediate single-mode stage).

    Returns (refined_probe, refined_obj) in h5_conv's axis conventions.
    """
    workdir = os.path.join(out_dir, f"{scan_id}_crop_workdir")
    os.makedirs(workdir, exist_ok=True)

    # Write cropped HDF5 in HXN format.
    # dp_data is fftshifted (DC in center) — same format the backend expects.
    # Use raw (untransformed) positions from ptycho_data — direction/angle
    # corrections are applied later in process_data for the final output only.
    points_um = np.stack([
        ptycho_data['x_positions'] * 1e6,
        ptycho_data['y_positions'] * 1e6,
    ])
    h5_path = os.path.join(workdir, f"scan_{scan_id}.h5")
    write_hxn_h5(h5_path, dp_data, points_um, ptycho_data)

    orig_sign = "t1"
    sign = "t1_crop"
    if orig_config_path and os.path.exists(orig_config_path):
        cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
        cfg.read(orig_config_path)
        if cfg.has_option('GUI', 'sign'):
            orig_sign = cfg.get('GUI', 'sign')
            sign = orig_sign + "_crop"

    crop_N = dp_data.shape[-1]

    def _make_overrides(prb_path, obj_path, n_modes, n_iters, shm_suffix="", from_scratch=False):
        return {
            'gui': 'False',
            'init_prb_flag': 'True' if (n_modes == 1 or from_scratch) else 'False',
            'init_obj_flag': 'True',
            'prb_path': '' if (n_modes == 1 or from_scratch) else os.path.abspath(prb_path),
            #'obj_path': os.path.abspath(obj_path),
            'prb_mode_num': str(n_modes),
            'start_update_probe': '0',
            'start_update_object': '0',
            'n_iterations': str(n_iters),
            'nx': str(crop_N),
            'ny': str(crop_N),
            'nz': str(dp_data.shape[0]),
            'working_directory': os.path.abspath(workdir),
            'scan_num': str(scan_id),
            'gpu_flag': 'True',
            'gpus': '[0, 1, 2, 3]', #f'[{gpu_id}]', #
            'gpu_batch_size': '64' if n_modes == 1 else '4',
            'preview_flag': 'False',
            'save_tmp_pic_flag': 'False',
            'precision': 'single',
            'alg_flag': 'DM',
            'alg2_flag': 'mADMM',
            'alg_percentage': '0.1',
            'shm_name': f'ptycho_crop_{scan_id}{shm_suffix}',
            'x_direction': '-1.0',
            'y_direction': '-1.0',
        }

    # Save rescaled probe and object for backend.
    # Undo h5_conv axis conventions before saving:
    #   probe: (n_modes, 1, H, W) → (n_modes, H, W) for backend
    #   object: (n_slices, H, W) where axes were transposed → un-transpose back
    prb_for_backend = probe[:, 0, :, :]
    obj_for_backend = np.transpose(obj, (0, 2, 1))
    if obj_for_backend.shape[0] == 1:
        obj_for_backend = obj_for_backend[0]

    prb_path = os.path.join(workdir, f"init_probe_{scan_id}.npy")
    obj_path = os.path.join(workdir, f"init_object_{scan_id}.npy")
    np.save(prb_path, prb_for_backend)
    np.save(obj_path, obj_for_backend)

    if multimode_refinement:
        # Run 5-mode reconstruction from scratch (no intermediate single-mode stage)
        print("  Running 5-mode reconstruction from scratch...")
        config_path_5m = os.path.join(workdir, f"{scan_id}_{sign}_5mode.ptycho_config.txt")
        write_ptycho_config(config_path_5m, orig_config_path,
                            _make_overrides(prb_path, obj_path, 5, n_iterations,
                                            shm_suffix="_5m", from_scratch=True))
        refined_probe_5m, refined_obj_5m = _run_backend_stage(
            scan_id, workdir, config_path_5m, orig_sign, ptycho_backend_path,
            "5-mode probe refinement from scratch"
        )
        return refined_probe_5m, refined_obj_5m

    # Single-mode refinement
    config_path = os.path.join(workdir, f"{scan_id}_{sign}.ptycho_config.txt")
    write_ptycho_config(config_path, orig_config_path,
                        _make_overrides(prb_path, obj_path, 1, n_iterations))
    refined_probe, refined_obj = _run_backend_stage(
        scan_id, workdir, config_path, orig_sign, ptycho_backend_path,
        "single-mode probe refinement"
    )
    # Keep only first mode from single-mode result
    if refined_probe.shape[0] > 1:
        refined_probe = refined_probe[0:1]
    return refined_probe, refined_obj


def load_ptycho_data(file_path):
    """Load ptycho data from an HDF5 file."""
    with h5.File(file_path, 'r') as f:
        ccd_pixel_size_m = float(f['ccd_pixel_um'][()]) * 1e-6
        wavelength_m = float(f['lambda_nm'][()]) * 1e-9
        hc_Jm = 6.62607015e-34 * 299792458
        hc_eVm = hc_Jm / 1.602176634e-19
        probe_energy_eV = hc_eVm / wavelength_m
        detector_distance_m = float(f['z_m'][()])
        tomography_angle_deg = float(f['angle'][()])
        h5_positions = f['/points']
        x_pixel_m = float(f['x_pixel_m'][()]) if 'x_pixel_m' in f else None
        y_pixel_m = float(f['y_pixel_m'][()]) if 'y_pixel_m' in f else None
        dr_x_um = float(f['dr_x'][()]) if 'dr_x' in f else None
        dr_y_um = float(f['dr_y'][()]) if 'dr_y' in f else None
        x_range_um = float(f['x_range'][()]) if 'x_range' in f else None
        y_range_um = float(f['y_range'][()]) if 'y_range' in f else None

        x_positions = h5_positions[0] * 1e-6
        y_positions = h5_positions[1] * 1e-6

        if '/diffamp' in f:
            diff_data = f['/diffamp'][:]
            # diffamp is stored DC-at-corner (fftshifted); bring DC back to center
            # to match the output dp convention (ring in center).
            diff_data = [np.fft.fftshift(diff_data[i]) for i in range(diff_data.shape[0])]
        elif '/raw_data/filename' in f:
            fname = f['/raw_data/filename'][:][0].decode('utf-8')
            roi = f['/raw_data/roi'][()]
            # roi[0] = [row_start, row_end_excl], roi[1] = [col_start, col_end_excl]
            r0, r1 = int(roi[0, 0]), int(roi[0, 1])
            c0, c1 = int(roi[1, 0]), int(roi[1, 1])
            with h5.File(fname, 'r') as f_raw:
                # Crop to the detector ROI (exclusive end) giving the true
                # diffraction window (typically 256×256). Reading the full
                # frame caused the argmax-based crop in _resize_dp to lock
                # onto hot pixels rather than the diffraction signal.
                # roi[0] = [row_start, row_end), roi[1] = [col_start, col_end)
                raw_counts = f_raw['/entry/data/data'][:, r0:r1, c0:c1].astype(np.float32)
            # Mask hot/saturated pixels before converting to amplitude.
            hot_mask = raw_counts > 60000
            raw_counts[hot_mask] = 0.0
            # Raw detector frames already have DC in center — no fftshift needed.
            # Convert counts → amplitude so squaring in process_data recovers
            # photon-count intensity, matching the diffamp path contract.
            diff_data = list(np.sqrt(raw_counts))
        else:
            raise KeyError("Neither '/diffamp' nor '/raw_data/filename' found in the HDF5 file.")

        np.savetxt("h5_positions.txt", h5_positions)

    return {
        'ccd_pixel_size_m': ccd_pixel_size_m,
        'wavelength_m': wavelength_m,
        'probe_energy_eV': probe_energy_eV,
        'detector_distance_m': detector_distance_m,
        'tomography_angle_deg': tomography_angle_deg,
        'x_positions': x_positions,
        'y_positions': y_positions,
        'x_pixel_m': x_pixel_m,
        'y_pixel_m': y_pixel_m,
        'dr_x_um': dr_x_um,
        'dr_y_um': dr_y_um,
        'x_range_um': x_range_um,
        'y_range_um': y_range_um,
        'diff_data': diff_data,
    }


def load_recons_settings(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reconstruction settings file not found: {file_path}")

    x_direction_mult = 1.0
    y_direction_mult = 1.0
    angle_correction_flag = False
    angle = 0.0
    obj_pad = 30  # Hard coded in ptycho
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("x_direction ="):
                x_direction_mult = float(line.split('=')[1].strip())
            elif line.startswith("y_direction ="):
                y_direction_mult = float(line.split('=')[1].strip())
            elif line.startswith("angle_correction_flag ="):
                angle_correction_flag = line.split('=')[1].strip().lower() == 'true'
            elif line.startswith("angle ="):
                angle = float(line.split('=')[1].strip())

    return {
        'x_direction_mult': x_direction_mult,
        'y_direction_mult': y_direction_mult,
        'angle_correction_flag': angle_correction_flag,
        'angle': angle,
        'obj_pad': obj_pad,
    }


def adjust_obj_for_backend(obj, scale_y, scale_x, obj_pad):
    """
    Correct the rescaled object size so the ptycho backend sees the right shape.

    When the object is scaled by (scale_y, scale_x), the obj_pad region scales
    too, but the backend always allocates obj_pad fixed pixels of padding.
    The correction trims (scale > 1) or pads (scale < 1) the object by
    obj_pad * (scale - 1) pixels, split evenly on both sides of each axis.
    """
    corr_H = int(round(obj_pad * (scale_y - 1)))
    corr_W = int(round(obj_pad * (scale_x - 1)))

    if corr_H > 0:
        top = corr_H // 2
        bot = corr_H - top
        obj = obj[:, top:obj.shape[-2] - bot, :]
    elif corr_H < 0:
        pad = -corr_H
        top = pad // 2
        obj = np.pad(obj, ((0, 0), (top, pad - top), (0, 0)), mode='constant')

    if corr_W > 0:
        lft = corr_W // 2
        rgt = corr_W - lft
        obj = obj[:, :, lft:obj.shape[-1] - rgt]
    elif corr_W < 0:
        pad = -corr_W
        lft = pad // 2
        obj = np.pad(obj, ((0, 0), (0, 0), (lft, pad - lft)), mode='constant')

    return obj


def _resize_dp(dp_data, target_N):
    """Crop (around the peak) or zero-pad each DP to target_N x target_N."""
    resized = []
    for pattern in dp_data:
        if pattern.shape[-1] > target_N or pattern.shape[-2] > target_N:
            com_y, com_x = np.unravel_index(np.argmax(pattern), pattern.shape)
            start_x = max(com_x - target_N // 2, 0)
            end_x = min(com_x + target_N // 2, pattern.shape[-1])
            start_y = max(com_y - target_N // 2, 0)
            end_y = min(com_y + target_N // 2, pattern.shape[-2])
            pattern = pattern[start_y:end_y, start_x:end_x]

        if pattern.shape[-1] < target_N or pattern.shape[-2] < target_N:
            # Zero-pad. Do NOT apply a Hann window — apodizing the DP bakes a
            # fake smooth roll-off into the measurement and injects a
            # sub-photon noise floor in the padded region.
            padded = np.zeros((target_N, target_N), dtype=pattern.dtype)
            px = (target_N - pattern.shape[-1]) // 2
            py = (target_N - pattern.shape[-2]) // 2
            padded[py:py + pattern.shape[-2], px:px + pattern.shape[-1]] = pattern
            pattern = padded

        resized.append(pattern)
    return np.array(resized)


def _resize_dp_and_probeobj(dp_data, probe, obj, ptycho_data, target_N, recons_settings):
    """
    Resize diffraction patterns (crop or pad) to target_N x target_N, and
    rescale probe/object to match the new pixel grid.

    Returns (dp_data, probe, obj) with updated ptycho_data pixel sizes in-place.
    """
    wavelength_m = ptycho_data['wavelength_m']
    detector_distance_m = ptycho_data['detector_distance_m']
    ccd_pixel_size_m = ptycho_data['ccd_pixel_size_m']
    orig_N_x = ptycho_data['diff_data'][0].shape[-1]
    orig_N_y = ptycho_data['diff_data'][0].shape[-2]

    old_x_pixel_m = (wavelength_m * detector_distance_m) / (orig_N_x * ccd_pixel_size_m)
    old_y_pixel_m = (wavelength_m * detector_distance_m) / (orig_N_y * ccd_pixel_size_m)
    new_x_pixel_m = (wavelength_m * detector_distance_m) / (target_N * ccd_pixel_size_m)
    new_y_pixel_m = (wavelength_m * detector_distance_m) / (target_N * ccd_pixel_size_m)

    print(f"  Pixel size change: x {old_x_pixel_m*1e9:.2f} -> {new_x_pixel_m*1e9:.2f} nm, "
          f"y {old_y_pixel_m*1e9:.2f} -> {new_y_pixel_m*1e9:.2f} nm")

    dp_data = _resize_dp(dp_data, target_N)

    # Rescale probe to new grid
    new_probe = np.empty((probe.shape[0], probe.shape[1], target_N, target_N), dtype=np.complex64)
    for m in range(probe.shape[0]):
        for s in range(probe.shape[1]):
            new_probe[m, s] = rescale_complex_array(probe[m, s], (target_N, target_N))
    probe = new_probe

    # Rescale object: physical extent stays the same, pixel size changes
    scale_x = old_x_pixel_m / new_x_pixel_m
    scale_y = old_y_pixel_m / new_y_pixel_m
    new_obj_W = int(round(obj.shape[-1] * scale_x))
    new_obj_H = int(round(obj.shape[-2] * scale_y))
    print(f"  Rescaling object from {obj.shape[-2]}x{obj.shape[-1]} to {new_obj_H}x{new_obj_W}")
    new_obj = np.empty((obj.shape[0], new_obj_H, new_obj_W), dtype=obj.dtype)
    for s in range(obj.shape[0]):
        new_obj[s] = rescale_complex_array(obj[s], (new_obj_H, new_obj_W))
    obj = new_obj
    obj = adjust_obj_for_backend(obj, scale_y, scale_x, recons_settings['obj_pad'])
    print(f"  Object size after pad correction: {obj.shape[-2]}x{obj.shape[-1]}")

    ptycho_data['x_pixel_m'] = new_x_pixel_m
    ptycho_data['y_pixel_m'] = new_y_pixel_m

    return dp_data, probe, obj


def process_data(scan_id, src_hdf5=None, src_hdf5_fmt="scan_%s.h5",
                 recons_dir="recon_result/recon_data/", recons_prefix_fmt="recon_%s_t1_",
                 out_dir="converted_data/", ptycho_backend_path="run-ptycho-backend",
                 gpu_id=0, multimode_refinement=False):
    print(f"Processing data for scan ID {scan_id}...")

    if src_hdf5 is None:
        src_hdf5 = src_hdf5_fmt % scan_id

    # Load reconstruction data
    recons_prefix = os.path.join(recons_dir, recons_prefix_fmt % scan_id)
    probe = np.load(recons_prefix + "probe.npy")
    obj = np.load(recons_prefix + "object.npy")
    if obj.ndim == 2:
        obj = obj[np.newaxis, :, :]
    obj = np.transpose(obj, (0, 2, 1))
    if probe.ndim == 2:
        probe = probe[np.newaxis, np.newaxis, :, :]
    if probe.ndim == 3:
        print("Multiprobes detected!")
        probe = probe[:, np.newaxis, :, :]
    if probe.shape[0] > 1:
        print(f"  Truncating probe from {probe.shape[0]} modes to 1 (single probe mode).")
        probe = probe[0:1]
    probe = probe.astype(np.complex64)
    obj = obj.astype(np.complex64)

    txt_files = glob.glob(f"{recons_dir}/{scan_id}*.txt")
    if not txt_files:
        # Check for files without any extension
        txt_files = glob.glob(f"{recons_dir}/{scan_id}*")
        if not txt_files:
            raise FileNotFoundError(f"No reconstruction settings .txt file found for scan {scan_id} in {recons_dir}")
    recons_settings = load_recons_settings(txt_files[0])

    # Apply direction corrections to the object image.
    # A negative direction means the object is mirrored along that axis.
    # After the transpose above, obj[..., -2] = x, obj[..., -1] = y.
    if recons_settings['x_direction_mult'] < 0:
        obj = np.flip(obj, axis=-2).copy()
    if recons_settings['y_direction_mult'] < 0:
        obj = np.flip(obj, axis=-1).copy()

    orig_config_files = glob.glob(os.path.join(recons_dir, f"{scan_id}*.ptycho_*.txt"))
    orig_config_path = orig_config_files[0] if orig_config_files else None

    ptycho_data = load_ptycho_data(src_hdf5)
    dp_data = [p.astype(np.float32) for p in ptycho_data['diff_data']]
    x_positions = ptycho_data['x_positions']
    y_positions = ptycho_data['y_positions']
    ccd_pixel_size_m = ptycho_data['ccd_pixel_size_m']
    wavelength_m = ptycho_data['wavelength_m']
    detector_distance_m = ptycho_data['detector_distance_m']

    probe_before_refinement = None
    obj_before_refinement = None

    dp_h, dp_w = dp_data[0].shape[-2], dp_data[0].shape[-1]
    if dp_h != 256 or dp_w != 256:
        action = "Cropping" if dp_h > 256 or dp_w > 256 else "Padding"
        print(f"{action} diffraction patterns from {dp_data[0].shape} to (n_dp, 256, 256)...")

        if probe.shape[-1] == 256 and probe.shape[-2] == 256:
            # Existing reconstruction was already done at 256x256 — its probe
            # and object grid already corresponds to the 256-DP pixel size.
            # Just crop the source DPs; skip the probe/obj rescale and refine.
            print("  Reconstruction already at 256x256 — only cropping DPs, skipping refinement.")
            dp_data = _resize_dp(dp_data, 256)
            new_pixel_m = (wavelength_m * detector_distance_m) / (256 * ccd_pixel_size_m)
            ptycho_data['x_pixel_m'] = new_pixel_m
            ptycho_data['y_pixel_m'] = new_pixel_m
        else:
            n_iterations = 500
            dp_data, probe, obj = _resize_dp_and_probeobj(
                dp_data, probe, obj, ptycho_data, 256, recons_settings
            )

            print("  Refining probe and object via run-ptycho-backend...")
            probe_before_refinement = probe.copy()
            obj_before_refinement = obj.copy()
            probe, obj = run_ptycho_refinement(
                scan_id, dp_data, ptycho_data, obj, probe, recons_dir, out_dir,
                orig_config_path=orig_config_path,
                n_iterations=n_iterations,
                ptycho_backend_path=ptycho_backend_path,
                gpu_id=gpu_id,
                multimode_refinement=multimode_refinement,
            )
            print("  Refinement complete.")

    x_pixel_m = ptycho_data['x_pixel_m']
    if x_pixel_m is None:
        x_pixel_m = (wavelength_m * detector_distance_m) / (dp_data[0].shape[-1] * ccd_pixel_size_m)

    y_pixel_m = ptycho_data['y_pixel_m']
    if y_pixel_m is None:
        y_pixel_m = (wavelength_m * detector_distance_m) / (dp_data[0].shape[-2] * ccd_pixel_size_m)

    print(f"Calculated pixel size at sample: x: {x_pixel_m} m, y: {y_pixel_m} m "
          f"({ccd_pixel_size_m*1e6} um detector pixel size, {wavelength_m*1e9} nm wavelength, "
          f"{detector_distance_m} m detector distance)")

    output_dir = f"{out_dir}{scan_id}/"
    os.makedirs(output_dir, exist_ok=True)

    # Plot probe and object before/after refinement if refinement was run
    if probe_before_refinement is not None:
        n_modes = probe.shape[0]
        fig, axes = plt.subplots(n_modes, 4, figsize=(16, 4 * n_modes), squeeze=False)
        col_labels = ['Original Amplitude', 'Original Phase', 'Refined Amplitude', 'Refined Phase']
        for m in range(n_modes):
            prb_before = probe_before_refinement[0, 0]
            prb_after = probe[m, 0]
            imgs = [np.abs(prb_before), np.angle(prb_before), np.abs(prb_after), np.angle(prb_after)]
            cmaps = ['viridis', 'hsv', 'viridis', 'hsv']
            for col, (img, cmap, label) in enumerate(zip(imgs, cmaps, col_labels)):
                ax = axes[m, col]
                im = ax.imshow(img, cmap=cmap, origin='lower')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f'Mode {m} — {label}')
                ax.axis('off')
        fig.suptitle(f'Scan {scan_id} — Probe before/after refinement')
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{scan_id}_probe_refinement.png"), dpi=150)
        plt.close()

        n_slices = obj.shape[0]
        fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4 * n_slices), squeeze=False)
        for s in range(n_slices):
            obj_b = obj_before_refinement[s]
            obj_a = obj[s]
            imgs = [np.abs(obj_b), np.angle(obj_b), np.abs(obj_a), np.angle(obj_a)]
            cmaps = ['gray', 'hsv', 'gray', 'hsv']
            for col, (img, cmap, label) in enumerate(zip(imgs, cmaps, col_labels)):
                ax = axes[s, col]
                im = ax.imshow(img, cmap=cmap, origin='lower', vmin = np.percentile(img, 5), vmax=np.percentile(img, 95))
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f'Slice {s} — {label}')
                ax.axis('off')
        fig.suptitle(f'Scan {scan_id} — Object before/after refinement')
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{scan_id}_object_refinement.png"), dpi=150)
        plt.close()

    x_positions = x_positions * recons_settings['x_direction_mult']
    y_positions = y_positions * recons_settings['y_direction_mult']
    if recons_settings['angle_correction_flag']:
        if np.abs(recons_settings['angle']) <= 45.:
            x_positions *= np.abs(np.cos(recons_settings['angle'] * np.pi / 180.))
        else:
            x_positions *= np.abs(np.sin(recons_settings['angle'] * np.pi / 180.))
        if recons_settings['angle'] <= -45:
            x_positions = -x_positions

    x_pad = (probe.shape[-1] // 2 + recons_settings['obj_pad'] // 2) * x_pixel_m
    y_pad = (probe.shape[-2] // 2 + recons_settings['obj_pad'] // 2) * y_pixel_m

    x_positions = x_positions - np.min(x_positions) + x_pad
    y_positions = y_positions - np.min(y_positions) + y_pad

    tgt_pos_idx = min(100, len(x_positions) - 1)

    avg_amplitudes = np.array([np.mean(dp_data[i]) for i in range(len(dp_data))])
    fig, ax = plt.subplots(2, 1, figsize=(8, 16))
    ax[0].scatter(x_positions, y_positions, c=avg_amplitudes, cmap='viridis')
    ax[0].scatter(x_positions[tgt_pos_idx], y_positions[tgt_pos_idx],
                  color='red', marker='x', s=100, label=f'Point {tgt_pos_idx} Position')
    ax[0].legend()
    ax[0].set_xlabel('X Position (m)')
    ax[0].set_ylabel('Y Position (m)')
    ax[0].set_title(f'Scan ID {scan_id} - Probe Positions Colored by Average Diffraction Amplitude')
    ax[0].grid()

    extent = [0, obj.shape[-1] * x_pixel_m, 0, obj.shape[-2] * y_pixel_m]
    ax[1].imshow(np.angle(obj[0]), extent=extent, origin='lower', cmap='gray')
    ax[1].scatter(x_positions[tgt_pos_idx], y_positions[tgt_pos_idx], c='red', s=10, label='Probe Positions')

    min_x, max_x = np.min(x_positions), np.max(x_positions)
    min_y, max_y = np.min(y_positions), np.max(y_positions)
    rectangle = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                               linewidth=1, edgecolor='blue', facecolor='none', label='Scan Area')
    ax[1].add_patch(rectangle)
    ax[1].set_xlabel('X Position (m)')
    ax[1].set_ylabel('Y Position (m)')
    ax[1].set_title(f'Scan ID {scan_id} - Object Phase with Probe Positions')
    ax[1].legend()

    plt.savefig(os.path.join(output_dir, f"{scan_id}_probe_positions.png"))
    plt.close()

    # NOTE: the HXN source HDF5 stores `diffamp` = sqrt(photon_counts), i.e.
    # diffraction AMPLITUDE, not intensity. The downstream ptycho-vit dataset
    # convention treats the `dp` field as INTENSITY (it applies its own sqrt
    # to produce the model input). Square here so the written `dp` field is
    # truly intensity (= photon counts) and the downstream pipeline produces
    # correct amplitude units. Verified: squared values from HXN diffamp are
    # exactly integer photon counts.
    dp_data_intensity = np.asarray(dp_data, dtype=np.float32) ** 2
    output_hdf5_path = os.path.join(output_dir, f"{scan_id}_dp.hdf5")
    with h5.File(output_hdf5_path, 'w') as f_out:
        f_out.create_dataset('dp', data=dp_data_intensity)

    para_hdf5_path = os.path.join(output_dir, f"{scan_id}_para.hdf5")
    with h5.File(para_hdf5_path, 'w') as f_out:
        obj_d = f_out.create_dataset('object', data=obj)
        obj_d.attrs['center_x_m'] = 0.0
        obj_d.attrs['center_y_m'] = 0.0
        obj_d.attrs['pixel_height_m'] = y_pixel_m
        obj_d.attrs['pixel_width_m'] = x_pixel_m

        # FFT-normalization convention fix: the HXN ptycho backend reconstructs
        # the probe assuming an orthonormal FFT (factor 1/N), but downstream
        # ptycho-vit's forward model uses an *unnormalized* `torch.fft.fft2`.
        # For an N x N FFT the |·|^2 conventions differ by N^2, so to make
        # `|fft2_unnormalized(obj * probe)|^2` equal photon counts we have to
        # absorb a 1/N factor into the probe itself.
        N = probe.shape[-1]
        probe_scaled = (probe / N).astype(probe.dtype)
        f_out.create_dataset('probe', data=probe_scaled)
        f_out.create_dataset('probe_position_indexes', data=np.arange(x_positions.shape[0]))
        f_out.create_dataset('probe_position_x_m', data=x_positions)
        f_out.create_dataset('probe_position_y_m', data=y_positions)

    print(f"Converted data for scan ID {scan_id} saved to {output_hdf5_path}")


def find_recon_dir(mode, scan_id):
    """
    Scan {mode}/S{scan_id}/ for a reconstruction directory.
    Returns (recon_data_path, recon_data_fmt) or (None, None) if not found.
    Only considers reconstructions where both probe and object npy files exist.
    """
    matches = glob.glob(f"{mode}/recon_result/S{scan_id}/*/recon_data/recon_{scan_id}_*_probe.npy")
    prefix = f"recon_{scan_id}_"
    suffix = "_probe.npy"
    for match in matches:
        fname = os.path.basename(match)
        sign = fname[len(prefix):-len(suffix)]
        obj_path = match.replace("_probe.npy", "_object.npy")
        if not os.path.exists(obj_path):
            continue
        recon_dir = os.path.basename(os.path.dirname(os.path.dirname(match)))
        recon_data_path = f"{mode}/recon_result/S{scan_id}/{recon_dir}/recon_data/"
        recon_data_fmt = f"recon_%s_{sign}_"
        return recon_data_path, recon_data_fmt
    return None, None


if __name__ == "__main__":
    import multiprocessing as mp

    mode = "HDF5-file-test"
    single_scan_only = True  # Set to False to process all scans in parallel
    ptycho_backend_path = "run-ptycho-backend"
    multimode_refinement = False  # Set to True to run 5-mode refinement after single-mode

    tgt_scanids = sorted(
        int(os.path.basename(f).split('_')[1].split('.')[0])
        for f in glob.glob(f"{mode}/scan_*.h5")
    )
    print(f"Found {len(tgt_scanids)} scan IDs to process.")

    if single_scan_only:
        scan_id = 381476 # tgt_scanids[0]
        recon_path, recon_fmt = find_recon_dir(mode, scan_id)
        if recon_path is None:
            print(f"Reconstruction data not found for scan ID {scan_id}, exiting...")
            exit(1)
        process_data(scan_id, f"{mode}/scan_{scan_id}.h5", "scan_%s.h5", recon_path, recon_fmt,
                     multimode_refinement=multimode_refinement)
    else:
        n_gpus = 16
        pool = mp.Pool(processes=n_gpus)
        results = []
        job_idx = 0
        for scan_id in tgt_scanids:
            recon_data_path, recon_data_fmt = find_recon_dir(mode, scan_id)
            if recon_data_path is None:
                print(f"Reconstruction data not found for scan ID {scan_id}, skipping...")
                continue
            result = pool.apply_async(
                process_data,
                args=(scan_id, f"{mode}/scan_{scan_id}.h5", "scan_%s.h5",
                      recon_data_path, recon_data_fmt, f"converted_data_{mode}_fix/",
                      ptycho_backend_path, job_idx % n_gpus),
                kwds={'multimode_refinement': multimode_refinement},
            )
            results.append(result)
            job_idx += 1
        for result in results:
            result.get()
        pool.close()
        pool.join()
