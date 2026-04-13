import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from datetime import datetime
import time
from utils.ptychi_utils import place_patches_fourier_shift
import numpy as np
import wandb


def _move_to_cpu(obj):
    # Recursively detach and move Torch tensors to CPU (for safe serialization).
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_move_to_cpu(v) for v in obj]
        return tuple(moved) if isinstance(obj, tuple) else moved
    return obj


def _atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


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


class Trainer(object):
    def __init__(
        self, 
        model, 
        mode, 
        run_num, 
        device, 
        model_save_path,
        is_main_process=True, 
        use_ddp=False, 
        wandb_enabled=True, 
        debug_mode=False,
        skip_batch_if_grad_norm_greater_than=None
    ):
        super().__init__()
        self.model = model
        self.mode = mode
        self.run_num = run_num
        self.device = device
        self.model_save_path = model_save_path
        self.is_main_process = is_main_process
        self.use_ddp = use_ddp
        self.wandb_enabled = wandb_enabled
        self.debug_mode = debug_mode
        self.skip_batch_if_grad_norm_greater_than = skip_batch_if_grad_norm_greater_than

    def synchronize_loss(self, loss_value):
        """Synchronize loss across all processes in DDP."""
        if self.use_ddp and dist.is_initialized():
            loss_tensor = torch.tensor(loss_value, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= dist.get_world_size()
            return loss_tensor.item()
        return loss_value

    def update_saved_model(self, name):
        """Update saved model (checkpoints and if validation loss is minimized)."""
        os.makedirs(self.model_save_path, exist_ok=True)
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        os.makedirs(run_path, exist_ok=True)

        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        # Save a CPU copy so serialization does not depend on CUDA contexts/devices.
        cpu_state_dict = _move_to_cpu(state_dict)
        _atomic_torch_save(cpu_state_dict, os.path.join(run_path, name + '.pth'))

    def save_config(self, config_path='config.yaml'):
        """Save a copy of the config file to the run path for reproducibility."""
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        if not os.path.isdir(run_path):
            os.makedirs(run_path, exist_ok=True)
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(run_path, 'config.yaml'))

    def generate_state_dict(self, epoch_num, metrics, optimizer, wandb_run_id=None, scheduler=None):
        """Returns a dictionary of the state_dicts of all states but not the model."""
        state = {
            'current_epoch': epoch_num + 1,
            'loss_tracker': metrics,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'wandb_run_id': wandb_run_id
        }
        return state

    def save_model_and_states_checkpoint(self, epoch_num, metrics, optimizer=None, wandb_run_id=None, scheduler=None):
        """Save a checkpoint state that can be loaded to continue training."""
        if not self.is_main_process:
            return

        state_dict = self.generate_state_dict(epoch_num, metrics, optimizer, wandb_run_id, scheduler)
        state_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        os.makedirs(state_path, exist_ok=True)

        self.update_saved_model('checkpoint_model')
        # Also save epoch-specific model
        self.update_saved_model(f'model_epoch_{epoch_num + 1:03d}')

        cpu_state_dict = _move_to_cpu(state_dict)
        _atomic_torch_save(cpu_state_dict, os.path.join(state_path, 'checkpoint.state'))

    def load_state_checkpoint(self, optimizer, scheduler=None):
        """Load everything but the model."""
        checkpoint_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        checkpoint_fname = os.path.join(checkpoint_path, 'checkpoint.state')
        if not os.path.exists(checkpoint_fname):
            raise FileNotFoundError(f"Checkpoint not found in {checkpoint_fname}")
        state_dict = torch.load(checkpoint_fname, map_location='cpu')
        current_epoch = state_dict['current_epoch']
        metrics = state_dict['loss_tracker']
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        # optimizer_state_dict_device_fix
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device, non_blocking=True)

        if state_dict['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        wandb_run_id = state_dict.get('wandb_run_id', None)
        return current_epoch, metrics, optimizer, wandb_run_id, scheduler

    def generate_plot(self, in_dp, out_dp, gt_amp, pred_amp, gt_ph, pred_ph, filename):
        f, ax = plt.subplots(nrows=2, ncols=3)

        in0 = ax[0, 0].imshow(in_dp, interpolation='none', norm=colors.LogNorm(), cmap='jet')
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(in0, cax=cax, orientation='vertical')
        ax[0, 0].set_title('Input Diff. Amp.')

        in1 = ax[0, 1].imshow(gt_amp, interpolation='none')
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(in1, cax=cax, orientation='vertical', format='%.2f')
        ax[0, 1].set_title('GT Amp.')

        in2 = ax[0, 2].imshow(gt_ph, interpolation='none', cmap='magma')
        divider = make_axes_locatable(ax[0, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(in2, cax=cax, orientation='vertical', format='%.1f')
        ax[0, 2].set_title('GT Phase')

        out0 = ax[1, 0].imshow(out_dp, interpolation='none', norm=colors.LogNorm(), cmap='jet')
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(out0, cax=cax, orientation='vertical')
        ax[1, 0].set_title('Output Diff. Amp.')

        out1 = ax[1, 1].imshow(pred_amp, interpolation='none')
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(out1, cax=cax, orientation='vertical', format='%.2f')
        ax[1, 1].set_title('Predicted Amp.')

        out2 = ax[1, 2].imshow(pred_ph, interpolation='none', cmap='magma')
        divider = make_axes_locatable(ax[1, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(out2, cax=cax, orientation='vertical', format='%.1f')
        ax[1, 2].set_title('Predicted Phase')

        plt.tight_layout()
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        f.savefig(os.path.join(run_path, filename), bbox_inches='tight', transparent=True)

    def generate_test_plot(self, dataloader, epoch, filename, central_crop=64, ph_crop=180):
        total_scan_points = len(dataloader.dataset)
        pred_amp = torch.zeros((total_scan_points, dataloader.dataset.pattern_shape[0], dataloader.dataset.pattern_shape[1]), device='cpu')
        pred_ph = torch.zeros(pred_amp.shape, device='cpu')
        gt_amp = torch.zeros(pred_amp.shape, device='cpu')
        gt_ph = torch.zeros(pred_amp.shape, device='cpu')
        scan_idx = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)) and len(batch) == 8:
                    diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale, _meta = batch
                else:
                    diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch
                batch_size = diff_amp.size(0)

                input_diff = diff_amp.to(self.device, non_blocking=True)
                input_probe = torch.view_as_real(probe.clone().detach()).to(self.device, non_blocking=True)
                input_norm = norm.to(self.device, non_blocking=True)
                input_scale = scale.to(self.device, non_blocking=True)

                _output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)
                pred_amp[scan_idx:scan_idx + batch_size] = output_amp.squeeze().detach().cpu()
                pred_ph[scan_idx:scan_idx + batch_size] = output_ph.squeeze().detach().cpu()
                gt_amp[scan_idx:scan_idx + batch_size] = amp_patch.squeeze().detach().cpu()
                gt_ph[scan_idx:scan_idx + batch_size] = ph_patch.squeeze().detach().cpu()

                scan_idx += batch_size

        if dataloader.dataset._cached_probe_positions is None:
            dataloader.dataset._cache_positions()
            dataloader.dataset._cache_object_data()

        object_size = dataloader.dataset.object_shape
        positions = dataloader.dataset._cached_probe_positions
        pred_amp_object = torch.zeros(object_size, device='cpu')
        pred_ph_object = torch.zeros(object_size, device='cpu')
        buffer = torch.zeros(object_size, device='cpu')

        central_crop = int(central_crop)
        if central_crop <= 0:
            raise ValueError("test_plot central_crop must be a positive integer.")
        pred_amp_object = place_patches_fourier_shift(
            pred_amp_object,
            positions,
            pred_amp[:, central_crop:-central_crop, central_crop:-central_crop],
            op="add",
            adjoint_mode=False,
            pad=32
        )
        pred_ph_object = place_patches_fourier_shift(
            pred_ph_object,
            positions,
            pred_ph[:, central_crop:-central_crop, central_crop:-central_crop],
            op="add",
            adjoint_mode=False,
            pad=32
        )
        buffer = place_patches_fourier_shift(
            buffer,
            positions,
            torch.ones_like(pred_ph[:, central_crop:-central_crop, central_crop:-central_crop]),
            op="add",
            adjoint_mode=False,
            pad=32
        )

        gt_amp_object = torch.zeros(object_size, device='cpu')
        gt_ph_object = torch.zeros(object_size, device='cpu')
        gt_amp_object = place_patches_fourier_shift(
            gt_amp_object,
            positions,
            gt_amp[:, central_crop:-central_crop, central_crop:-central_crop],
            op="add",
            adjoint_mode=False,
            pad=32
        )
        gt_ph_object = place_patches_fourier_shift(
            gt_ph_object,
            positions,
            gt_ph[:, central_crop:-central_crop, central_crop:-central_crop],
            op="add",
            adjoint_mode=False,
            pad=32
        )

        pred_amp_object = pred_amp_object / torch.clip(buffer, min=1)
        pred_ph_object = pred_ph_object / torch.clip(buffer, min=1)
        gt_amp_object = gt_amp_object / torch.clip(buffer, min=1)
        gt_ph_object = gt_ph_object / torch.clip(buffer, min=1)

        ph_crop = int(ph_crop)
        if ph_crop <= 0:
            raise ValueError("test_plot ph_crop must be a positive integer.")
        vmin_ph = torch.mean(pred_ph_object[ph_crop:-ph_crop, ph_crop:-ph_crop]) - (2 * torch.std(pred_ph_object[ph_crop:-ph_crop, ph_crop:-ph_crop]))
        vmax_ph = torch.mean(pred_ph_object[ph_crop:-ph_crop, ph_crop:-ph_crop]) + (2 * torch.std(pred_ph_object[ph_crop:-ph_crop, ph_crop:-ph_crop]))

        f, ax = plt.subplots(figsize=(9, 8), ncols=2, nrows=2)

        gt0 = ax[0, 0].imshow(gt_amp_object[180:-180, 180:-180], interpolation='none', vmin=0.9, vmax=1)
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(gt0, cax=cax, orientation='vertical')
        ax[0, 0].set_title('LSQML amplitude')

        gt1 = ax[0, 1].imshow(gt_ph_object[180:-180, 180:-180], interpolation='none', vmin=-1.3, vmax=1.3, cmap='magma')
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(gt1, cax=cax, orientation='vertical')
        ax[0, 1].set_title('LSQML phase')

        pred0 = ax[1, 0].imshow(pred_amp_object[180:-180, 180:-180], interpolation='none', vmin=0.9, vmax=1)
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(pred0, cax=cax, orientation='vertical')
        ax[1, 0].set_title('Predicted amplitude')

        pred1 = ax[1, 1].imshow(pred_ph_object[180:-180, 180:-180], interpolation='none', vmin=vmin_ph, vmax=vmax_ph, cmap='magma')
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(pred1, cax=cax, orientation='vertical')
        ax[1, 1].set_title('Predicted phase')

        plt.tight_layout()
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        f.savefig(os.path.join(run_path, filename), bbox_inches='tight', transparent=True)
        plt.close(f)

        if self.wandb_enabled:
            wandb.log({"test_plot": wandb.Image(os.path.join(run_path, filename), caption=f"Test: epoch {epoch}")}, step=epoch, commit=True)

    def train(self, dataloader, criterion, optimizer, metrics, profile=False, epoch=None):
        """
        Training loop.

        Args:
            dataloader: PyTorch DataLoader yielding batches of
                       (diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale[, meta])
        """
        running_loss = 0.0
        running_amp_loss = 0.0
        running_ph_loss = 0.0
        processed_batches = 0

        total_batches = len(dataloader)
        progress_milestones = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        milestone_batches = [int(total_batches * m) for m in progress_milestones]
        next_milestone_idx = 0


        if self.is_main_process:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training loop: {total_batches} total batches", flush=True)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading first batch... (this may take a while with lazy data loading)", flush=True)

        batch_0_io_start = time.time()
        train_end_time = None  # Will be set after each batch completes
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # adding some profiling
            if profile:
                if epoch == 1 and batch_idx == 0 and self.is_main_process:
                    torch.cuda.profiler.start()
                if batch_idx == 50:
                    if epoch == 1 and self.is_main_process:
                        torch.cuda.profiler.stop()
                    break

            torch.cuda.nvtx.range_push(f"step {batch_idx}")
            torch.cuda.nvtx.range_push(f"data copy in {batch_idx}")

            # Time IO (data loading) - this captures the time to get batch from DataLoader
            # The DataLoader fetch happens at the 'for' line above, so we time from end of previous batch
            if batch_idx < 10:
                if batch_idx == 0:
                    io_time = time.time() - batch_0_io_start
                else:
                    io_time = time.time() - train_end_time

            if batch_idx == 0 and self.is_main_process:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] First batch loaded! Starting training... (batch 1/{total_batches})", flush=True)
            elif batch_idx < 10 and self.is_main_process:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing batch {batch_idx + 1}/{total_batches}", flush=True)

            # ── Unpack ──────────────────────────────────────────────────────────
            if isinstance(batch, (list, tuple)) and len(batch) == 8:
                diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale, _meta = batch
            else:
                diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch

            # ── FIX: move required tensors to device right after unpack ─────────
            # Works for both normal DataLoader (CPU) and CUDAPrefetcher (already GPU)
            diff_amp  = diff_amp.to(self.device, non_blocking=True)
            amp_patch = amp_patch.to(self.device, non_blocking=True)
            ph_patch  = ph_patch.to(self.device, non_blocking=True)
            norm      = norm.to(self.device, non_blocking=True)
            scale     = scale.to(self.device, non_blocking=True)

            if batch_idx < 10:
                train_start_time = time.time()

            input_diff = diff_amp
            input_probe = torch.view_as_real(probe.clone().detach()).to(self.device, non_blocking=True)
            input_norm = norm
            input_scale = scale

            torch.cuda.nvtx.range_pop()  # copy in

            torch.cuda.nvtx.range_push(f"forward")

            output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)

            if self.mode == 'supervised':
                # amp_patch/ph_patch already on device
                loss = criterion(output_amp, amp_patch) + criterion(output_ph, ph_patch)
            else:
                loss = criterion(output_diff, input_diff)

            torch.cuda.nvtx.range_pop()  # forward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            total_norm = None
            global_max_norm = None
            need_grad_norm = (
                self.debug_mode
                or (self.skip_batch_if_grad_norm_greater_than is not None)
            )
            if need_grad_norm:
                total_norm_sq = 0.0
                for param in self.model.parameters():
                    if param.grad is None:
                        continue
                    grad_norm = param.grad.detach().float().norm(2)
                    total_norm_sq += float(grad_norm) ** 2
                total_norm = total_norm_sq ** 0.5
                global_max_norm = total_norm

                if self.use_ddp and dist.is_initialized() and self.skip_batch_if_grad_norm_greater_than is not None:
                    norm_tensor = torch.tensor(total_norm, device=self.device)
                    dist.all_reduce(norm_tensor, op=dist.ReduceOp.MAX)
                    global_max_norm = float(norm_tensor.item())

                if self.debug_mode and self.is_main_process and self.wandb_enabled:
                    wandb.log({"grad_norm": global_max_norm})

            skip_threshold = self.skip_batch_if_grad_norm_greater_than
            if skip_threshold is not None and global_max_norm is not None and global_max_norm > skip_threshold:
                if self.is_main_process:
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"[Skip Batch] Grad norm {global_max_norm:.2f} > {skip_threshold} "
                        f"(batch {batch_idx + 1}/{total_batches})",
                        flush=True
                    )
            else:
                torch.cuda.nvtx.range_push(f"optimizer")
                optimizer.step()
                torch.cuda.nvtx.range_pop()  # optimizer
                processed_batches += 1

            train_end_time = time.time()

            if batch_idx < 10:
                train_time = train_end_time - train_start_time
                total_time = io_time + train_time
                if self.is_main_process:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Batch {batch_idx + 1} Timing] IO: {io_time:.3f}s | Training: {train_time:.3f}s | Total: {total_time:.3f}s", flush=True)

            running_loss += loss.detach().item()

            # Track amp/phase losses (targets now guaranteed on same device)
            loss_amp = criterion(output_amp.detach(), amp_patch)
            loss_ph  = criterion(output_ph.detach(), ph_patch)
            running_amp_loss += loss_amp.item()
            running_ph_loss  += loss_ph.item()

            if batch_idx > 0 and self.is_main_process:
                if batch_idx < 1000 and (batch_idx + 1) % 1000 == 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Training] Batch {batch_idx + 1}/{total_batches} ({(batch_idx + 1) * 100.0 / total_batches:.2f}%)", flush=True)
                elif (batch_idx + 1) % 1000 == 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Training] Batch {batch_idx + 1}/{total_batches} ({(batch_idx + 1) * 100.0 / total_batches:.2f}%)", flush=True)

            if next_milestone_idx < len(milestone_batches) and batch_idx >= milestone_batches[next_milestone_idx]:
                progress_pct = int(progress_milestones[next_milestone_idx] * 100)
                if self.is_main_process:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Training Progress] {progress_pct}% complete ({batch_idx + 1}/{total_batches} batches)", flush=True)
                next_milestone_idx += 1


            torch.cuda.nvtx.range_pop()  # step

        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        print(f"Processed {processed_batches} batches", flush=True)
        num_batches = processed_batches
        if num_batches == 0:
            if self.is_main_process:
                print("[Warning] No training batches were processed.", flush=True)
            avg_train_loss = float("inf")
            avg_amp_loss = float("inf")
            avg_ph_loss = float("inf")
        else:
            avg_train_loss = running_loss / num_batches
            avg_amp_loss = running_amp_loss / num_batches
            avg_ph_loss = running_ph_loss / num_batches

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        if self.is_main_process:
            #print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training epoch complete: {num_batches} batches processed", flush=True)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training epoch complete: {num_batches} batches processed in {epoch_time:.3f} seconds", flush=True)

        avg_train_loss = self.synchronize_loss(avg_train_loss)
        avg_amp_loss = self.synchronize_loss(avg_amp_loss)
        avg_ph_loss = self.synchronize_loss(avg_ph_loss)

        if self.is_main_process and self.wandb_enabled:
            wandb.log({"train_loss": avg_train_loss}, step=epoch)
            wandb.log({"train_amp_loss": avg_amp_loss}, step=epoch)
            wandb.log({"train_ph_loss": avg_ph_loss}, step=epoch)

        metrics['training_loss'].append(avg_train_loss)
        metrics['train_amp_loss'].append(avg_amp_loss)
        metrics['train_ph_loss'].append(avg_ph_loss)

    def validate(self, dataloader, criterion, optimizer, metrics, plot=False, epoch=0, scheduler=None, profile=False):
        """
        Validation loop with SSIM and PSNR metrics.
        """
        if self.is_main_process:
            from datetime import datetime
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting validation loop: {len(dataloader)} batches", flush=True)

        val_loss = 0.0
        val_amp_loss = 0.0
        val_ph_loss = 0.0

        # SSIM and PSNR accumulators
        val_amp_ssim = 0.0
        val_amp_psnr = 0.0
        val_ph_ssim = 0.0
        val_ph_psnr = 0.0
        num_samples = 0
        processed_batches = 0

        total_batches = len(dataloader)
        progress_milestones = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        milestone_batches = [int(m * total_batches) for m in progress_milestones]
        next_milestone_idx = 0

        last_input_diff = None
        last_output_diff = None
        last_amp_patch = None
        last_output_amp = None
        last_ph_patch = None
        last_output_ph = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if profile:
                    if batch_idx == 50:
                        # early stop if profiling
                        break
                if isinstance(batch, (list, tuple)) and len(batch) == 8:
                    diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale, _meta = batch
                else:
                    diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch

                # ── FIX: move required tensors to device right after unpack ─────────
                diff_amp  = diff_amp.to(self.device, non_blocking=True)
                amp_patch = amp_patch.to(self.device, non_blocking=True)
                ph_patch  = ph_patch.to(self.device, non_blocking=True)
                norm      = norm.to(self.device, non_blocking=True)
                scale     = scale.to(self.device, non_blocking=True)

                input_diff = diff_amp
                input_probe = torch.view_as_real(probe.clone().detach()).to(self.device, non_blocking=True)
                input_norm = norm
                input_scale = scale

                output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)

                if self.mode == 'supervised':
                    loss = criterion(output_amp, amp_patch) + criterion(output_ph, ph_patch)
                else:
                    loss = criterion(output_diff, input_diff)

                val_loss += loss.detach().item()

                loss_amp = criterion(output_amp.detach(), amp_patch)
                loss_ph  = criterion(output_ph.detach(), ph_patch)
                val_amp_loss += loss_amp.item()
                val_ph_loss += loss_ph.item()
                processed_batches += 1

                # SSIM/PSNR metrics (vectorized; average over samples)
                batch_size = output_amp.size(0)

                pred_amp = output_amp.detach()
                pred_ph = output_ph.detach()

                batch_amp_ssim = compute_ssim(pred_amp, amp_patch)
                batch_ph_ssim = compute_ssim(pred_ph, ph_patch)
                val_amp_ssim += batch_amp_ssim * batch_size
                val_ph_ssim += batch_ph_ssim * batch_size

                # PSNR: compute per-sample PSNR then sum
                pred_amp_flat = pred_amp.flatten(1)
                amp_flat = amp_patch.flatten(1)
                amp_mse = (pred_amp_flat - amp_flat).pow(2).mean(dim=1)
                amp_range = (amp_flat.amax(dim=1) - amp_flat.amin(dim=1)).clamp(min=1e-8)
                amp_psnr = 10 * torch.log10((amp_range ** 2) / (amp_mse + 1e-8))
                val_amp_psnr += amp_psnr.sum().item()

                pred_ph_flat = pred_ph.flatten(1)
                ph_flat = ph_patch.flatten(1)
                ph_mse = (pred_ph_flat - ph_flat).pow(2).mean(dim=1)
                ph_range = (ph_flat.amax(dim=1) - ph_flat.amin(dim=1)).clamp(min=1e-8)
                ph_psnr = 10 * torch.log10((ph_range ** 2) / (ph_mse + 1e-8))
                val_ph_psnr += ph_psnr.sum().item()

                num_samples += batch_size

                if next_milestone_idx < len(milestone_batches) and batch_idx >= milestone_batches[next_milestone_idx]:
                    from datetime import datetime
                    progress_pct = int(progress_milestones[next_milestone_idx] * 100)
                    if self.is_main_process:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Validation Progress] {progress_pct}% complete ({batch_idx + 1}/{total_batches} batches)", flush=True)
                    next_milestone_idx += 1

                if plot and self.is_main_process:
                    last_input_diff = input_diff
                    last_output_diff = output_diff
                    last_amp_patch = amp_patch
                    last_output_amp = output_amp
                    last_ph_patch = ph_patch
                    last_output_ph = output_ph

        num_batches = processed_batches
        if num_batches == 0:
            if self.is_main_process:
                print("[Warning] No validation batches were processed.", flush=True)
            avg_val_loss = float("inf")
            avg_val_amp_loss = float("inf")
            avg_val_ph_loss = float("inf")
        else:
            avg_val_loss = val_loss / num_batches
            avg_val_amp_loss = val_amp_loss / num_batches
            avg_val_ph_loss = val_ph_loss / num_batches
        
        # Average SSIM and PSNR
        avg_amp_ssim = val_amp_ssim / num_samples if num_samples > 0 else 0.0
        avg_amp_psnr = val_amp_psnr / num_samples if num_samples > 0 else 0.0
        avg_ph_ssim = val_ph_ssim / num_samples if num_samples > 0 else 0.0
        avg_ph_psnr = val_ph_psnr / num_samples if num_samples > 0 else 0.0

        if self.is_main_process:
            from datetime import datetime
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation epoch complete: {num_batches} batches processed", flush=True)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Metrics - Amp SSIM: {avg_amp_ssim:.4f}, Amp PSNR: {avg_amp_psnr:.2f} dB, Phase SSIM: {avg_ph_ssim:.4f}, Phase PSNR: {avg_ph_psnr:.2f} dB", flush=True)

        avg_val_loss = self.synchronize_loss(avg_val_loss)
        avg_val_amp_loss = self.synchronize_loss(avg_val_amp_loss)
        avg_val_ph_loss = self.synchronize_loss(avg_val_ph_loss)
        avg_amp_ssim = self.synchronize_loss(avg_amp_ssim)
        avg_amp_psnr = self.synchronize_loss(avg_amp_psnr)
        avg_ph_ssim = self.synchronize_loss(avg_ph_ssim)
        avg_ph_psnr = self.synchronize_loss(avg_ph_psnr)

        if self.is_main_process and self.wandb_enabled:
            wandb.log({"val_loss": avg_val_loss}, step=epoch)
            wandb.log({"val_amp_loss": avg_val_amp_loss}, step=epoch)
            wandb.log({"val_ph_loss": avg_val_ph_loss}, step=epoch)
            wandb.log({"val_amp_ssim": avg_amp_ssim}, step=epoch)
            wandb.log({"val_amp_psnr": avg_amp_psnr}, step=epoch)
            wandb.log({"val_ph_ssim": avg_ph_ssim}, step=epoch)
            wandb.log({"val_ph_psnr": avg_ph_psnr}, step=epoch)

        metrics['validation_loss'].append(avg_val_loss)
        metrics['val_amp_loss'].append(avg_val_amp_loss)
        metrics['val_ph_loss'].append(avg_val_ph_loss)

        # Add SSIM/PSNR to metrics (initialize lists if not present)
        if 'val_amp_ssim' not in metrics:
            metrics['val_amp_ssim'] = []
        if 'val_amp_psnr' not in metrics:
            metrics['val_amp_psnr'] = []
        if 'val_ph_ssim' not in metrics:
            metrics['val_ph_ssim'] = []
        if 'val_ph_psnr' not in metrics:
            metrics['val_ph_psnr'] = []

        metrics['val_amp_ssim'].append(avg_amp_ssim)
        metrics['val_amp_psnr'].append(avg_amp_psnr)
        metrics['val_ph_ssim'].append(avg_ph_ssim)
        metrics['val_ph_psnr'].append(avg_ph_psnr)


        if plot and self.is_main_process and last_input_diff is not None:
            input_diff_np = last_input_diff[0, 0].detach().cpu().numpy()
            output_diff_np = last_output_diff[0, 0].detach().cpu().numpy()

            input_amp = last_amp_patch[0, 0].detach().cpu()
            output_amp = last_output_amp[0, 0].detach().cpu().numpy()

            input_ph = last_ph_patch[0, 0].detach().cpu()
            output_ph = last_output_ph[0, 0].detach().cpu().numpy()

            filename = 'plot_epoch' + str(epoch) + '.png'
            self.generate_plot(input_diff_np, output_diff_np, input_amp, output_amp, input_ph, output_ph, filename)

            if self.wandb_enabled:
                run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
                wandb.log({"val_plot": wandb.Image(os.path.join(run_path, filename), caption=f"Epoch {epoch}")}, step=epoch)

        if scheduler:
            # Only ReduceLROnPlateau expects a monitored metric; others use epoch progression.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
            if 'lr' not in metrics:
                metrics['lr'] = []
            metrics['lr'].append(optimizer.param_groups[0]['lr'])
            if self.is_main_process and self.wandb_enabled:
                wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=epoch)

        if avg_val_loss < metrics['best_val_loss']:
            if self.is_main_process:
                print("Saving improved model after Val. Loss improved from %.4f to %.5f"
                      % (metrics['best_val_loss'], avg_val_loss), flush=True)
                self.update_saved_model('best_model')
            metrics['best_val_loss'] = avg_val_loss

        # Synchronize all processes after validation and potential model saving
        # This prevents CUDA/HDF5 conflicts when other ranks continue while rank 0 saves
        if self.use_ddp and dist.is_initialized():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            dist.barrier()
