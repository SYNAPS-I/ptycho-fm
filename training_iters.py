import os
import shutil
import csv
import numpy as np
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
from utils.utils import get_norm
from utils.flops_utils import AnalyticalFlopsForTrainer
from utils.distributed import ddp_barrier
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
        criterion=None,
        optimizer=None,
        is_main_process=True, 
        use_ddp=False, 
        wandb_enabled=True, 
        wandb_run_id=None,
        scheduler=None,
        debug_mode=False,
        log_every=100,
        log_at_flops=None,
        max_iters=100000,
        flops_calcs: AnalyticalFlopsForTrainer | None = None,
    ):
        super().__init__()
        self.model = model
        self.mode = mode
        self.run_num = run_num
        self.device = device
        self.model_save_path = model_save_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.is_main_process = is_main_process
        self.use_ddp = use_ddp
        self.wandb_enabled = wandb_enabled
        self.debug_mode = debug_mode
        self.wandb_run_id = wandb_run_id
        self.scheduler = scheduler
        self.iters = 0
        self.epoch = 0
        self.iters_in_epoch = 0
        self.samples_seen_in_epoch = 0
        self.iters_since_last_log = 0
        self.log_every = log_every
        self.log_at_tflops = sorted(float(v) / 1e12 for v in (log_at_flops or []))
        self.next_flops_log_idx = 0
        self.max_iters = max_iters
        self.flops_calcs = flops_calcs
        if self.log_at_tflops and self.flops_calcs is None:
            raise ValueError("training.log_at_flops requires FLOPs calculations, but flops_calcs is not available.")
        self.logs = {}
        self.metrics = {
            'training_loss': [],
            'train_amp_loss': [],
            'train_ph_loss': [],
            'grad_norm': [],
            'validation_loss': [],
            'val_amp_loss': [],
            'val_ph_loss': [],
            'val_amp_ssim': [],
            'val_amp_psnr': [],
            'val_ph_ssim': [],
            'val_ph_psnr': [],
            'best_val_loss': np.inf,
        }
        self.running_loss = 0.0
        self.running_amp_loss = 0.0
        self.running_ph_loss = 0.0
        self.grad = 0.0

    def set_dataloaders(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def synchronize_loss(self, loss_value):
        """Synchronize loss across all processes in DDP."""
        if self.use_ddp and dist.is_initialized():
            loss_tensor = torch.tensor(loss_value, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= dist.get_world_size()
            return loss_tensor.item()
        return loss_value

    def log_memory_usage(self):
        """Logs the memory usage of the GPU"""
        if self.is_main_process:
            max_reserved_gb = torch.cuda.max_memory_reserved(device=self.device) / (
                1024.0 * 1024.0 * 1024.0
            )
            max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (
                1024.0 * 1024.0 * 1024.0
            )
            print(f"Memory usage: Max allocated: {max_mem_gb} GB, Max reserved: {max_reserved_gb} GB", flush=True)

    def save_config(self, config_path='config.yaml'):
        """Save a copy of the config file to the run path for reproducibility."""
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        if not os.path.isdir(run_path):
            os.makedirs(run_path, exist_ok=True)
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(run_path, 'config.yaml'))

    def _logs_path(self):
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        os.makedirs(run_path, exist_ok=True)
        return os.path.join(run_path, "logs.txt")

    def _log_value(self, value):
        if torch.is_tensor(value):
            value = value.detach().cpu()
            return value.item() if value.numel() == 1 else value.tolist()
        if hasattr(value, "item"):
            return value.item()
        return value

    def _tflops_consumed(self):
        fc = self.flops_calcs
        if fc is None:
            return None

        per_forward_unit_tflops = fc.calculator.flops_analytical()
        return (
            per_forward_unit_tflops
            * fc.train_batch_size
            * fc.world_size
            * self.iters
            * 3
        ) # 3 for fwd + bwd

    def _sync_flops_log_index_to_current(self):
        if not self.log_at_tflops:
            return

        consumed_tflops = self._tflops_consumed()
        while (
            self.next_flops_log_idx < len(self.log_at_tflops)
            and consumed_tflops >= self.log_at_tflops[self.next_flops_log_idx]
        ):
            self.next_flops_log_idx += 1

    def _should_log_this_iter(self):
        if not self.log_at_tflops:
            return self.iters_since_last_log % self.log_every == 0

        consumed_tflops = self._tflops_consumed()
        if self.next_flops_log_idx >= len(self.log_at_tflops):
            return False
        if consumed_tflops < self.log_at_tflops[self.next_flops_log_idx]:
            return False

        while (
            self.next_flops_log_idx < len(self.log_at_tflops)
            and consumed_tflops >= self.log_at_tflops[self.next_flops_log_idx]
        ):
            self.next_flops_log_idx += 1
        return True

    def write_logs(self, logs=None):
        """Append scalar logs to logs.txt so completed runs can be queried offline."""
        if not self.is_main_process:
            return

        logs = self.logs if logs is None else logs
        if not logs:
            return

        log_file = self._logs_path()
        row = {"iter": self.iters}
        row.update({k: self._log_value(v) for k, v in logs.items()})

        file_exists = os.path.exists(log_file) and os.path.getsize(log_file) > 0
        if not file_exists:
            with open(log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            return

        with open(log_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            new_fields = [k for k in row.keys() if k not in fieldnames]
            if new_fields:
                existing_rows = list(reader)
            else:
                existing_rows = None

        if new_fields:
            fieldnames.extend(new_fields)
            with open(log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(row)
        else:
            with open(log_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

    def generate_state_dict(self, wandb_run_id=None, resume_epoch=None, resume_iters_in_epoch=None, resume_samples_in_epoch=None):
        """Returns training state (not model weights). Resume fields default to live trainer counters."""
        state = {
            'iters': self.iters,
            'epoch': self.epoch if resume_epoch is None else resume_epoch,
            'iters_in_epoch': self.iters_in_epoch if resume_iters_in_epoch is None else resume_iters_in_epoch,
            'samples_seen_in_epoch': self.samples_seen_in_epoch if resume_samples_in_epoch is None else resume_samples_in_epoch,
            'loss_tracker': self.metrics,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'wandb_run_id': wandb_run_id,
        }
        return state

    def _save_model_weights(self, run_path: str, filename: str):
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            sd = self.model.module.state_dict()
        else:
            sd = self.model.state_dict()
        _atomic_torch_save(_move_to_cpu(sd), os.path.join(run_path, filename))

    def checkpoint(
        self,
        wandb_run_id=None,
        is_best=False,
        resume_epoch=None,
        resume_iters_in_epoch=None,
        resume_samples_in_epoch=None,
    ):
        """Save model weights (torch.save) and training state for resume."""
        if not self.is_main_process:
            return

        wb_id = self.wandb_run_id if wandb_run_id is None else wandb_run_id
        state_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        os.makedirs(state_path, exist_ok=True)

        self._save_model_weights(state_path, 'checkpoint_model.pth')
        self._save_model_weights(state_path, f'model_iters_{self.iters}.pth')
        if is_best:
            self._save_model_weights(state_path, 'best_model.pth')

        cpu_state = _move_to_cpu(
            self.generate_state_dict(
                wb_id,
                resume_epoch=resume_epoch,
                resume_iters_in_epoch=resume_iters_in_epoch,
                resume_samples_in_epoch=resume_samples_in_epoch,
            )
        )
        # checkpoint.state: always-overwritten latest state for normal resume.
        # state_iters_N.pth: per-iteration archive so any checkpoint can be branched from.
        _atomic_torch_save(cpu_state, os.path.join(state_path, 'checkpoint.state'))
        _atomic_torch_save(cpu_state, os.path.join(state_path, f'state_iters_{self.iters}.pth'))

    def load_state_checkpoint(self, skip_scheduler=False, checkpoint_file=None):
        """Load optimizer/scheduler/metrics and resume counters. Model weights loaded separately in main.

        Args:
            skip_scheduler: If True, do not restore the scheduler state (e.g. cooldown runs that
                            start a fresh LR schedule from an existing optimizer state).
            checkpoint_file: Full path to a specific state file (e.g. state_iters_N.pth).
                             Defaults to checkpoint.state in the run's own save directory.
        """
        if checkpoint_file is not None:
            checkpoint_fname = checkpoint_file
        else:
            checkpoint_fname = os.path.join(
                self.model_save_path, 'run' + str(self.run_num), 'checkpoint.state'
            )
        if not os.path.exists(checkpoint_fname):
            raise FileNotFoundError(f"Checkpoint not found in {checkpoint_fname}")
        state_dict = torch.load(checkpoint_fname, map_location='cpu')
        self.iters = state_dict['iters']
        self.epoch = state_dict.get('epoch', 0)
        self.iters_in_epoch = state_dict['iters_in_epoch']
        self.samples_seen_in_epoch = state_dict.get('samples_seen_in_epoch', 0)
        self.metrics = state_dict['loss_tracker']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device, non_blocking=True)

        if not skip_scheduler and state_dict['scheduler_state_dict'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        wandb_run_id = state_dict.get('wandb_run_id', None)
        return self.epoch, wandb_run_id

    def save_final(self, wandb_run_id=None):
        """Save last model + state at end of training (rank 0 only)."""
        if not self.is_main_process:
            return
        wb = self.wandb_run_id if wandb_run_id is None else wandb_run_id
        self.checkpoint(
            wandb_run_id=wb,
            is_best=False,
            resume_epoch=self.epoch + 1,
            resume_iters_in_epoch=0,
            resume_samples_in_epoch=0,
        )

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
        plt.close(f)

    def generate_test_plot(self, dataloader, filename, central_crop=64, ph_crop=180):
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
            wandb.log({"test_plot": wandb.Image(os.path.join(run_path, filename), caption=f"Test: iters {self.iters}")}, step=self.iters, commit=True)

    def train(self, profile=False, epoch=None):
        """
        Training loop.

        Args:
            profile: bool, whether to profile the training loop
            epoch: int, the current epoch
        """
        self.model.train()
        if epoch is not None:
            self.epoch = epoch
        dataloader = self.train_loader
        criterion = self.criterion
        optimizer = self.optimizer
        total_batches = len(dataloader)
        self._sync_flops_log_index_to_current()

        if self.is_main_process:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training loop: {total_batches} total batches", flush=True)

        epoch_start_time = time.time()
        self.train_time = time.time()
        epoch_completed = True
        processed_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if self.iters >= self.max_iters:
                if self.is_main_process:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Reached max iterations ({self.max_iters})", flush=True)
                if self.iters_since_last_log > 0 and not self.log_at_tflops:
                    self.validate()
                epoch_completed = False
                break
            # adding some profiling
            if profile:
                if epoch == 1 and batch_idx == 0 and self.is_main_process:
                    torch.cuda.profiler.start()
                if batch_idx == 50:
                    if epoch == 1 and self.is_main_process:
                        torch.cuda.profiler.stop()
                    epoch_completed = False
                    break

            torch.cuda.nvtx.range_push(f"step {batch_idx}")
            torch.cuda.nvtx.range_push(f"data copy in {batch_idx}")

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

            input_diff = diff_amp
            input_probe = torch.view_as_real(probe.clone().detach()).to(self.device, non_blocking=True)
            input_norm = norm
            input_scale = scale

            torch.cuda.nvtx.range_pop()  # copy in

            torch.cuda.nvtx.range_push("forward")

            output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)

            if self.mode == 'supervised':
                # amp_patch/ph_patch already on device
                loss = criterion(output_amp, amp_patch) + criterion(output_ph, ph_patch)
            else:
                loss = criterion(output_diff, input_diff)

            torch.cuda.nvtx.range_pop()  # forward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            self.grad += get_norm(self.model.parameters())

            torch.cuda.nvtx.range_push("optimizer")
            optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            torch.cuda.nvtx.range_pop()  # optimizer
            processed_batches += 1

            batch_size = diff_amp.size(0)
            self.iters += 1
            self.iters_in_epoch += 1
            self.iters_since_last_log += 1
            self.samples_seen_in_epoch += batch_size

            self.running_loss += loss.detach().item()
            loss_amp = criterion(output_amp.detach(), amp_patch)
            loss_ph  = criterion(output_ph.detach(), ph_patch)
            self.running_amp_loss += loss_amp.item()
            self.running_ph_loss  += loss_ph.item()


            torch.cuda.nvtx.range_pop()  # step

            if self._should_log_this_iter():
                at_boundary = (batch_idx + 1) >= total_batches
                self.validate_and_log(profile=profile, at_epoch_boundary=at_boundary)
                if torch.distributed.is_initialized():
                    ddp_barrier(torch.distributed.get_world_size(), self.device)

        if epoch_completed:
            self.iters_in_epoch = 0
            self.samples_seen_in_epoch = 0
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        if self.is_main_process:
            print(
                f"########################################################",
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ",
                f"Training epoch complete: {processed_batches} batches processed in ",
                f"{epoch_time:.3f} seconds",
                f"########################################################",
                flush=True,
            )
        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings


    def validate_and_log(self, profile=False, at_epoch_boundary=False):
        """
        Validate and log metrics to WandB.
        """
        # log some train metrics
        avg_train_loss = self.running_loss / self.iters_since_last_log
        avg_amp_loss = self.running_amp_loss / self.iters_since_last_log
        avg_ph_loss = self.running_ph_loss / self.iters_since_last_log
        avg_grad_norm = self.grad / self.iters_since_last_log
        avg_train_loss = self.synchronize_loss(avg_train_loss)
        avg_amp_loss = self.synchronize_loss(avg_amp_loss)
        avg_ph_loss = self.synchronize_loss(avg_ph_loss)
        avg_grad_norm = self.synchronize_loss(avg_grad_norm)
        self.metrics['training_loss'].append(avg_train_loss)
        self.metrics['train_amp_loss'].append(avg_amp_loss)
        self.metrics['train_ph_loss'].append(avg_ph_loss)
        self.metrics['grad_norm'].append(avg_grad_norm)

        if self.is_main_process:
            log_payload = {
                "train_loss": avg_train_loss,
                "train_amp_loss": avg_amp_loss,
                "train_ph_loss": avg_ph_loss,
                "grad_norm": avg_grad_norm,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            tflops_consumed = self._tflops_consumed()
            if tflops_consumed is not None:
                log_payload["tflops_consumed"] = tflops_consumed
            self.logs = log_payload
            if self.wandb_enabled:
                wandb.log(log_payload, step=self.iters)

        if self.is_main_process:
            print(
                f"########################################################",
                flush=True,
            )
            self.log_memory_usage()
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Training iters {self.iters} complete: {self.iters_since_last_log} iters processed in "
                f"{time.time() - self.train_time:.3f} seconds",
                flush=True,
            )
            print(
                f"Train loss: {avg_train_loss:.4f}, ",
                f"Train amp loss: {avg_amp_loss:.4f}, ",
                f"Train ph loss: {avg_ph_loss:.4f}",
                flush=True,
            )

        self.running_loss = self.running_amp_loss = self.running_ph_loss = 0.0
        self.grad = 0.0
        self.iters_since_last_log = 0

        self.val_time = time.time()
        self.validate(plot=True, at_epoch_boundary=at_epoch_boundary)

        # Generate test plot
        if self.is_main_process and (not profile) and self.test_loader is not None:
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation complete in "
                f"{time.time() - self.val_time:.3f} seconds",
                flush=True,
            )
            self.log_memory_usage()
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating test plot...", flush=True)
            test_time = time.time()
            self.generate_test_plot(
                self.test_loader,
                'test_iters' + str(self.iters) + '.png',
            )
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Test plot generated in "
                f"{time.time() - test_time:.3f} seconds",
                flush=True,
            )
            print(
                f"########################################################",
                flush=True,
            )
        self.train_time = time.time() # reset train time


    def validate(self, plot=False, profile=False, at_epoch_boundary=False):
        """
        Validation loop with SSIM and PSNR metrics.
        """
        dataloader = self.val_loader
        criterion = self.criterion
        self.model.eval()

        if self.is_main_process:
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
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation epoch complete: {num_batches} batches processed", flush=True)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Metrics - Amp SSIM: {avg_amp_ssim:.4f}, Amp PSNR: {avg_amp_psnr:.2f} dB, Phase SSIM: {avg_ph_ssim:.4f}, Phase PSNR: {avg_ph_psnr:.2f} dB", flush=True)

        avg_val_loss = self.synchronize_loss(avg_val_loss)
        avg_val_amp_loss = self.synchronize_loss(avg_val_amp_loss)
        avg_val_ph_loss = self.synchronize_loss(avg_val_ph_loss)
        avg_amp_ssim = self.synchronize_loss(avg_amp_ssim)
        avg_amp_psnr = self.synchronize_loss(avg_amp_psnr)
        avg_ph_ssim = self.synchronize_loss(avg_ph_ssim)
        avg_ph_psnr = self.synchronize_loss(avg_ph_psnr)

        val_payload = {
            "val_loss": avg_val_loss,
            "val_amp_loss": avg_val_amp_loss,
            "val_ph_loss": avg_val_ph_loss,
            "val_amp_ssim": avg_amp_ssim,
            "val_amp_psnr": avg_amp_psnr,
            "val_ph_ssim": avg_ph_ssim,
            "val_ph_psnr": avg_ph_psnr,
        }

        if self.is_main_process:
            if self.wandb_enabled:
                wandb.log(val_payload, step=self.iters)
            self.write_logs({**self.logs, **val_payload})
            self.logs = {}

        self.metrics['validation_loss'].append(avg_val_loss)
        self.metrics['val_amp_loss'].append(avg_val_amp_loss)
        self.metrics['val_ph_loss'].append(avg_val_ph_loss)
        self.metrics['val_amp_ssim'].append(avg_amp_ssim)
        self.metrics['val_amp_psnr'].append(avg_amp_psnr)
        self.metrics['val_ph_ssim'].append(avg_ph_ssim)
        self.metrics['val_ph_psnr'].append(avg_ph_psnr)


        if plot and self.is_main_process and last_input_diff is not None:
            input_diff_np = last_input_diff[0, 0].detach().cpu().numpy()
            output_diff_np = last_output_diff[0, 0].detach().cpu().numpy()

            input_amp = last_amp_patch[0, 0].detach().cpu()
            output_amp = last_output_amp[0, 0].detach().cpu().numpy()

            input_ph = last_ph_patch[0, 0].detach().cpu()
            output_ph = last_output_ph[0, 0].detach().cpu().numpy()

            filename = 'plot_iters' + str(self.iters) + '.png'
            self.generate_plot(input_diff_np, output_diff_np, input_amp, output_amp, input_ph, output_ph, filename)

            if self.wandb_enabled:
                run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
                wandb.log({"val_plot": wandb.Image(os.path.join(run_path, filename), caption=f"Iters {self.iters}")}, step=self.iters)

        if avg_val_loss < self.metrics['best_val_loss']:
            if self.is_main_process:
                print("Saving improved model after Val. Loss improved from %.4f to %.5f"
                      % (self.metrics['best_val_loss'], avg_val_loss), flush=True)
            self.metrics['best_val_loss'] = avg_val_loss
            if at_epoch_boundary:
                self.checkpoint(
                    is_best=True,
                    resume_epoch=self.epoch + 1,
                    resume_iters_in_epoch=0,
                    resume_samples_in_epoch=0,
                )
            else:
                self.checkpoint(is_best=True)
        else:
            if self.is_main_process:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving checkpoint for current iteration", flush=True)
            if at_epoch_boundary:
                self.checkpoint(
                    is_best=False,
                    resume_epoch=self.epoch + 1,
                    resume_iters_in_epoch=0,
                    resume_samples_in_epoch=0,
                )
            else:
                self.checkpoint(is_best=False)

        # Synchronize all processes after validation and potential model saving
        # This prevents CUDA/HDF5 conflicts when other ranks continue while rank 0 saves
        if self.use_ddp and dist.is_initialized():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                dist.barrier(device_ids=[self.device.index])
            else:
                dist.barrier()
        self.model.train() # back to training
